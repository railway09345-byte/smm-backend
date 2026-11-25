"""Durable daily campaign queue on the existing pipeline tables.

Each network stage is checkpointed. Atomic leases fence concurrent workers;
video operation IDs survive restarts. Only verified stored media is completed.
"""
import asyncio
import hashlib
import io
import json
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from sqlalchemy import or_, and_
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
from PIL import Image, ImageOps
from app.core.config import settings
from app.db.session import SessionLocal
from app.models.organization import Organization
from app.models.brand_profile import BrandProfile
from app.models.autonomous_pipeline import BusinessSchedule, PipelineRun, ContentBrief, GeneratedAsset, AutonomousPost, PostVariant
from app.services.studio_brand_context import studio_brand_context
from app.services.daily_ai_provider import DailyAIProvider, DailyCampaign, ProviderFailure
from app.integrations.storage.cloudinary_storage import CloudinaryStorageProvider


def utcnow():
    return datetime.utcnow()


def media_counts(schedule):
    ct = (getattr(schedule, "content_type", None) or "both").lower().strip()
    is_video_only = ct in ("video", "videos")
    is_image_only = ct in ("image", "images")

    raw_images = schedule.image_count if schedule.image_count is not None else 1
    raw_videos = schedule.video_count if schedule.video_count is not None else 1

    images = 0 if is_video_only else (max(1, raw_images) if raw_images > 0 else 1)
    videos = 0 if is_image_only else (max(1, raw_videos) if raw_videos > 0 else 1)

    if not (0 <= images <= 4 and 0 <= videos <= 2 and images + videos > 0):
        raise HTTPException(400, "Choose 1–4 images and/or 1–2 videos per campaign.")
    return images, videos


def enqueue_daily(db, organization_id, brand_id, target_date=None):
    # Serialize reservations per tenant so concurrent brands cannot exceed quota.
    org = db.query(Organization).filter(Organization.id == organization_id, Organization.is_active.is_(True)).with_for_update().first()
    brand = db.query(BrandProfile).filter(BrandProfile.id == brand_id, BrandProfile.organization_id == organization_id).first()
    if not org or not brand:
        raise HTTPException(404, "Brand not found in your organization.")
    snapshot_text = studio_brand_context(brand)
    snapshot = json.loads(snapshot_text)
    schedule = db.query(BusinessSchedule).filter(BusinessSchedule.brand_profile_id == brand.id).first()
    if not schedule:
        raise HTTPException(400, "Save the daily generation schedule first.")
    if not schedule.is_active:
        raise HTTPException(400, "Daily generation is disabled for this brand.")
    local_date = target_date or datetime.now(ZoneInfo(schedule.timezone)).date()
    if schedule.generation_date and local_date.isoformat() < schedule.generation_date:
        raise HTTPException(400, f"Automation is scheduled to start on {schedule.generation_date}. Generation before start date is disabled.")
    existing = db.query(PipelineRun).filter(PipelineRun.brand_profile_id == brand_id, PipelineRun.run_date == local_date).first()
    if existing:
        db.rollback()
        return existing
    images, videos = media_counts(schedule)
    month_start = local_date.replace(day=1)
    runs = db.query(PipelineRun).filter(PipelineRun.organization_id == organization_id, PipelineRun.run_date >= month_start).all()
    reserved = sum((r.stats or {}).get("reserved_assets", 0) for r in runs)
    if reserved + images + videos > settings.DAILY_MONTHLY_ASSET_LIMIT:
        raise HTTPException(429, "Monthly daily-generation asset allowance reached.")
    config = {"image_count": images, "video_count": videos, "video_seconds": settings.DAILY_VIDEO_SECONDS,
        "platforms": schedule.target_platforms, "preferences": schedule.content_preferences,
        "posting_time": schedule.posting_time, "timezone": schedule.timezone,
        "auto_approve": schedule.auto_approve, "auto_schedule": schedule.auto_schedule,
        "campaign_date": local_date.isoformat()}
    run = PipelineRun(brand_profile_id=brand_id, organization_id=organization_id, run_date=local_date,
        status="pending", stage="Queued", stats={"engine": "daily_v2", "brand_snapshot": snapshot,
            "brand_version": hashlib.sha256(snapshot_text.encode()).hexdigest(), "config": config,
            "reserved_assets": images + videos, "provider_calls": 0})
    db.add(run)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return db.query(PipelineRun).filter(PipelineRun.brand_profile_id == brand_id, PipelineRun.run_date == local_date).one()
    db.refresh(run)
    return run


def claim_run(db, run_id):
    now = utcnow()
    token = uuid.uuid4().hex
    count = db.query(PipelineRun).filter(PipelineRun.id == run_id, PipelineRun.attempts < settings.DAILY_MAX_ATTEMPTS,
        or_(PipelineRun.next_attempt_at.is_(None), PipelineRun.next_attempt_at <= now),
        or_(PipelineRun.status.in_(["pending", "retry_pending"]),
            and_(PipelineRun.status == "running", PipelineRun.lease_expires_at < now))).update({
                "status": "running", "lease_token": token, "lease_expires_at": now + timedelta(seconds=settings.DAILY_LEASE_SECONDS),
                "started_at": now}, synchronize_session=False)
    db.commit()
    return token if count == 1 else None


def assert_lease(db, run_id, token):
    count = db.query(PipelineRun).filter(PipelineRun.id == run_id, PipelineRun.lease_token == token,
        PipelineRun.status == "running", PipelineRun.lease_expires_at > utcnow()).update({
            "lease_expires_at": utcnow() + timedelta(seconds=settings.DAILY_LEASE_SECONDS)}, synchronize_session=False)
    if count != 1:
        db.rollback()
        raise RuntimeError("Daily run lease was lost")


def verify_image(data):
    with Image.open(io.BytesIO(data)) as image:
        if min(image.size) < 256 or image.width * image.height > 40_000_000:
            raise ValueError("Invalid generated image dimensions")
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue(), {"width": image.width, "height": image.height}


def verify_video(data):
    import cv2
    import subprocess
    from app.core.ffmpeg_utils import get_ffmpeg_executable

    if len(data) < 1024 or len(data) > 150_000_000:
        raise ValueError("Invalid video size")
    with tempfile.TemporaryDirectory(prefix="daily-video-") as folder:
        path = Path(folder) / "check.mp4"
        path.write_bytes(data)
        capture = cv2.VideoCapture(str(path))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            decoded = 0
            while capture.isOpened():
                ok, frame = capture.read()
                if not ok:
                    break
                decoded += 1
                if decoded > 3600:
                    raise ValueError("Video exceeds the bounded daily duration")
            if fps <= 0 or min(width, height) < 256 or decoded < 2 or (expected > 0 and decoded < expected - 1):
                raise ValueError("Generated video cannot be fully decoded")
            duration_s = decoded / fps

            # Enforce 30-second target duration requirement: reject short 2-3s snippets
            if duration_s < 20.0:
                raise ValueError(
                    f"Generated video duration ({duration_s:.1f}s) is shorter than target 30s. "
                    "Refusing short snippet without full scene assembly."
                )

            # Enforce audio stream existence and audible voiceover
            ffmpeg_exe = get_ffmpeg_executable()
            proc = subprocess.run([ffmpeg_exe, "-i", str(path)], stderr=subprocess.PIPE, text=True, timeout=15)
            if "Audio:" not in proc.stderr:
                raise ValueError("Generated video is missing an audible voiceover audio stream.")

            return {
                "width": width,
                "height": height,
                "fps": fps,
                "duration_seconds": duration_s,
                "decoded_frames": decoded,
                "has_audio": True
            }
        finally:
            capture.release()


def store_asset(data, organization_id, brand_id, run_id, asset_id, kind):
    """Persist verified campaign media in the project's configured Cloudinary CDN."""
    if kind not in {"image", "video"}:
        raise ValueError("Unsupported daily asset type")
    if not CloudinaryStorageProvider.is_configured():
        raise ProviderFailure("cloudinary", "Configure Cloudinary storage before generating daily media")
    suffix = ".png" if kind == "image" else ".mp4"
    path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f"daily_{organization_id}_{brand_id}_{run_id}_{asset_id}_", suffix=suffix, delete=False) as handle:
            handle.write(data)
            path = handle.name
        result = CloudinaryStorageProvider.upload_local_file_path(
            path,
            organization_id=organization_id,
            folder_prefix=f"socialai/daily_ai/brand_{brand_id}/run_{run_id}",
            resource_type="video" if kind == "video" else "image",
        )
        url = result.get("file_url")
        if not url:
            raise ProviderFailure("cloudinary", "Cloudinary did not return a delivery URL", True)
        return url, {"storage": "cloudinary", "public_id": result.get("public_id"), "size_bytes": result.get("file_size_bytes", len(data))}
    finally:
        if path:
            Path(path).unlink(missing_ok=True)


async def execute_daily_run(run_id, session_factory=SessionLocal, provider=DailyAIProvider, storage=store_asset):
    db = session_factory()
    token = claim_run(db, run_id)
    if not token:
        db.close()
        return
    try:
        run = db.get(PipelineRun, run_id)
        stats = dict(run.stats or {})
        if stats.get("engine") != "daily_v2":
            raise ProviderFailure("configuration", "This is a legacy run; migration is required")
        brand = db.query(BrandProfile).filter(BrandProfile.id == run.brand_profile_id, BrandProfile.organization_id == run.organization_id).one()
        schedule = db.query(BusinessSchedule).filter(BusinessSchedule.brand_profile_id == brand.id).one()
        if not schedule.is_active:
            raise ProviderFailure("schedule", "Generation was disabled")
        snapshot, config = stats["brand_snapshot"], stats["config"]
        brief = db.query(ContentBrief).filter(ContentBrief.run_id == run.id).first()
        history = [{"title": b.topic, "caption": (b.raw_json or {}).get("caption", b.caption_brief)} for b in
            db.query(ContentBrief).join(PipelineRun).filter(PipelineRun.organization_id == run.organization_id,
                ContentBrief.brand_profile_id == brand.id, PipelineRun.id != run.id).order_by(ContentBrief.created_at.desc()).limit(30)]
        if not brief:
            run.stage = "Planning daily campaign"
            run.progress_percent = 10
            db.commit()
            campaign, telemetry = await provider.campaign(snapshot, history, config)
            assert_lease(db, run.id, token)
            stats["text_generation"] = telemetry
            stats["provider_calls"] += 1
            brief = ContentBrief(run_id=run.id, brand_profile_id=brand.id, topic=campaign.title, angle=campaign.theme,
                reason=campaign.objective, caption_brief=campaign.description, image_prompt="\n".join(campaign.image_prompts),
                video_prompt="\n".join(campaign.video_prompts), hashtags=campaign.hashtags, cta=campaign.call_to_action,
                tone=snapshot.get("tone_of_voice") or "As supplied", target_platforms=config["platforms"], raw_json=campaign.model_dump())
            db.add(brief)
            run.stats = stats
            db.flush()
            for kind, prompts in (("image", campaign.image_prompts), ("video", campaign.video_prompts)):
                for index, prompt in enumerate(prompts):
                    db.add(GeneratedAsset(brief_id=brief.id, type=kind, status="pending", aspect_ratio="1:1" if kind == "image" else "9:16",
                        metadata_json={"prompt": prompt, "index": index, "brand_version": stats["brand_version"]}))
            db.commit()
        campaign = DailyCampaign.model_validate(brief.raw_json)
        assets = db.query(GeneratedAsset).filter(GeneratedAsset.brief_id == brief.id).order_by(GeneratedAsset.id).all()
        for asset in assets:
            if asset.status == "done":
                continue
            metadata = dict(asset.metadata_json or {})
            prompt = metadata["prompt"]
            run.stage = "Creating " + asset.type
            run.progress_percent = 30 + int(50 * sum(a.status == "done" for a in assets) / max(1, len(assets)))
            asset.status = "processing"
            if asset.attempts >= settings.DAILY_MAX_ATTEMPTS:
                raise ProviderFailure("generation", "Asset retry limit reached")
            db.commit()
            if asset.type == "image":
                asset.attempts += 1
                db.commit()
                data, telemetry = await provider.image(prompt, snapshot, asset.aspect_ratio)
                data, technical = await asyncio.to_thread(verify_image, data)
            else:
                if not asset.operation_id:
                    asset.attempts += 1
                    db.commit()
                    def _dump_scene(sc):
                        if hasattr(sc, "model_dump"):
                            return sc.model_dump()
                        if hasattr(sc, "dict"):
                            return sc.dict()
                        return dict(sc) if isinstance(sc, dict) else str(sc)

                    script = json.dumps([_dump_scene(scene) for scene in (campaign.video_script or [])])
                    operation, telemetry = await provider.start_video(prompt + f"\nDuration: {config['video_seconds']} seconds. Scenes: {script}", snapshot)
                    assert_lease(db, run.id, token)
                    asset.operation_id = operation
                    metadata.update(telemetry)
                    asset.metadata_json = metadata
                    db.commit()
                data = await provider.poll_video(asset.operation_id)
                telemetry = {k: metadata[k] for k in ("provider", "model", "usage") if k in metadata}
                if data is None:
                    assert_lease(db, run.id, token)
                    run.status = "pending"
                    run.next_attempt_at = utcnow() + timedelta(seconds=30)
                    run.lease_token = None
                    # Stop polling permanently after a bounded provider processing window.
                    if utcnow() - asset.created_at > timedelta(hours=2):
                        raise ProviderFailure("gemini", "Video processing exceeded two hours")
                    db.commit()
                    return
                technical = await asyncio.to_thread(verify_video, data)
            assert_lease(db, run.id, token)
            reference, storage_info = await asyncio.to_thread(storage, data, run.organization_id, brand.id, run.id, asset.id, asset.type)
            assert_lease(db, run.id, token)
            asset.file_path = reference
            asset.status = "done"
            asset.error = None
            asset.metadata_json = {**metadata, **telemetry, **storage_info, "technical_validation": technical}
            stats["provider_calls"] += 1
            run.stats = dict(stats)
            db.commit()
        from app.services.quality_service import QualityService
        audit = QualityService.audit_post(brand, {"topic": campaign.title, "caption_brief": campaign.description},
            campaign.platform_captions, [{"file_path": a.file_path, "type": a.type} for a in assets], history)
        from app.services.pipeline_service import calculate_utc_scheduled_time
        scheduled_time = calculate_utc_scheduled_time(config["posting_time"], config["timezone"], run.run_date)
        approved = bool(config["auto_approve"] and audit.passed)
        post_status = "scheduled" if approved and config["auto_schedule"] else ("ready_to_publish" if approved else "pending_approval")
        assert_lease(db, run.id, token)
        post = db.query(AutonomousPost).filter(AutonomousPost.brief_id == brief.id).first()
        if not post:
            post = AutonomousPost(brief_id=brief.id, brand_profile_id=brand.id, organization_id=run.organization_id,
                status=post_status, scheduled_time=scheduled_time, auto_approved=approved, quality_report=audit.to_dict())
            db.add(post)
            db.flush()
            for platform in config["platforms"]:
                preferred_kind = "video" if platform in ("youtube", "tiktok") else "image"
                media = next((a for a in assets if a.type == preferred_kind), None)
                if not media and preferred_kind == "image":
                    media = next(iter(assets), None)
                if not media:
                    continue  # Never silently create a video to publish an image-only campaign.
                db.add(PostVariant(post_id=post.id, platform=platform, caption=campaign.platform_captions[platform],
                    title=campaign.title, hashtags=campaign.hashtags, media_path=media.file_path, media_type=media.type,
                    aspect_ratio=media.aspect_ratio, scheduled_time=scheduled_time, status=post_status))
        run.status = "completed"
        run.stage = "Completed"
        run.progress_percent = 100
        run.finished_at = utcnow()
        run.error = None
        run.lease_token = None
        stats["post_id"] = post.id
        run.stats = dict(stats)
        db.commit()
    except Exception as exc:
        logger.error(f"Daily run {run_id} failed: {exc}", exc_info=True)
        db.rollback()
        run = db.query(PipelineRun).filter(PipelineRun.id == run_id, PipelineRun.lease_token == token).first()
        if run:
            run.attempts += 1
            retryable = isinstance(exc, ProviderFailure) and exc.retryable
            run.status = "retry_pending" if retryable and run.attempts < settings.DAILY_MAX_ATTEMPTS else "failed"
            run.stage = "Retry pending" if run.status == "retry_pending" else "Failed"
            run.error = str(exc) if isinstance(exc, ProviderFailure) else f"{type(exc).__name__}: generation or storage validation failed"
            run.next_attempt_at = utcnow() + timedelta(seconds=min(900, 60 * 2 ** run.attempts))
            run.lease_token = None
            db.commit()
    finally:
        db.close()


async def daily_worker_tick():
    if not settings.DAILY_WORKER_ENABLED:
        return
    with SessionLocal() as db:
        # Scan schedules every tick: missed cron triggers and restarts do not lose a day.
        schedules = db.query(BusinessSchedule).filter(BusinessSchedule.is_active.is_(True)).all()
        for schedule in schedules:
            try:
                local_now = datetime.now(ZoneInfo(schedule.timezone))
                if local_now.strftime("%H:%M") < schedule.generation_time:
                    continue
                if schedule.generation_date and local_now.date().isoformat() < schedule.generation_date:
                    continue
                brand = db.get(BrandProfile, schedule.brand_profile_id)
                last = db.query(PipelineRun).filter(PipelineRun.brand_profile_id == brand.id).order_by(PipelineRun.run_date.desc()).first()
                if last and last.run_date < local_now.date() and (local_now.date() - last.run_date).days < (schedule.cadence_days or 1):
                    continue
                enqueue_daily(db, brand.organization_id, brand.id, local_now.date())
            except Exception:
                db.rollback()
        candidates = db.query(PipelineRun).filter(PipelineRun.status.in_(["pending", "retry_pending", "running"])).order_by(PipelineRun.created_at).limit(100).all()
        ids = [r.id for r in candidates if (r.stats or {}).get("engine") == "daily_v2" and
            (r.next_attempt_at is None or r.next_attempt_at <= utcnow()) and
            (r.status != "running" or (r.lease_expires_at and r.lease_expires_at < utcnow()))][:4]
    await asyncio.gather(*(execute_daily_run(run_id) for run_id in ids))
