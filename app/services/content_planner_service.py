# pyrefly: ignore-errors
# mypy: ignore-errors
import json
import re
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.core.logging import logger
from app.core.config import settings
from app.models.brand_profile import BrandProfile
from app.models.campaign import Campaign
from app.models.content import Content
from app.models.media_asset import MediaAsset, MediaType
from app.models.media_category import MediaCategory
from app.models.collection import Collection
from app.models.quality_report import QualityReport
from app.models.scheduled_post import ScheduledPost
from app.models.approval import Approval
from app.models.social_account import SocialAccount
from app.models.user import User
from app.models.generation_job import GenerationJob
from app.utils.enums import ContentStatus, PlatformType, ApprovalStatus, GenerationType, JobStatus
from app.services.gemini_service import GeminiService
from app.services.studio_brand_context import validate_studio_brand, studio_brand_context
from app.services.ai_provider_manager import AIProviderManager
from app.core.ai_model_config import centralized_ai_config


class ContentPlannerService:
    """Enterprise AI Content Planner service synthesizing multi-day structured campaign calendars."""

    OPTIMAL_HOURS = ["09:30", "11:00", "13:30", "17:00", "19:30"]

    # Centralized unified model hierarchy matching Gemini Creative Studio
    TEXT_PRIMARY_MODEL = centralized_ai_config.TEXT_PRIMARY_MODEL
    TEXT_SECONDARY_ROTATION = getattr(centralized_ai_config, "TEXT_SECONDARY_ROTATION", [
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-flash-latest"
    ])
    TEXT_OPENAI_MODEL = centralized_ai_config.TEXT_OPENAI_MODEL

    @classmethod
    def _generate_deterministic_fallback_plan(
        cls,
        brand_info: Dict[str, Any],
        duration_days: int,
        slot_count: int,
        target_platforms: List[str],
        campaign_goal: str,
        custom_focus: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deterministic brand-grounded fallback plan synthesizer.
        Preserves 100% of brand profile guidelines, pillars, and platform formats.
        """
        return AIProviderManager.synthesize_degraded_brand_plan(
            brand_info=brand_info,
            duration_days=duration_days,
            slot_count=slot_count,
            target_platforms=target_platforms,
            campaign_goal=campaign_goal,
            custom_focus=custom_focus
        )

    PILLARS = [
        "Product Spotlight & Craftsmanship",
        "Problem Solver & Educational Tip",
        "Customer Proof & Community Love",
        "Behind-the-Scenes & Brand Heritage",
        "Engaging Poll & Audience Discussion",
        "Limited-Time Value & Irresistible CTA",
        "Inspirational Lifestyle Upgrade"
    ]

    FORMATS = ["image", "video", "text"]

    @staticmethod
    def _attach_generated_media(db, content, asset_ids, organization_id):
        media = dict((content.meta_info or {}).get("generated_media", {}))
        campaign = db.query(Campaign).filter(Campaign.id == content.campaign_id).first() if content.campaign_id else None
        category_id = (campaign.tags or {}).get("media_category_id") if campaign and isinstance(campaign.tags, dict) else None
        collection_id = (campaign.tags or {}).get("media_collection_id") if campaign and isinstance(campaign.tags, dict) else None
        brand = db.query(BrandProfile).filter(BrandProfile.id == content.brand_profile_id).first() if content.brand_profile_id else None
        if not brand and organization_id:
            brand = db.query(BrandProfile).filter(BrandProfile.organization_id == organization_id).first()

        for asset_id in asset_ids:
            asset = db.query(MediaAsset).filter(MediaAsset.id == asset_id).first()
            if not asset:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Media asset #{asset_id} not found.")
            if asset.organization_id != organization_id:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Access denied: Media asset #{asset_id} belongs to another organization.")

            # Ensure QualityReport exists and is linked
            from app.models.quality_report import QualityReport
            report = db.query(QualityReport).filter(
                QualityReport.media_asset_id == asset_id,
                QualityReport.organization_id == organization_id
            ).first()

            if not report:
                try:
                    from app.services.quality_service import QualityService
                    kind_str = asset.media_type.value if hasattr(asset.media_type, "value") else str(asset.media_type)
                    if kind_str == "image":
                        QualityService.validate_image(
                            image_url=asset.file_url,
                            brand=brand,
                            organization_id=organization_id,
                            db=db,
                            candidate_asset_id=asset.id
                        )
                    elif kind_str == "video":
                        QualityService.validate_video(
                            video_url=asset.file_url,
                            brand=brand,
                            organization_id=organization_id,
                            db=db,
                            candidate_asset_id=asset.id
                        )
                except Exception as q_err:
                    logger.warning(f"[ContentPlanner] Notice checking QualityReport on attach: {q_err}")

            kind = asset.media_type.value if hasattr(asset.media_type, "value") else str(asset.media_type)
            if kind not in ("image", "video"):
                continue
            media[kind] = {f"{kind}_url": asset.file_url, "media_asset_id": asset.id}
            content.media_asset_id = asset.id
            if category_id and not asset.category_id:
                asset.category_id = category_id
            if collection_id and not any(existing.id == collection_id for existing in asset.collections):
                collection = db.query(Collection).filter(Collection.id == collection_id, Collection.organization_id == organization_id).first()
                if collection:
                    asset.collections.append(collection)
        content.meta_info = {**(content.meta_info or {}), "generated_media": media}

    @classmethod
    def _build_brand_relevant_prompt(
        cls,
        brand: Optional[BrandProfile],
        content: Content,
        media_type: str = "image"
    ) -> Optional[str]:
        """
        Builds a high-fidelity, brand-specific prompt incorporating:
        - Brand name, industry, and core identity
        - Verified products and services offered
        - Specific card topic, objective, and visual direction
        - Brand color palette accents
        - Realistic photography style matching the brand's industry
        Returns None if brand information is insufficient or if topic is generic without brand context.
        """
        if not brand:
            return None

        meta_b = brand.meta_info if isinstance(brand.meta_info, dict) else {}
        raw_name = (brand.name or "").strip()
        clean_b_name = re.sub(r'(?i)\s*(pvt\.?\s*ltd\.?|private\s*limited|inc\.?|llc\.?|corp\.?)$', '', raw_name).strip()

        # Insufficient brand check: avoid generating unrelated media if brand details are absent
        industry = (brand.industry or meta_b.get("category") or "").strip()
        brand_desc = meta_b.get("description") or brand.brand_guidelines or ""
        
        # Gather verified products, services, and key themes
        verified_items: List[str] = []
        for key in ("products", "services"):
            val = meta_b.get(key)
            if isinstance(val, list):
                verified_items.extend([str(item).strip() for item in val if str(item).strip()])
            elif isinstance(val, str) and val.strip():
                verified_items.append(val.strip())

        if brand.key_themes and isinstance(brand.key_themes, list):
            verified_items.extend([str(t).strip() for t in brand.key_themes if str(t).strip()])

        if (not clean_b_name or clean_b_name.lower() in ("brand", "my brand", "company")) and not brand_desc and not verified_items:
            return None

        meta_c = content.meta_info if isinstance(content.meta_info, dict) else {}
        topic = (content.title or "").strip()
        v_direction = (
            meta_c.get("visual_prompt_direction") or
            meta_c.get("scene_blueprint") or
            meta_c.get("caption_brief") or
            meta_c.get("custom_focus") or
            ""
        ).strip()
        caption = (content.caption or "").strip()

        # Strip prefixes from topic
        clean_topic = re.sub(r'(?i)\b(hook|details|focus|caption|call to action|cta|day\s*\d+)\s*:.*$', '', topic).strip(' :.-,;')
        if not clean_topic or clean_topic.lower() in ("modern business innovation", "post idea", "daily post", "day 1", "day 2", "day 3", "day 4", "day 5", "day 6", "day 7"):
            if v_direction:
                clean_topic = v_direction[:120]
            elif verified_items:
                clean_topic = f"{verified_items[0]} showcase"
            elif caption:
                clean_topic = caption[:100]
            else:
                # No specific topic and no product details available: do not generate unrelated generic image!
                return None

        # Determine matched product/service context for this post
        matched_subject = ""
        topic_lower = f"{clean_topic} {caption} {v_direction}".lower()
        matched = [item for item in verified_items if item.lower() in topic_lower]
        if matched:
            matched_subject = f", highlighting authentic {', '.join(matched[:3])}"
        elif verified_items:
            matched_subject = f", featuring {', '.join(verified_items[:2])}"

        # Color accents
        color_accents = ""
        if brand.color_palette and isinstance(brand.color_palette, list):
            valid_colors = [c for c in brand.color_palette if isinstance(c, str) and c.startswith("#")]
            if valid_colors:
                color_accents = f", color harmony with {', '.join(valid_colors[:2])}"

        is_video = "video" in str(media_type).lower() or "reel" in str(media_type).lower()
        ind_lower = industry.lower()

        # Build industry-specific composition & photography style
        is_ceramics_or_craft = any(k in f"{ind_lower} {brand_desc.lower()} {clean_b_name.lower()}".lower() for k in ["ceramic", "pottery", "clay", "artisan", "handcrafted", "lifestyle", "tableware", "planter", "mug", "bowl"])
        is_kids_or_toys = any(k in f"{ind_lower} {brand_desc.lower()}".lower() for k in ["toy", "kids", "children", "baby", "game", "doll", "plush"])
        is_fashion = any(k in f"{ind_lower} {brand_desc.lower()}".lower() for k in ["fashion", "apparel", "clothing", "wear", "couture", "luxury", "boutique"])
        is_food_beverage = any(k in f"{ind_lower} {brand_desc.lower()}".lower() for k in ["food", "beverage", "restaurant", "cafe", "coffee", "bakery", "dining", "cuisine"])
        is_tech = any(k in f"{clean_b_name} {ind_lower} {clean_topic}".lower() for k in ["tech", "software", "saas", "cloud", "automation", "crm", "app", "code", "ai", "platform", "bpo", "outsourcing"])

        scene_subject = v_direction if (v_direction and len(v_direction) > 20 and not v_direction.startswith("Highlight key")) else f"{clean_topic}{matched_subject}"

        if is_video:
            prompt = (
                f"Cinematic 30s commercial reel for {clean_b_name}: {scene_subject}{color_accents}. "
                f"Dynamic gliding camera motion, smooth transitions, ambient natural lighting, 4k ultra-high definition, professional color grading, sharp focus"
            )
        elif is_ceramics_or_craft:
            prompt = (
                f"Artisanal editorial product photography of {scene_subject} for {clean_b_name}{color_accents}. "
                f"Handcrafted stoneware pottery details, natural rustic studio workshop background, soft morning sunlight, tactile ceramic glaze texture, photorealistic 8k uhd, sharp focus, no text overlays, no watermark"
            )
        elif is_kids_or_toys:
            prompt = (
                f"High-end editorial commercial photography of {scene_subject} for {clean_b_name}{color_accents}. "
                f"Warm studio lighting, elegant clean display, soft pastel aesthetic, high quality 8k resolution, sharp focus, no text overlays, no watermark"
            )
        elif is_fashion:
            prompt = (
                f"High-end fashion editorial photography of {scene_subject} for {clean_b_name}{color_accents}. "
                f"Elegant atelier lighting, luxurious fabric textures, sophisticated aesthetic, 8k uhd, sharp focus, no text overlays, no watermark"
            )
        elif is_food_beverage:
            prompt = (
                f"Artisanal culinary product photography of {scene_subject} for {clean_b_name}{color_accents}. "
                f"Mouth-watering presentation, fresh ingredients, warm ambient restaurant lighting, shallow depth of field, 8k uhd, sharp focus, no text overlays, no watermark"
            )
        elif is_tech:
            prompt = (
                f"Modern commercial photography of {scene_subject} for {clean_b_name}{color_accents}. "
                f"Shot on 85mm f/1.4 lens, natural daylight studio lighting, clean elegant composition, sharp focus, 8k uhd, no text overlays, no watermark"
            )
        else:
            prompt = (
                f"Photorealistic commercial photography of {scene_subject} for {clean_b_name}{color_accents}. "
                f"Shot on 85mm f/1.4 lens, shallow depth of field, professional studio lighting, 8k resolution, elegant aesthetic, sharp focus, high quality, no watermark, no text overlays"
            )

        return prompt.strip()

    @classmethod
    def _sync_scheduled_generation_job(
        cls,
        db: Session,
        content: Content,
        organization_id: int,
        user_id: Optional[int] = None,
        suggested_date: Optional[str] = None,
        suggested_time: Optional[str] = None,
        timezone_str: str = "Asia/Kolkata",
        content_format: Optional[str] = None,
        force_sync: bool = False,
    ) -> Optional[GenerationJob]:
        """
        Creates or updates a persistent GenerationJob in MySQL for a planned campaign idea.
        Enforces user opt-in for automatic media generation and brand relevance.
        """
        meta = dict(content.meta_info or {})
        s_date = suggested_date or meta.get("suggested_date")
        s_time = suggested_time or meta.get("suggested_time", "10:00")
        if not s_date:
            return None

        # 1. Enforce content format: text posts never generate media
        fmt = (content_format or meta.get("content_format", "image")).lower()
        if "text" in fmt and "image" not in fmt and "video" not in fmt:
            return None

        # 2. Enforce explicit user opt-in: automatic media generation must be explicitly enabled
        sp = db.query(ScheduledPost).filter(ScheduledPost.content_id == content.id).first()
        camp = db.query(Campaign).filter(Campaign.id == content.campaign_id).first() if content.campaign_id else None
        is_auto_gen = bool(
            force_sync or
            meta.get("auto_generate_media") is not False or
            (sp and (sp.publish_metadata or {}).get("auto_generate_media") is not False) or
            (camp and isinstance(camp.tags, dict) and camp.tags.get("auto_generate_media") is not False) or
            (meta.get("is_planned_idea") and s_date)
        )
        if not is_auto_gen:
            # If a pending job was scheduled before, cancel it now
            job_id = meta.get("scheduled_job_id")
            if job_id:
                prev_job = db.query(GenerationJob).filter(
                    GenerationJob.id == job_id,
                    GenerationJob.status == JobStatus.PENDING
                ).first()
                if prev_job:
                    prev_job.status = JobStatus.FAILED
                    prev_job.error_message = "Cancelled: Automatic media generation is not enabled."
                    if isinstance(prev_job.parameters, dict):
                        prev_job.parameters["cancelled"] = True
                    db.flush()
            return None

        import pytz
        tz_name = timezone_str or meta.get("timezone", "Asia/Kolkata")
        try:
            tz = pytz.timezone(tz_name or "Asia/Kolkata")
            naive_dt = datetime.strptime(f"{s_date} {s_time}", "%Y-%m-%d %H:%M")
            local_dt = tz.localize(naive_dt)
            target_utc = local_dt.astimezone(pytz.utc).replace(tzinfo=None)
        except Exception as dt_err:
            logger.warning(f"[AutoMediaScheduler] Failed to parse schedule timing {s_date} {s_time}: {dt_err}")
            return None

        local_iso = f"{s_date}T{s_time}:00"
        target_utc_iso = target_utc.isoformat()

        is_video = "video" in fmt or "reel" in fmt
        job_type = GenerationType.VIDEO_GENERATION if is_video else GenerationType.IMAGE_GENERATION
        provider = "multi-scene-cinema-director" if is_video else "cloudflare-workers-ai"
        model_name = "google-veo-cinema-director" if is_video else "@cf/black-forest-labs/flux-1-schnell"

        # 3. Resolve Brand Profile and build brand-relevant prompt
        brand = db.query(BrandProfile).filter(BrandProfile.id == content.brand_profile_id).first() if content.brand_profile_id else None
        if not brand and organization_id:
            brand = db.query(BrandProfile).filter(BrandProfile.organization_id == organization_id).first()

        prompt = cls._build_brand_relevant_prompt(brand, content, media_type="video" if is_video else "image")
        if not prompt:
            logger.info(f"[AutoMediaScheduler] Skipping scheduled generation job for content #{content.id}: Insufficient brand context or non-specific topic.")
            return None

        # Look up existing job
        job = None
        job_id = meta.get("scheduled_job_id")
        if job_id:
            job = db.query(GenerationJob).filter(GenerationJob.id == job_id).first()

        if not job:
            active_jobs = db.query(GenerationJob).filter(
                GenerationJob.organization_id == organization_id,
                GenerationJob.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
            ).all()
            for candidate in active_jobs:
                if isinstance(candidate.parameters, dict) and candidate.parameters.get("content_id") == content.id:
                    job = candidate
                    break

        job_params = {
            "content_id": content.id,
            "tenant_id": organization_id,
            "campaign_id": content.campaign_id,
            "media_type": "video" if is_video else "image",
            "scheduled_generation_time": local_iso,
            "scheduled_time_utc": target_utc_iso,
            "timezone": tz_name,
            "provider": provider,
            "is_scheduled_media_generation": True,
            "retries": 0,
            "max_retries": 3,
        }

        user_assigned = user_id or content.creator_id
        if not user_assigned:
            admin_u = db.query(User).filter(User.organization_id == organization_id).first()
            user_assigned = admin_u.id if admin_u else 1

        if job:
            # If job is already COMPLETED and media exists, do nothing
            if job.status == JobStatus.COMPLETED and (meta.get("generated_media") or content.media_asset_id):
                return job

            setattr(job, "job_type", job_type)
            setattr(job, "model_name", model_name)
            setattr(job, "prompt", prompt)
            setattr(job, "parameters", job_params)
            setattr(job, "status", JobStatus.PENDING)
            setattr(job, "error_message", None)
            db.flush()
            logger.info(f"[AutoMediaScheduler] Scheduled job #{job.id} updated for content #{content.id} at {local_iso} ({tz_name}).")
        else:
            job = GenerationJob(
                job_type=job_type,
                status=JobStatus.PENDING,
                prompt=prompt,
                model_name=model_name,
                parameters=job_params,
                result_data={},
                user_id=user_assigned,
                organization_id=organization_id
            )
            db.add(job)
            db.flush()
            db.refresh(job)
            logger.info(f"[AutoMediaScheduler] Scheduled job #{job.id} created for content #{content.id} at {local_iso} ({tz_name}).")

        meta["scheduled_job_id"] = job.id
        meta["scheduled_time_utc"] = target_utc_iso
        meta["suggested_datetime"] = local_iso
        meta["generation_status"] = "scheduled"
        meta["generation_error"] = None
        content.meta_info = meta
        return job

    @classmethod
    def _calculate_schedule_slots(
        cls,
        start_date: datetime,
        duration_days: int,
        frequency: str,
        preferred_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Compute chronological publishing dates and optimal time slots."""
        slots = []
        base_date = start_date or (datetime.utcnow() + timedelta(days=1))
        # Zero out hours/minutes for clean day calculations
        current_day = datetime(base_date.year, base_date.month, base_date.day)
        fixed_time = preferred_time if preferred_time and preferred_time.strip() else "10:00"

        if frequency == "2_per_day":
            for d in range(duration_days):
                day_date = current_day + timedelta(days=d)
                slots.append({"day": d + 1, "date": day_date.strftime("%Y-%m-%d"), "time": fixed_time})
                slots.append({"day": d + 1, "date": day_date.strftime("%Y-%m-%d"), "time": "18:30"})
        elif frequency == "3_per_week":
            # Monday, Wednesday, Friday pattern
            count = 3 if duration_days <= 7 else int((duration_days / 7) * 3)
            step_days = max(1, duration_days // max(1, count))
            for i in range(count):
                day_offset = min(duration_days - 1, i * step_days)
                day_date = current_day + timedelta(days=day_offset)
                slots.append({
                    "day": day_offset + 1,
                    "date": day_date.strftime("%Y-%m-%d"),
                    "time": fixed_time if preferred_time else cls.OPTIMAL_HOURS[i % len(cls.OPTIMAL_HOURS)]
                })
        elif frequency == "5_per_week":
            # Weekdays pattern
            count = 5 if duration_days <= 7 else int((duration_days / 7) * 5)
            step_days = max(1, duration_days // max(1, count))
            for i in range(count):
                day_offset = min(duration_days - 1, i * step_days)
                day_date = current_day + timedelta(days=day_offset)
                slots.append({
                    "day": day_offset + 1,
                    "date": day_date.strftime("%Y-%m-%d"),
                    "time": fixed_time if preferred_time else cls.OPTIMAL_HOURS[i % len(cls.OPTIMAL_HOURS)]
                })
        elif frequency == "alternate_days":
            count = (duration_days + 1) // 2
            for i in range(count):
                day_offset = i * 2
                day_date = current_day + timedelta(days=day_offset)
                slots.append({
                    "day": day_offset + 1,
                    "date": day_date.strftime("%Y-%m-%d"),
                    "time": fixed_time if preferred_time else cls.OPTIMAL_HOURS[i % len(cls.OPTIMAL_HOURS)]
                })
        else:
            # Default: 1 post per day using user's fixed daily time
            for d in range(duration_days):
                day_date = current_day + timedelta(days=d)
                slots.append({
                    "day": d + 1,
                    "date": day_date.strftime("%Y-%m-%d"),
                    "time": fixed_time
                })

        return slots

    @classmethod
    async def generate_plan(
        cls,
        db: Session,
        organization_id: int,
        brand_profile_id: Optional[int] = None,
        duration_days: int = 7,
        platforms: Optional[List[str]] = None,
        posting_frequency: str = "1_per_day",
        campaign_goal: str = "product_launch",
        custom_focus: Optional[str] = None,
        start_date: Optional[datetime] = None,
        media_format_preference: str = "both",
        daily_posting_time: Optional[str] = None,
        timezone: Optional[str] = "Asia/Kolkata",
        media_category_id: Optional[int] = None,
        media_collection_id: Optional[int] = None,
        allow_degraded_fallback: bool = False,
    ) -> Dict[str, Any]:
        """Generate a complete structured content plan using Brand Profile and AI."""
        # 1. Strict Multi-Tenant Brand Resolution & Validation
        brand = None
        if brand_profile_id:
            brand = db.query(BrandProfile).filter(
                BrandProfile.id == brand_profile_id,
                BrandProfile.organization_id == organization_id
            ).first()
            if not brand:
                foreign = db.query(BrandProfile).filter(BrandProfile.id == brand_profile_id).first()
                if foreign:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied: Cross-tenant brand profile access is strictly forbidden."
                    )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Brand profile #{brand_profile_id} was not found in your organization."
                )
        else:
            brand = db.query(BrandProfile).filter(
                BrandProfile.organization_id == organization_id
            ).order_by(BrandProfile.updated_at.desc(), BrandProfile.id.desc()).first()

        if not brand:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A valid Brand Profile is required before generating a campaign plan. Please configure your Brand Profile first."
            )

        if media_category_id:
            category_exists = db.query(MediaCategory.id).filter(
                MediaCategory.id == media_category_id,
                MediaCategory.organization_id == organization_id,
            ).first()
            if not category_exists:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Selected media category was not found in your organization.")

        selected_collection = None
        if media_collection_id:
            selected_collection = db.query(Collection).filter(
                Collection.id == media_collection_id,
                Collection.organization_id == organization_id,
            ).first()
            if not selected_collection:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Selected Media Library collection was not found in your organization.")

        validate_studio_brand(brand)
        brand_name = brand.name
        industry = brand.industry if brand.industry else "Commercial"
        tone = brand.tone_of_voice if brand and brand.tone_of_voice else "Engaging, Authentic, Inspiring"
        target_aud = brand.target_audience if brand and brand.target_audience else "Not specified"
        tagline = brand.tagline if brand and brand.tagline else ""
        themes = ", ".join(brand.key_themes) if brand and brand.key_themes else "Not specified"
        brand_cta = brand.cta if brand and brand.cta else "Tap the link in our bio to explore more"
        brand_avoid = brand.avoid if brand and brand.avoid else "clichéd claims, off-brand slang, false promises"

        meta = brand.meta_info if (brand and isinstance(brand.meta_info, dict)) else {}
        brand_desc = (brand.brand_guidelines or meta.get("description") or meta.get("desc") or meta.get("business_description") or "").strip()
        location = (brand.location or "").strip()
        timings = (brand.timings or "").strip()
        website = (meta.get("website") or meta.get("website_url") or "").strip()
        phone = (meta.get("phone") or meta.get("contact_phone") or "").strip()
        email = (meta.get("email") or meta.get("contact_email") or "").strip()
        category = (meta.get("category") or "").strip()
        visual_identity = (meta.get("visual_identity") or "").strip()
        typography = (meta.get("typography") or "").strip() or "Clean modern typography"
        palette = brand.color_palette if isinstance(brand.color_palette, list) and brand.color_palette else ["#6366f1", "#ec4899"]
        palette_hex = ", ".join(palette)
        messaging_style = (meta.get("messaging_style") or "").strip()
        usp = meta.get("usp", "Not specified")
        positioning = meta.get("brand_positioning", "Not specified")
        marketing_objectives = meta.get("marketing_objectives", "")
        buyer_personas = meta.get("buyer_personas", [])
        content_format_ratios = meta.get("content_format_ratios", {})
        products = meta.get("products", [])

        products_lines = []
        if products and isinstance(products, list):
            for p in products[:8]:
                if isinstance(p, dict) and p.get("name"):
                    price_info = f" (${p.get('price')})" if p.get("price") else ""
                    products_lines.append(f"  • {p['name']}{price_info}: {p.get('description', '')}")
        products_str = ("\nVerified Brand Products:\n" + "\n".join(products_lines)) if products_lines else ""

        # Format active buyer personas
        active_personas = [p for p in buyer_personas if isinstance(p, dict) and p.get("active", True)] if isinstance(buyer_personas, list) else []
        personas_str = ""
        if active_personas:
            p_lines = [f"  • {p.get('title', 'Target Persona')}: {p.get('desc', '')}" for p in active_personas[:4]]
            personas_str = "\nTarget Buyer Personas (Tailor hook angles, tone, and audience resonance to these personas):\n" + "\n".join(p_lines) + "\n"

        # 2. Determine target platforms
        valid_platforms = [p.lower() for p in (platforms or ["instagram", "facebook", "youtube"]) if p.lower() in ("instagram", "facebook", "youtube")]
        if not valid_platforms:
            valid_platforms = ["instagram", "facebook", "youtube"]

        duration = 30 if duration_days == 30 else 7
        slots = cls._calculate_schedule_slots(start_date, duration, posting_frequency, daily_posting_time)
        slot_count = len(slots)

        # 3. Campaign Goal descriptor - supports user's manual custom entry or presets
        goal_labels = {
            "product_launch": "Launch flagship product offerings with desire, curiosity and social excitement",
            "brand_awareness": "Build brand recognition, distinct visual presence and memorable storytelling",
            "community_engagement": "Drive comments, conversations, shares, and authentic audience interaction",
            "promotional_deals": "Convert warm followers with compelling offers, bundles, and value propositions",
            "educational_tips": "Position the brand as a helpful expert with tactical styling, how-tos and advice",
            "behind_the_scenes": "Humanize the company with atelier craft, founder notes, and authentic culture",
            "custom": custom_focus or "High-engagement social campaign"
        }
        clean_goal = str(campaign_goal or "").strip()
        if clean_goal in goal_labels:
            goal_desc = goal_labels[clean_goal]
        elif clean_goal.lower() in goal_labels:
            goal_desc = goal_labels[clean_goal.lower()]
        elif clean_goal and clean_goal not in ["product_launch", "default"]:
            # User manually entered the primary campaign goal: use it directly
            goal_desc = clean_goal
        elif marketing_objectives:
            # Fallback to the real-time configured marketing objective from Content Goals & Audience Targeting
            goal_desc = marketing_objectives
        else:
            goal_desc = "High-engagement brand and social growth campaign"

        if custom_focus:
            goal_desc += f" (Specific Focus: {custom_focus})"

        # Media format preference directive and format mix ratio
        media_pref = (media_format_preference or "both").lower()
        if media_pref in ["images", "image"]:
            format_requirement = "Every single idea MUST be an IMAGE-based format ('image'). Do NOT output video formats. Tailor visuals, hooks and concepts for still photography, graphics and high-res visual product highlights."
            format_schema = "'image'"
        elif media_pref in ["videos", "video"]:
            format_requirement = "Every single idea MUST be a VIDEO-based format ('video'). Tailor hooks, pacing, audio concepts and briefs strictly for short-form video reels, shorts and clips. Do NOT output image formats."
            format_schema = "'video'"
        else:
            if isinstance(content_format_ratios, dict) and content_format_ratios:
                r_reels = content_format_ratios.get('reels', 50)
                r_imgs = content_format_ratios.get('images', 40)
                r_txt = content_format_ratios.get('text', 10)
                format_requirement = f"Distribute content formats according to the user's strategic distribution mix: ~{r_reels}% short-form video reels ('video'), ~{r_imgs}% high-res photography ('image'), ~{r_txt}% community updates."
            else:
                format_requirement = "Balance content formats evenly across static images ('image') and short-form video reels ('video') for a dynamic multi-channel mix."
            format_schema = "one of ['image', 'video']"

        # 4. Construct AI Prompt with all 5 tabs and full instant caption/description requirement
        system_instruction = (
            f"You are the Chief Social Media Strategist and Lead Creative Director for '{brand_name}'.\n"
            f"Brand Tone: {tone}. Industry: {industry} ({category}). Target Audience: {target_aud}.\n"
            f"Core Mission & Ethos: {brand_desc or 'Delivering high-value, streamlined solutions for modern businesses'}.\n"
            f"Location & Origins: {location or 'Global'}. Tagline: {tagline}.\n"
            f"Your job is to generate a comprehensive, cohesive {duration}-day Content Plan with exactly {slot_count} distinct post ideas.\n"
            f"CRITICAL REQUIREMENTS:\n"
            f"1. INSTANT FULL DESCRIPTION: For EVERY SINGLE day, you MUST write a complete, rich, publication-ready post caption ('description') of 120-220 words with tasteful emojis, paragraph breaks, real storytelling connecting the brand mission, and a strong CTA. Do NOT provide a short summary.\n"
            f"2. PROGRESSIVE 7-DAY PILLARS: Day 1: Brand Origin & Mission; Day 2: Core Product Feature Spotlight; Day 3: Tactical Tip & How-To; Day 4: Customer Transformation & Social Proof; Day 5: Behind-the-Scenes Craft & Culture; Day 6: Industry Perspective & Community Poll; Day 7: Weekend VIP Access & Direct CTA.\n"
            f"3. DIVERSITY: Every day must feature completely distinct topics, angles, hooks, and descriptions. Do NOT repeat topics.\n"
            f"4. Media Format Constraint: {format_requirement}\n"
            f"5. Strictly respect brand restrictions: Never mention '{brand_avoid}'.\n"
            f"6. Return valid JSON strictly matching the requested array schema."
        )

        user_prompt = (
            f"Brand Knowledge Profile:\n"
            f"- Brand Name: {brand_name}\n"
            f"- Industry & Category: {industry} / {category}\n"
            f"- Business Description & Mission: {brand_desc}\n"
            f"- Location / Roots: {location}\n"
            f"- Operating Hours / Timings: {timings}\n"
            f"- Website: {website}\n"
            f"- Contact Email: {email}\n"
            f"- Contact Phone: {phone}\n"
            f"- Tagline: {tagline}\n"
            f"- Tone of Voice: {tone} | Messaging Style: {messaging_style}\n"
            f"- Target Audience: {target_aud} (Positioning: {positioning})\n"
            f"- USP / Value Proposition: {usp}\n"
            f"- Core Themes: {themes}\n"
            f"- Standard CTA: {brand_cta}\n"
            f"- Aesthetics / Restrictions to Strictly Avoid: {brand_avoid}\n"
            f"- Visual Identity Aesthetic: {visual_identity} (Color Palette: {palette_hex})\n"
            f"- Campaign Objective: {goal_desc}\n"
            f"- Media Format Requirement: {media_pref.upper()} - {format_requirement}\n"
            f"- Plan Duration: {duration} days ({slot_count} total post slots)\n"
            f"- Target Platforms: {', '.join(valid_platforms)}\n"
            f"{products_str}"
            f"{personas_str}\n\n"
            f"Generate a strategic social media content calendar with exactly {slot_count} post ideas.\n"
            f"Return a JSON object with:\n"
            f"- campaign_title: A high-impact title for this {duration}-day plan (string)\n"
            f"- campaign_summary: 2-sentence executive summary of the strategy (string)\n"
            f"- ideas: An array of exactly {slot_count} objects, each containing:\n"
            f"    - title: Punchy topic name (string, 4-8 words)\n"
            f"    - hook: Scroll-stopping opening line (string)\n"
            f"    - description: Complete, ready-to-post, full-length caption and description (120-220 words) with emojis, narrative storytelling, tactical value, and CTA (string)\n"
            f"    - caption_brief: Concise 1-2 sentence core message summary (string)\n"
            f"    - content_format: {format_schema}\n"
            f"    - pillar: strategic pillar name (string)\n"
            f"    - visual_prompt_direction: Detailed, cinematic photographic description for the visual artwork (composition, hero subject, setting, natural lighting, brand color harmonies, camera perspective) (string, 25-45 words)\n"
            f"    - suggested_hashtags: array of 4-6 curated relevant hashtags (list of strings)\n"
            f"    - cta: Contextual call to action for this specific post (string)\n"
        )

        # 5. Execute AI Generation with Centralized Multi-Provider Resilience Cascade
        brand_info_payload = {
            "name": brand_name,
            "industry": industry,
            "tone_of_voice": tone,
            "target_audience": target_aud,
            "usp": usp,
            "cta": brand_cta,
            "products": products,
            "key_themes": brand.key_themes if (brand and brand.key_themes) else [themes],
            "guidelines": brand_desc,
            "avoid": brand_avoid
        }

        def _validate_plan_response(data: Dict[str, Any]) -> bool:
            if not isinstance(data, dict):
                return False
            ideas_list = data.get("ideas")
            if not isinstance(ideas_list, list) or len(ideas_list) != slot_count:
                return False
            return all(
                isinstance(idea, dict) and all(
                    isinstance(idea.get(field), str) and idea[field].strip()
                    for field in ("title", "hook", "caption_brief", "description")
                )
                for idea in ideas_list
            )

        generated_data = await AIProviderManager.generate_json_with_resilience(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            brand_info=brand_info_payload,
            duration_days=duration,
            slot_count=slot_count,
            target_platforms=valid_platforms,
            campaign_goal=goal_desc,
            custom_focus=custom_focus,
            validation_fn=_validate_plan_response,
            allow_degraded_fallback=allow_degraded_fallback
        )

        if not isinstance(generated_data, dict) or not isinstance(generated_data.get("ideas"), list):
            raise HTTPException(status_code=503, detail="AI planning is unavailable. Check provider configuration and retry.")
        if len(generated_data["ideas"]) != slot_count or any(
            not isinstance(idea, dict) or not all(
                isinstance(idea.get(field), str) and idea[field].strip()
                for field in ("title", "hook", "caption_brief", "description")
            ) for idea in generated_data["ideas"]
        ):
            raise HTTPException(status_code=502, detail="AI returned an incomplete campaign plan. Please regenerate.")

        # 7. Merge generated ideas with schedule slots and platform assignments
        raw_ideas = generated_data.get("ideas", [])
        structured_ideas = []
        seen_titles = set()

        for idx, slot in enumerate(slots):
            raw = raw_ideas[idx] if idx < len(raw_ideas) else {}
            title = raw.get("title") or f"{brand_name} Spotlight #{idx + 1}"
            
            # Deduplication
            if title.lower() in seen_titles:
                title = f"{title} (Part {idx + 1})"
            seen_titles.add(title.lower())

            raw_fmt = str(raw.get("content_format", "")).lower()
            if media_pref in ["images", "image"]:
                fmt = "image"
            elif media_pref in ["videos", "video"]:
                fmt = "video"
            else:
                if "video" in raw_fmt or "reel" in raw_fmt:
                    fmt = "video"
                elif "text" in raw_fmt or "status" in raw_fmt:
                    fmt = "text"
                else:
                    fmt = "image" if idx % 3 != 1 else "video"

            pillar = raw.get("pillar") or cls.PILLARS[idx % len(cls.PILLARS)]
            hook = raw.get("hook") or f"Here is how {brand_name} is redefining {industry}."
            brief = raw.get("caption_brief") or f"Highlight key customer benefits and invite followers to take action."
            cta = raw.get("cta") or brand_cta
            hashtags = raw.get("suggested_hashtags") or [f"#{brand_name.replace(' ', '')}", f"#{industry.replace(' ', '')}", "#Growth", "#Community"]

            full_desc = raw.get("description") or raw.get("caption")
            if not full_desc or len(full_desc.strip()) < 30:
                full_desc = f"{hook}\n\n{brief}\n\nAt {brand_name}, we believe in seamless excellence and tangible results.\n\n{cta}"

            suggested_datetime = f"{slot['date']}T{slot['time']}:00"

            idea_id = f"idea-d{slot['day']}-s{idx + 1}"
            structured_ideas.append({
                "id": idea_id,
                "day_number": slot["day"],
                "slot_index": idx,
                "title": title,
                "topic": title,
                "hook": hook,
                "description": full_desc,
                "caption": full_desc,
                "caption_brief": brief,
                "content_format": fmt,
                "platform": valid_platforms[idx % len(valid_platforms)],
                "platforms": valid_platforms,
                "suggested_date": slot["date"],
                "suggested_time": slot["time"],
                "suggested_datetime": suggested_datetime,
                "pillar": pillar,
                "content_pillar": pillar,
                "hashtags": hashtags,
                "cta": cta,
                "call_to_action": cta,
                "custom_focus": custom_focus or "",
                "visual_prompt_direction": (raw.get("visual_prompt_direction") or raw.get("image_prompt") or "").strip(),
                "image_prompt": (raw.get("visual_prompt_direction") or raw.get("image_prompt") or "").strip(),
                "plan_status": "planned"
            })

        plan_title = generated_data.get("campaign_title") or f"{brand_name} {duration}-Day {campaign_goal.replace('_', ' ').title()} Plan"
        plan_summary = generated_data.get("campaign_summary") or f"A structured {duration}-day editorial roadmap optimized for {', '.join(valid_platforms)}."

        return {
            "title": plan_title,
            "campaign_name": plan_title,
            "brand_profile_id": brand.id,
            "summary": plan_summary,
            "campaign_goal": campaign_goal,
            "custom_focus": custom_focus or "",
            "duration_days": duration,
            "posting_frequency": posting_frequency,
            "media_format_preference": media_pref,
            "media_category_id": media_category_id,
            "media_collection_id": selected_collection.id if selected_collection else None,
            "media_collection_name": selected_collection.name if selected_collection else None,
            "media_collection_description": selected_collection.description if selected_collection else None,
            "daily_posting_time": daily_posting_time or "10:00",
            "timezone": timezone or "Asia/Kolkata",
            "platforms": valid_platforms,
            "start_date": slots[0]["date"] if slots else datetime.utcnow().strftime("%Y-%m-%d"),
            "end_date": slots[-1]["date"] if slots else datetime.utcnow().strftime("%Y-%m-%d"),
            "total_ideas": len(structured_ideas),
            "ideas_count": len(structured_ideas),
            "ideas": structured_ideas
        }

    @classmethod
    def save_plan_to_database(
        cls,
        db: Session,
        organization_id: int,
        current_user: User,
        plan_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Save a generated plan as a Campaign and its ideas as Content items in the database."""
        # Bind persisted content to the explicitly selected, tenant-owned brand.
        brand_id = plan_data.get("brand_profile_id")
        brand_query = db.query(BrandProfile).filter(BrandProfile.organization_id == organization_id)
        if brand_id:
            brand_query = brand_query.filter(BrandProfile.id == brand_id)
        brand = brand_query.order_by(BrandProfile.updated_at.desc(), BrandProfile.id.desc()).first()
        if not brand:
            raise HTTPException(status_code=400, detail="Select a valid Brand Profile in your organization.")
        brand_id = brand.id

        # 2. Check for existing campaign or create new
        campaign_id = plan_data.get("campaign_id")
        campaign = None
        if campaign_id:
            campaign = db.query(Campaign).filter(
                Campaign.id == campaign_id,
                Campaign.organization_id == organization_id
            ).first()

        if campaign_id and not campaign:
            raise HTTPException(status_code=404, detail="Campaign plan not found.")
        if campaign and campaign.brand_profile_id != brand_id:
            raise HTTPException(status_code=400, detail="A campaign cannot be moved to another brand.")

        start_dt = None
        end_dt = None
        if plan_data.get("start_date"):
            try:
                start_dt = datetime.strptime(plan_data["start_date"], "%Y-%m-%d")
            except Exception:
                start_dt = datetime.utcnow()
        if plan_data.get("end_date"):
            try:
                end_dt = datetime.strptime(plan_data["end_date"], "%Y-%m-%d")
            except Exception:
                end_dt = (start_dt or datetime.utcnow()) + timedelta(days=plan_data.get("duration_days", 7))

        campaign_tags = {
            "duration_days": plan_data.get("duration_days", 7),
            "posting_frequency": plan_data.get("posting_frequency", "1_per_day"),
            "platforms": plan_data.get("platforms", ["instagram", "facebook", "youtube"]),
            "campaign_goal": plan_data.get("campaign_goal", "product_launch"),
            "custom_focus": plan_data.get("custom_focus", ""),
            "media_format_preference": plan_data.get("media_format_preference", "both"),
            "summary": plan_data.get("summary", ""),
            "daily_posting_time": plan_data.get("daily_posting_time", "10:00"),
            "timezone": plan_data.get("timezone", "Asia/Kolkata"),
            "media_category_id": plan_data.get("media_category_id"),
            "media_collection_id": plan_data.get("media_collection_id"),
            "auto_generate_media": True,
            "total_ideas": len(plan_data.get("ideas", []))
        }

        media_category_id = plan_data.get("media_category_id")
        if media_category_id:
            category_exists = db.query(MediaCategory.id).filter(
                MediaCategory.id == media_category_id,
                MediaCategory.organization_id == organization_id,
            ).first()
            if not category_exists:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Selected media category was not found in your organization.")

        media_collection_id = plan_data.get("media_collection_id")
        if media_collection_id:
            selected_collection = db.query(Collection).filter(
                Collection.id == media_collection_id,
                Collection.organization_id == organization_id,
            ).first()
            if not selected_collection:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Selected Media Library collection was not found in your organization.")
            campaign_tags["media_collection_name"] = selected_collection.name
            campaign_tags["media_collection_description"] = selected_collection.description

        if not campaign:
            campaign = Campaign(
                title=plan_data.get("title") or f"Campaign Plan ({datetime.utcnow().strftime('%b %d, %Y')})",
                objective=plan_data.get("summary") or plan_data.get("campaign_goal", "Social content strategy"),
                start_date=start_dt,
                end_date=end_dt,
                tags=campaign_tags,
                status="active",
                organization_id=organization_id,
                brand_profile_id=brand_id
            )
            db.add(campaign)
            db.flush()
            db.refresh(campaign)
        else:
            campaign.title = plan_data.get("title", campaign.title)
            campaign.objective = plan_data.get("summary", campaign.objective)
            campaign.start_date = start_dt
            campaign.end_date = end_dt
            campaign.tags = campaign_tags
            db.flush()
            db.refresh(campaign)

        # 3. Create or update Content ideas
        ideas = plan_data.get("ideas", [])
        saved_ideas = []

        for idea in ideas:
            idea_id = idea.get("id")
            content_record = None

            if isinstance(idea_id, int):
                content_record = db.query(Content).filter(
                    Content.id == idea_id,
                    Content.campaign_id == campaign.id,
                    Content.organization_id == organization_id
                ).first()
                if not content_record:
                    raise HTTPException(status_code=404, detail="Planned idea does not belong to this campaign.")

            platforms_list = idea.get("platforms") or plan_data.get("platforms", ["instagram"])
            # Format normalized platforms for PlatformType enum
            normalized_platforms = []
            for p in platforms_list:
                try:
                    normalized_platforms.append(PlatformType(p.lower()))
                except Exception:
                    pass
            if not normalized_platforms:
                normalized_platforms = [PlatformType.INSTAGRAM]

            full_desc = idea.get("description") or idea.get("caption") or (
                f"{idea.get('hook', '')}\n\n"
                f"{idea.get('caption_brief', '')}\n\n"
                f"{idea.get('cta', '')}"
            ).strip()

            meta_info = {
                "is_planned_idea": True,
                "auto_generate_media": True,
                "day_number": idea.get("day_number", 1),
                "slot_index": idea.get("slot_index", 0),
                "hook": idea.get("hook", ""),
                "caption_brief": idea.get("caption_brief", ""),
                "description": full_desc,
                "custom_focus": idea.get("custom_focus", plan_data.get("custom_focus", "")),
                "visual_prompt_direction": (idea.get("visual_prompt_direction") or idea.get("image_prompt") or "").strip(),
                "image_prompt": (idea.get("visual_prompt_direction") or idea.get("image_prompt") or "").strip(),
                "content_format": idea.get("content_format", "image"),
                "pillar": idea.get("pillar", "Strategic Focus"),
                "cta": idea.get("cta", ""),
                "suggested_date": idea.get("suggested_date", ""),
                "suggested_time": idea.get("suggested_time", "10:00"),
                "suggested_datetime": idea.get("suggested_datetime", ""),
                "plan_status": idea.get("plan_status", "planned")
            }

            caption_content = full_desc

            if not content_record:
                content_record = Content(
                    title=idea.get("title", "Untitled Planned Idea"),
                    caption=caption_content,
                    hashtags=idea.get("hashtags", []),
                    target_platforms=normalized_platforms,
                    status=ContentStatus.DRAFT,
                    meta_info=meta_info,
                    organization_id=organization_id,
                    brand_profile_id=brand_id,
                    campaign_id=campaign.id,
                    creator_id=current_user.id
                )
                db.add(content_record)
                db.flush()
                db.refresh(content_record)
            else:
                content_record.title = idea.get("title", content_record.title)
                content_record.caption = caption_content
                content_record.hashtags = idea.get("hashtags", content_record.hashtags)
                content_record.target_platforms = normalized_platforms
                # Merge existing meta
                existing_meta = dict(content_record.meta_info or {})
                existing_meta.update(meta_info)
                content_record.meta_info = existing_meta
                db.flush()
                db.refresh(content_record)

            if idea.get("media_asset_ids"):
                cls._attach_generated_media(db, content_record, idea["media_asset_ids"], organization_id)

            if meta_info.get("suggested_date"):
                cls._sync_scheduled_generation_job(
                    db=db,
                    content=content_record,
                    organization_id=organization_id,
                    user_id=current_user.id,
                    timezone_str=campaign_tags.get("timezone", "Asia/Kolkata"),
                    force_sync=True
                )
            primary_platform = [p.value if hasattr(p, "value") else str(p) for p in content_record.target_platforms][0] if content_record.target_platforms else "Instagram"
            saved_ideas.append({
                "id": content_record.id,
                "day_number": meta_info["day_number"],
                "title": content_record.title,
                "topic": content_record.title,
                "hook": meta_info["hook"],
                "description": content_record.caption or meta_info.get("description", ""),
                "caption": content_record.caption or meta_info.get("description", ""),
                "caption_brief": meta_info["caption_brief"],
                "content_format": meta_info["content_format"],
                "platform": primary_platform,
                "platforms": [p.value if hasattr(p, "value") else str(p) for p in content_record.target_platforms],
                "suggested_date": meta_info["suggested_date"],
                "suggested_time": meta_info["suggested_time"],
                "suggested_datetime": meta_info["suggested_datetime"],
                "pillar": meta_info["pillar"],
                "content_pillar": meta_info["pillar"],
                "hashtags": content_record.hashtags or [],
                "cta": meta_info["cta"],
                "call_to_action": meta_info["cta"],
                "custom_focus": meta_info.get("custom_focus", ""),
                "visual_prompt_direction": meta_info.get("visual_prompt_direction", ""),
                "image_prompt": meta_info.get("image_prompt", ""),
                "plan_status": meta_info["plan_status"]
            })

        db.commit()
        return {
            "campaign_id": campaign.id,
            "campaign": {
                "id": campaign.id,
                "name": campaign.title,
                "title": campaign.title,
                "goal": campaign.objective,
                "tags": campaign_tags,
                "start_date": campaign.start_date.isoformat() if campaign.start_date else None,
                "end_date": campaign.end_date.isoformat() if campaign.end_date else None
            },
            "title": campaign.title,
            "summary": campaign.objective,
            "duration_days": campaign_tags.get("duration_days", 7),
            "posting_frequency": campaign_tags.get("posting_frequency", "1_per_day"),
            "campaign_goal": campaign_tags.get("campaign_goal", "product_launch"),
            "platforms": campaign_tags.get("platforms", []),
            "start_date": campaign.start_date.strftime("%Y-%m-%d") if campaign.start_date else None,
            "end_date": campaign.end_date.strftime("%Y-%m-%d") if campaign.end_date else None,
            "total_ideas": len(saved_ideas),
            "count": len(saved_ideas),
            "ideas": saved_ideas
        }

    @classmethod
    def get_campaign_ideas(cls, db: Session, organization_id: int, campaign_id: int) -> Dict[str, Any]:
        """Fetch saved campaign and its associated Content ideas directly from database."""
        start_time = time.time()
        logger.info(f"[ContentPlanner] Loading campaign ideas for campaign_id={campaign_id}, org={organization_id}")

        campaign = db.query(Campaign).filter(
            Campaign.id == campaign_id,
            Campaign.organization_id == organization_id
        ).first()

        if not campaign:
            logger.warning(f"[ContentPlanner] Campaign plan {campaign_id} not found for org {organization_id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Campaign plan not found.")

        contents = db.query(Content).filter(
            Content.campaign_id == campaign_id,
            Content.organization_id == organization_id
        ).order_by(Content.id.asc()).all()

        tags = campaign.tags if isinstance(campaign.tags, dict) else {}

        ideas = []
        for c in contents:
            meta = c.meta_info if isinstance(c.meta_info, dict) else {}
            raw_platforms = c.target_platforms if isinstance(c.target_platforms, (list, tuple)) else []
            c_plat = [p.value if hasattr(p, "value") else str(p) for p in raw_platforms]
            primary_p = c_plat[0] if c_plat else "Instagram"
            full_text = meta.get("description") or c.caption or ""
            ideas.append({
                "id": c.id,
                "generated_media": meta.get("generated_media", {}),
                "day_number": meta.get("day_number", 1),
                "title": c.title or "Untitled Idea",
                "topic": c.title or "Untitled Idea",
                "hook": meta.get("hook", ""),
                "description": full_text,
                "caption": full_text,
                "caption_brief": meta.get("caption_brief", full_text[:120] if full_text else ""),
                "content_format": meta.get("content_format", "image"),
                "platform": primary_p,
                "platforms": c_plat,
                "suggested_date": meta.get("suggested_date", ""),
                "suggested_time": meta.get("suggested_time", "10:00"),
                "suggested_datetime": meta.get("suggested_datetime", ""),
                "pillar": meta.get("pillar", "Strategic Pillar"),
                "content_pillar": meta.get("pillar", "Strategic Pillar"),
                "hashtags": c.hashtags or [],
                "cta": meta.get("cta", ""),
                "call_to_action": meta.get("cta", ""),
                "custom_focus": meta.get("custom_focus", tags.get("custom_focus", "")),
                "plan_status": meta.get("plan_status", "planned"),
                "scheduled_job_id": meta.get("scheduled_job_id"),
                "scheduled_time_utc": meta.get("scheduled_time_utc"),
                "generation_status": meta.get("generation_status", "scheduled" if meta.get("suggested_date") else "idle"),
                "generation_error": meta.get("generation_error"),
                "is_generating": bool(meta.get("auto_media_in_progress") or meta.get("generation_status") == "processing")
            })

        # Sort by day_number or suggested_date
        ideas.sort(key=lambda x: (x.get("day_number", 0), x.get("suggested_date", "")))

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[ContentPlanner] Successfully loaded {len(ideas)} ideas for campaign {campaign_id} "
            f"in {elapsed_ms:.1f}ms (org={organization_id})"
        )

        return {
            "campaign_id": campaign.id,
            "campaign": {
                "id": campaign.id,
                "name": campaign.title,
                "title": campaign.title,
                "goal": campaign.objective,
                "tags": tags,
                "start_date": campaign.start_date.isoformat() if campaign.start_date else None,
                "end_date": campaign.end_date.isoformat() if campaign.end_date else None
            },
            "title": campaign.title,
            "summary": campaign.objective,
            "duration_days": tags.get("duration_days", 7),
            "posting_frequency": tags.get("posting_frequency", "1_per_day"),
            "campaign_goal": tags.get("campaign_goal", "product_launch"),
            "platforms": tags.get("platforms", ["instagram", "facebook", "youtube"]),
            "start_date": campaign.start_date.strftime("%Y-%m-%d") if campaign.start_date else None,
            "end_date": campaign.end_date.strftime("%Y-%m-%d") if campaign.end_date else None,
            "total_ideas": len(ideas),
            "count": len(ideas),
            "ideas": ideas
        }

    @classmethod
    def update_idea(
        cls,
        db: Session,
        organization_id: int,
        content_id: int,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update fields or status of a single planned idea."""
        content = db.query(Content).filter(
            Content.id == content_id,
            Content.organization_id == organization_id
        ).first()

        if not content:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Planned idea not found.")

        if payload.get("media_asset_ids"):
            cls._attach_generated_media(db, content, payload["media_asset_ids"], organization_id)
            # Synchronize any active GenerationJob so it doesn't redundantly re-generate
            try:
                active_jobs = db.query(GenerationJob).filter(
                    GenerationJob.organization_id == organization_id,
                    GenerationJob.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
                ).all()
                for aj in active_jobs:
                    if isinstance(aj.parameters, dict) and aj.parameters.get("content_id") == content.id:
                        aj.status = JobStatus.COMPLETED
                        aj.result_data = {"media_asset_ids": payload["media_asset_ids"], "completed_via": "manual_or_direct_attach"}
            except Exception as sync_job_err:
                logger.debug(f"[ContentPlannerService] Notice completing job on manual attach: {sync_job_err}")

        if "title" in payload:
            content.title = payload["title"]
        elif "topic" in payload:
            content.title = payload["topic"]

        if "hashtags" in payload:
            content.hashtags = payload["hashtags"]

        if "platforms" in payload:
            norm = []
            for p in payload["platforms"]:
                try:
                    norm.append(PlatformType(p.lower()))
                except Exception:
                    pass
            if norm:
                content.target_platforms = norm
        elif "platform" in payload and payload["platform"]:
            try:
                content.target_platforms = [PlatformType(payload["platform"].lower())]
            except Exception:
                pass

        meta = dict(content.meta_info or {})
        for field in ["hook", "caption_brief", "content_format", "pillar", "content_pillar", "cta", "call_to_action", "suggested_date", "suggested_time", "plan_status"]:
            if field in payload:
                target_field = "pillar" if field == "content_pillar" else ("cta" if field == "call_to_action" else field)
                meta[target_field] = payload[field]

        if "suggested_date" in payload or "suggested_time" in payload:
            meta["generation_status"] = "scheduled"
            meta["generation_error"] = None
            meta["auto_media_attempt_count"] = 0
            meta["auto_media_in_progress"] = False
            d = meta.get("suggested_date", "")
            t = meta.get("suggested_time", "10:00")
            if d:
                meta["suggested_datetime"] = f"{d}T{t}:00"
                # Keep any associated ScheduledPost aligned with the new timing
                try:
                    sp = db.query(ScheduledPost).filter(ScheduledPost.content_id == content.id).first()
                    import pytz
                    tz_str = getattr(sp, "timezone", "Asia/Kolkata") if sp else "Asia/Kolkata"
                    tz = pytz.timezone(tz_str or "Asia/Kolkata")
                    naive_dt = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M")
                    local_dt = tz.localize(naive_dt)
                    target_utc = local_dt.astimezone(pytz.utc).replace(tzinfo=None)

                    if sp:
                        sp.scheduled_time = target_utc
                        sp.status = "scheduled"
                    else:
                        social_acc = db.query(SocialAccount).filter(
                            SocialAccount.organization_id == organization_id,
                            SocialAccount.is_connected == True
                        ).first()
                        camp = db.query(Campaign).filter(Campaign.id == content.campaign_id).first() if content.campaign_id else None
                        camp_tags = camp.tags if camp and isinstance(camp.tags, dict) else {}

                        pub_meta = {
                            "auto_generate_media": True,
                            "campaign_id": content.campaign_id,
                            "auto_approve": True,
                            "custom_focus": meta.get("custom_focus", ""),
                            "day_number": meta.get("day_number", 1),
                            "pillar": meta.get("pillar", ""),
                            "media_format_preference": camp_tags.get("media_format_preference", "both") if camp else meta.get("media_format_preference", "both")
                        }

                        fmt_str = str(meta.get("content_format", "image")).lower()
                        post_fmt = "reel" if "video" in fmt_str or "reel" in fmt_str else "post"
                        c_plat = content.target_platforms[0].value if content.target_platforms and hasattr(content.target_platforms[0], "value") else "instagram"

                        sp = ScheduledPost(
                            title=content.title,
                            caption=content.caption or meta.get("description", ""),
                            hashtags=content.hashtags or [],
                            internal_tags=["ai_planner", f"campaign_{content.campaign_id}"] if content.campaign_id else ["ai_planner"],
                            platform=c_plat.lower(),
                            post_format=post_fmt,
                            scheduled_time=target_utc,
                            status="scheduled",
                            timezone=tz_str,
                            publish_metadata=pub_meta,
                            content_id=content.id,
                            social_account_id=social_acc.id if social_acc else None,
                            organization_id=organization_id,
                            user_id=content.creator_id
                        )
                        db.add(sp)
                        content.status = ContentStatus.SCHEDULED
                        meta["plan_status"] = "scheduled"
                except Exception as sp_timing_err:
                    logger.debug(f"[ContentPlannerService] ScheduledPost timing update notice: {sp_timing_err}")

        if "description" in payload:
            content.caption = payload["description"]
            meta["description"] = payload["description"]
        elif "caption" in payload:
            content.caption = payload["caption"]
            meta["description"] = payload["caption"]
        elif "hook" in payload or "caption_brief" in payload or "cta" in payload or "call_to_action" in payload:
            content.caption = f"{meta.get('hook', '')}\n\n{meta.get('caption_brief', '')}\n\n{meta.get('cta', '')}".strip()
            meta["description"] = content.caption

        # Synchronize persistent GenerationJob whenever timing, title, or format changes
        if meta.get("suggested_date"):
            try:
                camp = db.query(Campaign).filter(Campaign.id == content.campaign_id).first() if content.campaign_id else None
                camp_tz = (camp.tags or {}).get("timezone", "Asia/Kolkata") if camp and isinstance(camp.tags, dict) else "Asia/Kolkata"
                cls._sync_scheduled_generation_job(
                    db=db,
                    content=content,
                    organization_id=organization_id,
                    user_id=content.creator_id,
                    suggested_date=meta.get("suggested_date"),
                    suggested_time=meta.get("suggested_time", "10:00"),
                    timezone_str=camp_tz,
                    content_format=meta.get("content_format", "image")
                )
            except Exception as sync_gen_err:
                logger.warning(f"[ContentPlannerService] Scheduled generation job sync error: {sync_gen_err}")

        content.meta_info = meta

        db.commit()
        db.refresh(content)

        c_plat = [p.value if hasattr(p, "value") else str(p) for p in content.target_platforms]
        primary_p = c_plat[0] if c_plat else "Instagram"

        full_text = meta.get("description") or content.caption or ""
        return {
            "id": content.id,
            "title": content.title,
            "topic": content.title,
            "hook": meta.get("hook", ""),
            "description": full_text,
            "caption": full_text,
            "caption_brief": meta.get("caption_brief", full_text[:120] if full_text else ""),
            "content_format": meta.get("content_format", "image"),
            "platform": primary_p,
            "platforms": c_plat,
            "suggested_date": meta.get("suggested_date", ""),
            "suggested_time": meta.get("suggested_time", "10:00"),
            "suggested_datetime": meta.get("suggested_datetime", ""),
            "pillar": meta.get("pillar", ""),
            "content_pillar": meta.get("pillar", ""),
            "hashtags": content.hashtags or [],
            "cta": meta.get("cta", ""),
            "call_to_action": meta.get("cta", ""),
            "plan_status": meta.get("plan_status", "planned"),
            "scheduled_job_id": meta.get("scheduled_job_id"),
            "generation_status": meta.get("generation_status", "scheduled" if meta.get("suggested_date") else "idle"),
            "generation_error": meta.get("generation_error"),
            "is_generating": bool(meta.get("auto_media_in_progress") or meta.get("generation_status") == "processing"),
            "generated_media": meta.get("generated_media", {})
        }

    @classmethod
    def delete_idea(cls, db: Session, organization_id: int, content_id: int) -> bool:
        """
        Delete a planned idea from the plan, cancelling any pending/active background
        generation jobs, removing associated scheduled posts, and cleaning up temporary media.
        """
        content = db.query(Content).filter(
            Content.id == content_id,
            Content.organization_id == organization_id
        ).first()

        if not content:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Planned idea not found.")

        # 1. Cancel all active and pending GenerationJobs associated with this content
        active_jobs = db.query(GenerationJob).filter(
            GenerationJob.organization_id == organization_id,
            GenerationJob.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
        ).all()
        for job in active_jobs:
            job_c_id = (job.parameters or {}).get("content_id") if isinstance(job.parameters, dict) else None
            scheduled_job_id = (content.meta_info or {}).get("scheduled_job_id")
            if job_c_id == content_id or job.id == scheduled_job_id:
                job.status = JobStatus.FAILED
                job.error_message = f"Cancelled: Parent content #{content_id} was deleted."
                if isinstance(job.parameters, dict):
                    job.parameters["cancelled"] = True
                    job.parameters["retries"] = 999
                logger.info(f"[ContentPlanner] Cancelled GenerationJob #{job.id} for deleted content #{content_id}.")

        # 2. Clean up associated ScheduledPosts
        sps = db.query(ScheduledPost).filter(ScheduledPost.content_id == content_id).all()
        for sp in sps:
            db.delete(sp)

        # 3. Clean up unlinked media assets and temporary files for this content
        meta = dict(content.meta_info or {})
        gen_media = meta.get("generated_media", {})
        for m_type in ("image", "video"):
            m_info = gen_media.get(m_type, {})
            a_id = m_info.get("media_asset_id")
            if a_id:
                try:
                    asset = db.query(MediaAsset).filter(MediaAsset.id == a_id).first()
                    if asset and not asset.campaign_id:
                        f_url = asset.file_url or ""
                        if f_url.startswith("/uploads/") or "/media/" in f_url:
                            clean_rel = f_url.lstrip("/").replace("uploads/", "media/")
                            local_p = os.path.join(settings.LOCAL_STORAGE_DIR, clean_rel)
                            if os.path.exists(local_p):
                                try:
                                    os.remove(local_p)
                                except Exception:
                                    pass
                        db.delete(asset)
                except Exception as clean_err:
                    logger.warning(f"[ContentPlanner] Notice cleaning up asset #{a_id} during delete: {clean_err}")

        db.delete(content)
        db.commit()
        return True

    @classmethod
    async def regenerate_single_idea(
        cls,
        db: Session,
        organization_id: int,
        content_id: int,
        allow_degraded_fallback: bool = False
    ) -> Dict[str, Any]:
        """Regenerate a single planned idea with fresh AI output while preserving schedule date/slot."""
        content = db.query(Content).filter(
            Content.id == content_id,
            Content.organization_id == organization_id
        ).first()

        if not content:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Planned idea not found.")

        brand = db.query(BrandProfile).filter(BrandProfile.id == content.brand_profile_id).first()
        validate_studio_brand(brand)
        brand_name = brand.name
        industry = brand.industry if brand else "Retail"
        tone = brand.tone_of_voice if brand else "Engaging"
        meta = dict(content.meta_info or {})

        # Fetch sibling ideas to ensure uniqueness
        sibling_ideas = db.query(Content).filter(
            Content.campaign_id == content.campaign_id,
            Content.id != content_id
        ).all()
        existing_titles = [c.title for c in sibling_ideas if c.title]

        system_instruction = (
            f"Verified brand facts: {studio_brand_context(brand)}. Never invent products, offers, or business details.\n"
            f"You are the social media copywriter for '{brand_name}'. Tone: {tone}. Industry: {industry}.\n"
            f"Generate an original, high-impact social media post idea. Avoid repeating: {', '.join(existing_titles[:8])}.\n"
            f"Return valid JSON strictly matching the requested schema."
        )

        user_prompt = (
            f"Generate a creative post idea for slot day {meta.get('day_number', 1)}.\n"
            f"Format needed: {meta.get('content_format', 'image')}.\n"
            f"Return JSON with: title (string), description (complete ready-to-post caption string), hook (string), caption_brief (string), pillar (string), suggested_hashtags (list of strings), cta (string)."
        )

        brand_info_payload = {
            "name": brand_name,
            "industry": industry,
            "tone_of_voice": tone,
            "target_audience": getattr(brand, "target_audience", "Social audience"),
            "usp": getattr(brand, "usp", "Exceptional quality and craft"),
            "cta": getattr(brand, "cta", "Tap link in bio to learn more"),
            "products": (brand.meta_info or {}).get("products", []) if isinstance(getattr(brand, "meta_info", None), dict) else [],
            "key_themes": brand.key_themes if (brand and brand.key_themes) else ["Quality", "Craftsmanship"],
            "guidelines": brand.brand_guidelines or "",
            "avoid": getattr(brand, "avoid", "")
        }

        def _validate_single_idea(data: Dict[str, Any]) -> bool:
            if not isinstance(data, dict):
                return False
            # Check if direct idea object
            if all(isinstance(data.get(f), str) and data[f].strip() for f in ("title", "hook", "caption_brief", "description")):
                return True
            # Or wrapped in ideas list
            ideas = data.get("ideas")
            if isinstance(ideas, list) and len(ideas) > 0 and isinstance(ideas[0], dict):
                return all(isinstance(ideas[0].get(f), str) and ideas[0][f].strip() for f in ("title", "hook", "caption_brief", "description"))
            return False

        res = await AIProviderManager.generate_json_with_resilience(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            brand_info=brand_info_payload,
            duration_days=1,
            slot_count=1,
            target_platforms=[p.value if hasattr(p, "value") else str(p) for p in content.target_platforms],
            campaign_goal=f"Regenerate slot {meta.get('day_number', 1)}",
            validation_fn=_validate_single_idea,
            allow_degraded_fallback=allow_degraded_fallback
        )

        new_data = res
        if isinstance(res, dict) and isinstance(res.get("ideas"), list) and len(res["ideas"]) > 0:
            new_data = res["ideas"][0]

        if not isinstance(new_data, dict) or not all(
            isinstance(new_data.get(field), str) and new_data[field].strip()
            for field in ("title", "hook", "caption_brief", "description")
        ):
            raise HTTPException(status_code=503, detail="AI regeneration failed. Your existing idea has been preserved. Please retry.")

        content.title = new_data.get("title", content.title)
        content.hashtags = new_data.get("suggested_hashtags", content.hashtags)
        meta["hook"] = new_data.get("hook", meta.get("hook"))
        meta["caption_brief"] = new_data.get("caption_brief", meta.get("caption_brief"))
        meta["pillar"] = new_data.get("pillar", meta.get("pillar"))
        meta["cta"] = new_data.get("cta", meta.get("cta"))
        meta["plan_status"] = "planned"
        content.meta_info = meta
        content.caption = new_data["description"]
        meta["description"] = content.caption

        db.commit()
        db.refresh(content)

        c_plat = [p.value if hasattr(p, "value") else str(p) for p in content.target_platforms]
        primary_p = c_plat[0] if c_plat else "Instagram"

        return {
            "id": content.id,
            "title": content.title,
            "topic": content.title,
            "description": content.caption,
            "caption": content.caption,
            "hook": meta["hook"],
            "caption_brief": meta["caption_brief"],
            "content_format": meta.get("content_format", "image"),
            "platform": primary_p,
            "platforms": c_plat,
            "suggested_date": meta.get("suggested_date", ""),
            "suggested_time": meta.get("suggested_time", "10:00"),
            "suggested_datetime": meta.get("suggested_datetime", ""),
            "pillar": meta["pillar"],
            "content_pillar": meta["pillar"],
            "hashtags": content.hashtags,
            "cta": meta["cta"],
            "call_to_action": meta["cta"],
            "plan_status": meta["plan_status"]
        }


    @classmethod
    def schedule_campaign_plan(
        cls,
        db: Session,
        organization_id: int,
        campaign_id: int,
        current_user: User,
        auto_approve: bool = True,
        timezone_str: str = "Asia/Kolkata",
        target_platform: Optional[str] = None,
        start_date: Optional[str] = None,
        daily_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Auto-schedules all ideas in a campaign plan into publishing pipeline with Auto-Post or Approval Required workflow."""
        campaign = db.query(Campaign).filter(
            Campaign.id == campaign_id,
            Campaign.organization_id == organization_id
        ).first()

        if not campaign:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Campaign plan not found.")

        contents = db.query(Content).filter(
            Content.campaign_id == campaign_id,
            Content.organization_id == organization_id
        ).all()

        if not contents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No planned ideas found in this campaign to schedule.")

        # Sort chronologically by day_number
        contents.sort(key=lambda c: ((c.meta_info or {}).get("day_number", 0), c.id))

        # Check for active social account for publishing
        social_acc = db.query(SocialAccount).filter(
            SocialAccount.organization_id == organization_id,
            SocialAccount.is_connected == True
        ).first()

        scheduled_posts_result = []
        now_utc = datetime.utcnow()

        for idx, c in enumerate(contents):
            meta = dict(c.meta_info or {})

            if start_date:
                try:
                    s_dt_obj = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=idx)
                    s_date = s_dt_obj.strftime("%Y-%m-%d")
                    meta["suggested_date"] = s_date
                except Exception:
                    s_date = meta.get("suggested_date", "")
            else:
                s_date = meta.get("suggested_date", "")

            if daily_time:
                s_time = daily_time
                meta["suggested_time"] = s_time
            else:
                s_time = meta.get("suggested_time", "10:00")

            c.meta_info = meta

            # Calculate scheduled datetime
            scheduled_dt = None
            if s_date and s_time:
                try:
                    import pytz
                    tz = pytz.timezone(timezone_str or "Asia/Kolkata")
                    naive_dt = datetime.strptime(f"{s_date} {s_time}", "%Y-%m-%d %H:%M")
                    local_dt = tz.localize(naive_dt)
                    scheduled_dt = local_dt.astimezone(pytz.utc).replace(tzinfo=None)
                except Exception:
                    pass

            if not scheduled_dt:
                # If no timing provided, schedule daily starting tomorrow at 10:00 AM
                scheduled_dt = now_utc.replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(days=idx + 1)
            elif scheduled_dt <= now_utc and idx == 0:
                # For Day 1, if scheduled for today and the minute already arrived or passed, trigger immediately!
                scheduled_dt = now_utc

            # Determine platform and format
            c_plat = target_platform or (c.target_platforms[0].value if c.target_platforms and hasattr(c.target_platforms[0], "value") else "instagram")
            post_fmt = "post"
            fmt_str = str(meta.get("content_format", "image")).lower()
            if "video" in fmt_str or "reel" in fmt_str:
                post_fmt = "reel"
            elif "carousel" in fmt_str:
                # Safe backward compatibility: map any planned carousel tag to standard post
                post_fmt = "post"

            # Workflow mode: auto-post vs approval required
            if auto_approve:
                post_status = "scheduled"
                c.status = ContentStatus.SCHEDULED
                meta["plan_status"] = "scheduled"
            else:
                post_status = "pending_approval"
                c.status = ContentStatus.PENDING_APPROVAL
                meta["plan_status"] = "awaiting_approval"

                # Create or update pending approval record
                existing_approval = db.query(Approval).filter(Approval.content_id == c.id).first()
                if not existing_approval:
                    apprv = Approval(
                        content_id=c.id,
                        requester_id=current_user.id,
                        status=ApprovalStatus.PENDING,
                        comment=f"Campaign '{campaign.title}' Day {meta.get('day_number', idx + 1)} post awaiting editorial approval."
                    )
                    db.add(apprv)
                else:
                    existing_approval.status = ApprovalStatus.PENDING

            # Create or update ScheduledPost record
            sp = db.query(ScheduledPost).filter(ScheduledPost.content_id == c.id).first()
            pub_meta = {
                "auto_generate_media": True,
                "campaign_id": campaign.id,
                "auto_approve": auto_approve,
                "custom_focus": meta.get("custom_focus", ""),
                "day_number": meta.get("day_number", idx + 1),
                "pillar": meta.get("pillar", ""),
                "media_format_preference": campaign.tags.get("media_format_preference", "both") if isinstance(campaign.tags, dict) else meta.get("media_format_preference", "both"),
            }

            if not sp:
                sp = ScheduledPost(
                    title=c.title,
                    caption=c.caption or meta.get("description", ""),
                    hashtags=c.hashtags or [],
                    internal_tags=["ai_planner", f"campaign_{campaign.id}"],
                    platform=c_plat.lower(),
                    post_format=post_fmt,
                    scheduled_time=scheduled_dt,
                    status=post_status,
                    timezone=timezone_str or "Asia/Kolkata",
                    publish_metadata=pub_meta,
                    content_id=c.id,
                    social_account_id=social_acc.id if social_acc else None,
                    organization_id=organization_id,
                    user_id=current_user.id
                )
                db.add(sp)
            else:
                sp.title = c.title
                sp.caption = c.caption or meta.get("description", "")
                sp.hashtags = c.hashtags or []
                sp.platform = c_plat.lower()
                sp.post_format = post_fmt
                sp.scheduled_time = scheduled_dt
                sp.status = post_status
                sp.timezone = timezone_str or "Asia/Kolkata"
                merged_meta = dict(sp.publish_metadata or {})
                merged_meta.update(pub_meta)
                sp.publish_metadata = merged_meta

            # Synchronize persistent GenerationJob for scheduled media generation
            if meta.get("suggested_date"):
                cls._sync_scheduled_generation_job(
                    db=db,
                    content=c,
                    organization_id=organization_id,
                    user_id=current_user.id,
                    suggested_date=meta.get("suggested_date"),
                    suggested_time=meta.get("suggested_time", "10:00"),
                    timezone_str=timezone_str or "Asia/Kolkata",
                    content_format=meta.get("content_format", "image")
                )

            c.meta_info = meta
            scheduled_posts_result.append({
                "content_id": c.id,
                "day_number": meta.get("day_number", idx + 1),
                "title": c.title,
                "status": post_status,
                "scheduled_time": scheduled_dt.isoformat(),
                "platform": c_plat,
                "post_format": post_fmt
            })

        campaign.status = "scheduled" if auto_approve else "in_review"
        db.commit()

        logger.info(
            f"[ContentPlannerService] Scheduled {len(scheduled_posts_result)} posts for campaign #{campaign_id} "
            f"(mode={'Auto-Post' if auto_approve else 'Approval-Required'})."
        )

        return {
            "success": True,
            "campaign_id": campaign.id,
            "campaign_title": campaign.title,
            "mode": "auto_post" if auto_approve else "approval_required",
            "scheduled_count": len(scheduled_posts_result),
            "posts": scheduled_posts_result,
            "message": (
                f"Successfully scheduled all {len(scheduled_posts_result)} days into the publishing pipeline "
                f"({'Auto-Post active: posts will automatically synthesize media and publish at scheduled times' if auto_approve else 'Approval Required: posts placed in Content Approvals queue'})."
            )
        }

    _is_auto_generating = False

    @classmethod
    async def auto_generate_due_planned_media(cls, target_content_id: Optional[int] = None) -> int:
        """
        Continuously monitors and automatically executes scheduled AI media generation tasks
        for planned campaign ideas at their exact configured date and time.
        Supports Cloudflare Workers AI image generation, video reel generation, and multi-asset bundles.
        Works independently when browser is closed, user is logged out, and without manual clicks.
        """
        if cls._is_auto_generating and not target_content_id:
            return 0
        if not target_content_id:
            cls._is_auto_generating = True

        from app.db.session import SessionLocal
        db = SessionLocal()
        processed_count = 0
        try:
            import pytz
            now_utc = datetime.utcnow()

            # -------------------------------------------------------------
            # Stage 1: Recovery of Interrupted / Stuck Processing Jobs (> 5 mins)
            # -------------------------------------------------------------
            stuck_cutoff = now_utc - timedelta(minutes=5)
            processing_jobs = db.query(GenerationJob).filter(
                GenerationJob.status == JobStatus.PROCESSING,
                GenerationJob.updated_at <= stuck_cutoff
            ).all()

            for stuck in processing_jobs:
                if not isinstance(stuck.parameters, dict) or not stuck.parameters.get("is_scheduled_media_generation"):
                    continue
                retries = int(stuck.parameters.get("retries", 0))
                c_id = stuck.parameters.get("content_id")
                c_rec = db.query(Content).filter(Content.id == c_id).first() if c_id else None
                if not c_rec:
                    stuck.status = JobStatus.FAILED
                    stuck.error_message = f"Cancelled: Parent content #{c_id} no longer exists."
                    if isinstance(stuck.parameters, dict):
                        stuck.parameters["cancelled"] = True
                        stuck.parameters["retries"] = 999
                    logger.info(f"[AutoMediaScheduler] Interrupted job #{stuck.id} cancelled because content #{c_id} was deleted.")
                elif retries < 3:
                    stuck.status = JobStatus.PENDING
                    stuck.parameters["retries"] = retries + 1
                    c_meta = dict(c_rec.meta_info or {})
                    c_meta["generation_status"] = "retrying"
                    c_meta["auto_media_in_progress"] = False
                    c_rec.meta_info = c_meta
                    logger.info(f"[AutoMediaScheduler] Recovered interrupted job #{stuck.id} for content #{c_id} back to PENDING (retry {retries + 1}/3).")
                else:
                    stuck.status = JobStatus.FAILED
                    stuck.error_message = "Job was interrupted during server shutdown and exceeded maximum retries (3)."
                    c_meta = dict(c_rec.meta_info or {})
                    c_meta["generation_status"] = "failed"
                    c_meta["generation_error"] = stuck.error_message
                    c_meta["auto_media_in_progress"] = False
                    c_rec.meta_info = c_meta
                    logger.error(f"[AutoMediaScheduler] Interrupted job #{stuck.id} for content #{c_id} exceeded max retries. Marked as FAILED.")
            db.commit()

            # -------------------------------------------------------------
            # Stage 2: Clean up orphaned jobs & backfill only opted-in ideas
            # -------------------------------------------------------------
            # Cancel any existing jobs for deleted content
            active_jobs = db.query(GenerationJob).filter(
                GenerationJob.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
            ).all()
            for aj in active_jobs:
                if isinstance(aj.parameters, dict) and aj.parameters.get("content_id"):
                    job_c_id = aj.parameters["content_id"]
                    if not db.query(Content.id).filter(Content.id == job_c_id).first():
                        aj.status = JobStatus.FAILED
                        aj.error_message = f"Cancelled: Parent content #{job_c_id} was deleted."
                        aj.parameters["cancelled"] = True
                        aj.parameters["retries"] = 999
                        logger.info(f"[AutoMediaScheduler] Cleaned up orphaned job #{aj.id} for deleted content #{job_c_id}.")
            db.commit()

            planned_contents = db.query(Content).filter(
                Content.campaign_id.isnot(None)
            ).all()
            for pc in planned_contents:
                pc_meta = dict(pc.meta_info or {})
                if not pc_meta.get("is_planned_idea") and not pc.campaign_id:
                    continue

                # Enforce explicit user opt-in: do not backfill unless auto_generate_media is True (or targeted manual trigger)
                is_target_match = bool(target_content_id and pc.id == target_content_id)
                sp = db.query(ScheduledPost).filter(ScheduledPost.content_id == pc.id).first()
                camp = db.query(Campaign).filter(Campaign.id == pc.campaign_id).first() if pc.campaign_id else None
                is_auto_gen = bool(
                    is_target_match or
                    pc_meta.get("auto_generate_media") is not False or
                    (sp and (sp.publish_metadata or {}).get("auto_generate_media") is not False) or
                    (camp and isinstance(camp.tags, dict) and camp.tags.get("auto_generate_media") is not False) or
                    (pc_meta.get("is_planned_idea") and pc_meta.get("suggested_date"))
                )
                if not is_auto_gen:
                    continue

                fmt_str = str(pc_meta.get("content_format", "image")).lower()
                if "text" in fmt_str and "image" not in fmt_str and "video" not in fmt_str:
                    continue

                gen_m = pc_meta.get("generated_media") or {}
                has_img = bool(gen_m.get("image", {}).get("image_url"))
                has_vid = bool(gen_m.get("video", {}).get("video_url"))
                is_vid_req = "video" in fmt_str or "reel" in fmt_str
                if is_vid_req and has_vid:
                    continue
                if not is_vid_req and has_img:
                    continue

                # If content has suggested_date but no job, sync it
                if pc_meta.get("suggested_date"):
                    existing_job = None
                    if pc_meta.get("scheduled_job_id"):
                        existing_job = db.query(GenerationJob).filter(GenerationJob.id == pc_meta["scheduled_job_id"]).first()
                    if not existing_job:
                        cls._sync_scheduled_generation_job(db, pc, pc.organization_id, pc.creator_id, force_sync=is_target_match)
            db.commit()

            # -------------------------------------------------------------
            # Stage 3: Query Due Scheduled Jobs
            # -------------------------------------------------------------
            candidate_jobs_query = db.query(GenerationJob).filter(
                GenerationJob.status == JobStatus.PENDING
            )
            if target_content_id:
                candidate_jobs_query = candidate_jobs_query.order_by(GenerationJob.id.desc())
            else:
                candidate_jobs_query = candidate_jobs_query.order_by(GenerationJob.id.asc())

            candidates = candidate_jobs_query.all()

            for job in candidates:
                if not target_content_id and processed_count >= 5:
                    break

                if not isinstance(job.parameters, dict) or not job.parameters.get("is_scheduled_media_generation"):
                    continue

                content_id = job.parameters.get("content_id")
                if not content_id:
                    continue

                if target_content_id and content_id != target_content_id:
                    continue

                # Protect deleted content: check if content exists before proceeding
                content_check = db.query(Content).filter(Content.id == content_id).first()
                if not content_check:
                    job.status = JobStatus.FAILED
                    job.error_message = f"Cancelled: Parent content #{content_id} was deleted."
                    job.parameters["cancelled"] = True
                    job.parameters["retries"] = 999
                    db.commit()
                    logger.info(f"[AutoMediaScheduler] Cancelled job #{job.id} because parent content #{content_id} no longer exists.")
                    continue

                # Control automatic background generation: verify user opt-in when running via background worker
                if target_content_id is None:
                    c_meta = dict(content_check.meta_info or {})
                    sp_check = db.query(ScheduledPost).filter(ScheduledPost.content_id == content_id).first()
                    camp_check = db.query(Campaign).filter(Campaign.id == content_check.campaign_id).first() if content_check.campaign_id else None
                    is_auto_gen = bool(
                        c_meta.get("auto_generate_media") is not False or
                        (sp_check and (sp_check.publish_metadata or {}).get("auto_generate_media") is not False) or
                        (camp_check and isinstance(camp_check.tags, dict) and camp_check.tags.get("auto_generate_media") is not False) or
                        (c_meta.get("is_planned_idea") and c_meta.get("suggested_date"))
                    )
                    if not is_auto_gen:
                        logger.info(f"[AutoMediaScheduler] Skipping Job #{job.id} for content #{content_id}: auto_generate_media is not enabled.")
                        continue

                # Check if media already exists (prevent duplicate generation)
                c_meta = dict(content_check.meta_info or {})
                gen_m = c_meta.get("generated_media") or {}
                job_media_type = job.parameters.get("media_type", "image")
                if (job_media_type == "video" and gen_m.get("video", {}).get("video_url")) or \
                   (job_media_type == "image" and gen_m.get("image", {}).get("image_url")) or \
                   (content_check.media_asset_id and job_media_type == "image"):
                    job.status = JobStatus.COMPLETED
                    job.error_message = None
                    db.commit()
                    continue

                scheduled_utc_str = job.parameters.get("scheduled_time_utc")
                if not scheduled_utc_str:
                    continue

                try:
                    target_utc = datetime.fromisoformat(scheduled_utc_str)
                except Exception:
                    continue

                # Check if due
                if not target_content_id and now_utc < target_utc:
                    continue

                logger.info(f"[AutoMediaScheduler] Job #{job.id} became due at {scheduled_utc_str} (Now UTC: {now_utc.isoformat()}).")

                # ---------------------------------------------------------
                # Stage 4: Atomic Job Claiming (Multi-Worker Safe)
                # ---------------------------------------------------------
                claimed = db.query(GenerationJob).filter(
                    GenerationJob.id == job.id,
                    GenerationJob.status == JobStatus.PENDING
                ).update({
                    GenerationJob.status: JobStatus.PROCESSING,
                    GenerationJob.updated_at: now_utc
                }, synchronize_session=False)
                db.commit()

                if claimed != 1:
                    logger.info(f"[AutoMediaScheduler] Job #{job.id} already claimed by another worker. Skipping.")
                    continue

                logger.info(f"[AutoMediaScheduler] Job #{job.id} claimed atomically by worker.")

                content = db.query(Content).filter(Content.id == content_id).first()
                if not content:
                    job.status = JobStatus.FAILED
                    job.error_message = f"Cancelled: Parent content record #{content_id} was deleted."
                    if isinstance(job.parameters, dict):
                        job.parameters["cancelled"] = True
                        job.parameters["retries"] = 999
                    db.commit()
                    continue

                meta = dict(content.meta_info or {})
                meta["auto_media_in_progress"] = True
                meta["generation_status"] = "processing"
                meta["processing_started_at"] = now_utc.isoformat()
                content.meta_info = meta
                db.commit()

                organization_id = job.organization_id or content.organization_id
                brand = db.query(BrandProfile).filter(BrandProfile.id == content.brand_profile_id).first() if content.brand_profile_id else None
                if not brand and organization_id:
                    brand = db.query(BrandProfile).filter(BrandProfile.organization_id == organization_id).first()

                user = db.query(User).filter(User.id == job.user_id).first()
                if not user:
                    user = db.query(User).filter(User.organization_id == organization_id).first()

                media_type = job.parameters.get("media_type", "image")
                provider = job.parameters.get("provider", "cloudflare-workers-ai")

                # Validate and build brand-relevant prompt
                prompt_to_use = cls._build_brand_relevant_prompt(brand, content, media_type=media_type)
                if not prompt_to_use:
                    logger.warning(f"[AutoMediaScheduler] Insufficient brand details to generate brand-relevant media for content #{content.id}. Skipping automatic generation.")
                    job.status = JobStatus.FAILED
                    job.error_message = "Skipped: Insufficient brand profile details or non-specific topic for brand-relevant media generation."
                    if isinstance(job.parameters, dict):
                        job.parameters["cancelled"] = True
                        job.parameters["retries"] = 999
                    meta["auto_media_in_progress"] = False
                    meta["generation_status"] = "skipped"
                    meta["generation_error"] = job.error_message
                    content.meta_info = meta
                    db.commit()
                    continue

                job.prompt = prompt_to_use
                db.commit()

                logger.info(f"[AutoMediaScheduler] {media_type.capitalize()} generation started for content #{content.id} (Job #{job.id}).")

                try:
                    asset_ids_to_attach = []

                    if media_type == "image":
                        from app.services.image_generation_service import ImageGenerationService
                        from app.schemas.generation import GeminiImageGenerationRequest
                        req = GeminiImageGenerationRequest(
                            prompt=prompt_to_use,
                            brand_profile_id=brand.id if brand else None,
                            aspect_ratio="1:1",
                            style_preset="modern_product"
                        )
                        gen_res = await ImageGenerationService.generate_image(req, db, user)
                        logger.info(f"[AutoMediaScheduler] Provider response received for job #{job.id} from {provider}.")
                        if gen_res and gen_res.result_data and gen_res.result_data.get("media_asset_id"):
                            asset_ids_to_attach.append(gen_res.result_data["media_asset_id"])
                    else:
                        from app.services.video_generation_service import VideoGenerationService
                        rendered = await asyncio.to_thread(
                            VideoGenerationService.render_branded_promotional_video,
                            prompt=prompt_to_use,
                            brand=brand,
                            aspect_ratio="9:16",
                            duration_seconds=20,
                            description=content.caption or prompt_to_use
                        )
                        logger.info(f"[AutoMediaScheduler] Provider response received for job #{job.id} from {provider}.")
                        if rendered and rendered.get("video_url"):
                            v_url = rendered["video_url"]
                            m_id = rendered.get("media_asset_id")
                            if not m_id:
                                from app.services.quality_service import QualityService
                                v_asset = MediaAsset(
                                    file_name=rendered.get("file_name", f"ai_reel_{content.id}.mp4"),
                                    file_url=v_url,
                                    file_size_bytes=rendered.get("file_size_bytes", 0),
                                    mime_type="video/mp4",
                                    media_type=MediaType.VIDEO,
                                    organization_id=content.organization_id,
                                    user_id=user.id if user else 1,
                                    meta_info={
                                        "prompt": prompt_to_use,
                                        "aspect_ratio": "9:16",
                                        "duration_seconds": rendered.get("duration_seconds", 20),
                                        "provider": "multi-scene-cinema-director",
                                        "content_id": content.id
                                    }
                                )
                                db.add(v_asset)
                                db.flush()
                                m_id = v_asset.id
                                QualityService.validate_video(
                                    video_url=v_url,
                                    brand=brand,
                                    organization_id=content.organization_id,
                                    db=db,
                                    candidate_asset_id=m_id
                                )
                            if m_id:
                                asset_ids_to_attach.append(m_id)

                    if not asset_ids_to_attach:
                        raise RuntimeError("Generation provider returned without a valid media asset.")

                    # CRITICAL ATTACH GUARD: Re-verify parent content existence immediately before attaching
                    fresh_content = db.query(Content).filter(Content.id == content_id).first()
                    if not fresh_content:
                        logger.warning(f"[AutoMediaScheduler] Content #{content_id} was deleted during generation! Discarding generated assets and aborting attach.")
                        for a_id in asset_ids_to_attach:
                            try:
                                a_rec = db.query(MediaAsset).filter(MediaAsset.id == a_id).first()
                                if a_rec:
                                    f_url = a_rec.file_url or ""
                                    if f_url.startswith("/uploads/") or "/media/" in f_url:
                                        clean_rel = f_url.lstrip("/").replace("uploads/", "media/")
                                        local_p = os.path.join(settings.LOCAL_STORAGE_DIR, clean_rel)
                                        if os.path.exists(local_p):
                                            try:
                                                os.remove(local_p)
                                            except Exception:
                                                pass
                                    db.delete(a_rec)
                            except Exception as clean_err:
                                logger.warning(f"[AutoMediaScheduler] Error cleaning up orphaned asset #{a_id}: {clean_err}")
                        job.status = JobStatus.FAILED
                        job.error_message = f"Cancelled: Parent content #{content_id} was deleted while generation was in progress."
                        if isinstance(job.parameters, dict):
                            job.parameters["cancelled"] = True
                            job.parameters["retries"] = 999
                        db.commit()
                        continue

                    # Save and attach media
                    cls._attach_generated_media(db, fresh_content, asset_ids_to_attach, organization_id)
                    attached_asset = db.query(MediaAsset).filter(MediaAsset.id == asset_ids_to_attach[0]).first()
                    logger.info(f"[AutoMediaScheduler] Media saved to MediaAsset #{attached_asset.id} (URL: {attached_asset.file_url}).")

                    meta = dict(fresh_content.meta_info or {})
                    meta["auto_media_in_progress"] = False
                    meta["generation_status"] = "completed"
                    meta["plan_status"] = "draft_created"
                    meta["generation_error"] = None
                    fresh_content.meta_info = meta
                    if fresh_content.status == ContentStatus.FAILED:
                        fresh_content.status = ContentStatus.DRAFT

                    # Keep ScheduledPost aligned
                    sp = db.query(ScheduledPost).filter(ScheduledPost.content_id == fresh_content.id).first()
                    if sp:
                        sp.media_asset_id = asset_ids_to_attach[0]
                        pub_m = dict(sp.publish_metadata or {})
                        pub_m["generated_media"] = meta.get("generated_media", {})
                        sp.publish_metadata = pub_m
                        if pub_m.get("auto_approve"):
                            sp.status = "scheduled"
                            fresh_content.status = ContentStatus.SCHEDULED
                            meta["plan_status"] = "scheduled"
                        else:
                            sp.status = "draft"
                            fresh_content.status = ContentStatus.DRAFT

                    job.status = JobStatus.COMPLETED
                    job.result_data = {
                        "media_asset_id": attached_asset.id,
                        "media_url": attached_asset.file_url,
                        "completed_at": datetime.utcnow().isoformat()
                    }
                    job.error_message = None
                    db.commit()

                    processed_count += 1
                    logger.info(f"[AutoMediaScheduler] Job #{job.id} completed successfully for content #{fresh_content.id}.")

                except Exception as gen_err:
                    err_str = str(gen_err)
                    # Check if parent content was deleted during failure
                    c_exists = db.query(Content.id).filter(Content.id == content_id).first()
                    if not c_exists:
                        job.status = JobStatus.FAILED
                        job.error_message = f"Cancelled: Parent content #{content_id} was deleted."
                        if isinstance(job.parameters, dict):
                            job.parameters["cancelled"] = True
                            job.parameters["retries"] = 999
                        db.commit()
                        logger.info(f"[AutoMediaScheduler] Aborted retry for Job #{job.id}: Parent content #{content_id} was deleted.")
                        continue

                    retries = int(job.parameters.get("retries", 0)) + 1
                    job.parameters["retries"] = retries
                    job.error_message = err_str

                    c_obj = db.query(Content).filter(Content.id == content_id).first()
                    if c_obj:
                        meta = dict(c_obj.meta_info or {})
                        meta["auto_media_in_progress"] = False
                        meta["generation_error"] = err_str

                        if retries < 3:
                            retry_utc = (datetime.utcnow() + timedelta(seconds=30)).isoformat()
                            job.parameters["scheduled_time_utc"] = retry_utc
                            job.status = JobStatus.PENDING
                            meta["generation_status"] = "retrying"
                            logger.warning(f"[AutoMediaScheduler] Job #{job.id} failed (attempt {retries}/3): {err_str}. Scheduled retry for {retry_utc}.")
                        else:
                            job.status = JobStatus.FAILED
                            meta["generation_status"] = "failed"
                            logger.error(f"[AutoMediaScheduler] Job #{job.id} failed permanently for content #{content_id} after 3 attempts: {err_str}.")

                        c_obj.meta_info = meta
                    db.commit()

        except Exception as outer_err:
            logger.error(f"[AutoMediaScheduler] Error in auto_generate_due_planned_media: {outer_err}")
        finally:
            cls._is_auto_generating = False
            db.close()
        return processed_count
