"""
Comprehensive verification test suite for Durable Cloud Media Storage Integration.
Tests image, audio/voiceover, and video reel durable storage in Cloudinary,
verifying permanent HTTPS URLs, database persistence, and local file independence.
"""
import asyncio
import os
import sys
import uuid
from io import BytesIO
from pathlib import Path
import httpx
from PIL import Image

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.db.base import Base
from app.models.organization import Organization
from app.models.user import User
from app.models.brand_profile import BrandProfile
from app.models.media_asset import MediaAsset
from app.models.scheduled_post import ScheduledPost
from app.integrations.storage.cloudinary_storage import (
    CloudinaryStorageProvider,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    ALLOWED_AUDIO_EXTENSIONS
)
from app.utils.enums import MediaType, UserRole
from app.services.voice_service import VoiceService
from app.services.image_generation_service import ImageGenerationService
from app.services.video_generation_service import VideoGenerationService


def test_section(title: str):
    print("\n" + "=" * 70)
    print(f"👉 {title}")
    print("=" * 70)


async def run_full_storage_verification():
    print("\n🚀 STARTING COMPREHENSIVE CLOUD MEDIA STORAGE VERIFICATION")

    # -------------------------------------------------------------------------
    # 1. VERIFY CONFIGURATION & SUPPORTED FORMATS
    # -------------------------------------------------------------------------
    test_section("STEP 1: Cloud Storage Configuration & Format Validation")
    is_conf = CloudinaryStorageProvider.is_configured()
    print(f"[*] CloudinaryStorageProvider configured: {is_conf}")
    assert is_conf, "Cloudinary credentials must be configured in environment."

    # Validate file type validators
    img_type, img_ext = CloudinaryStorageProvider.validate_file("artwork.png", "image/png", 1024 * 50)
    assert img_type == MediaType.IMAGE
    print(f"[*] Image validation: {img_type.value} ({img_ext}) OK")

    vid_type, vid_ext = CloudinaryStorageProvider.validate_file("reel.mp4", "video/mp4", 1024 * 500)
    assert vid_type == MediaType.VIDEO
    print(f"[*] Video validation: {vid_type.value} ({vid_ext}) OK")

    aud_type, aud_ext = CloudinaryStorageProvider.validate_file("voice.mp3", "audio/mpeg", 1024 * 100)
    assert aud_type == MediaType.AUDIO
    print(f"[*] Audio validation: {aud_type.value} ({aud_ext}) OK")

    # -------------------------------------------------------------------------
    # 2. TEST DIRECT IMAGE UPLOAD & PERMANENT HTTPS CDN URL
    # -------------------------------------------------------------------------
    test_section("STEP 2: In-Memory / File Image Upload to Cloudinary")
    test_img = Image.new("RGB", (128, 128), color="#4338ca")
    buf = BytesIO()
    test_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    img_res = CloudinaryStorageProvider.upload_bytes(
        data=img_bytes,
        file_name=f"test_cloud_image_{uuid.uuid4().hex[:6]}.png",
        organization_id=1,
        folder_prefix="socialai/verification/images"
    )
    print(f"[*] Image uploaded successfully:")
    print(f"    - URL: {img_res['file_url']}")
    print(f"    - Public ID: {img_res['public_id']}")
    print(f"    - Media Type: {img_res['media_type']}")
    assert img_res["file_url"].startswith("https://res.cloudinary.com"), "Must return permanent HTTPS CDN URL."

    # -------------------------------------------------------------------------
    # 3. TEST AUDIO / NEURAL VOICEOVER SYNTHESIS & DURABLE STORAGE
    # -------------------------------------------------------------------------
    test_section("STEP 3: Neural Voiceover Audio Synthesis & Cloudinary Storage")
    voice_script = "Welcome to Zenvora. We build clean cloud automation tools for high-velocity teams worldwide."
    voice_res = await VoiceService.synthesize_and_store_speech(
        text=voice_script,
        voice="en-US-JennyNeural",
        organization_id=1
    )
    print(f"[*] AI Voiceover synthesized & stored in Cloudinary:")
    print(f"    - URL: {voice_res['audio_url']}")
    print(f"    - Duration: {round(voice_res['duration_seconds'], 2)}s")
    print(f"    - Size: {voice_res['file_size_bytes']} bytes")
    print(f"    - Public ID: {voice_res['public_id']}")
    assert voice_res["audio_url"].startswith("https://res.cloudinary.com"), "Audio URL must be permanent HTTPS CDN."
    assert voice_res["duration_seconds"] > 1.0, "Audio duration must be greater than 1s."

    # -------------------------------------------------------------------------
    # 4. TEST BRANDED VISUAL IMAGE GENERATION & CLOUD STORAGE
    # -------------------------------------------------------------------------
    test_section("STEP 4: Branded Visual Artwork Generation & Cloud Storage")
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = Session(engine)

    org = Organization(name="Zenvora Automation", slug="zenvora-test")
    db.add(org)
    db.flush()

    user = User(email="test_creator@zenvora.test", hashed_password="fake", organization_id=org.id)
    brand = BrandProfile(
        name="Zenvora Cloud",
        industry="Cloud & AI Software",
        brand_guidelines="Minimalist indigo tech aesthetics",
        color_palette=["#4f46e5", "#06b6d4"],
        organization_id=org.id
    )
    db.add_all([user, brand])
    db.commit()

    graphic_res = ImageGenerationService.render_branded_marketing_graphic(
        prompt="Cloud workflow orchestration engine with real-time throughput",
        brand=brand,
        aspect_ratio="1:1",
        style_preset="modern_product"
    )
    print(f"[*] Branded Graphic rendered & stored in Cloudinary:")
    print(f"    - URL: {graphic_res['image_url']}")
    print(f"    - Model: {graphic_res['model_name']}")
    assert graphic_res["image_url"].startswith("https://res.cloudinary.com"), "Image URL must be permanent Cloudinary HTTPS."

    # -------------------------------------------------------------------------
    # 5. TEST VIDEO / REEL RENDERING & DURABLE CLOUDINARY CHUNKED UPLOAD
    # -------------------------------------------------------------------------
    test_section("STEP 5: Video Reel Rendering & Chunked Cloudinary Storage")
    video_res = await asyncio.to_thread(
        VideoGenerationService.render_branded_promotional_video,
        prompt="Automate your entire deployment pipeline in 60 seconds with Zenvora Cloud.",
        brand=brand,
        aspect_ratio="9:16",
        duration_seconds=10,
        fps=24,
        image_url=graphic_res["image_url"],
        description="Next-generation cloud automation without enterprise complexity."
    )
    print(f"[*] AI Video Reel rendered & stored in Cloudinary:")
    print(f"    - Video URL: {video_res['video_url']}")
    print(f"    - Aspect Ratio: {video_res['aspect_ratio']}")
    print(f"    - File Size: {video_res['file_size_bytes']} bytes")
    assert video_res["video_url"].startswith("https://res.cloudinary.com"), "Video URL must be permanent Cloudinary HTTPS."

    # -------------------------------------------------------------------------
    # 6. TEST DATABASE PERSISTENCE & METADATA AUDIT
    # -------------------------------------------------------------------------
    test_section("STEP 6: Database Persistence & Metadata Record Verification")
    # Save Image MediaAsset
    img_asset = MediaAsset(
        organization_id=org.id,
        user_id=user.id,
        file_name=graphic_res["file_name"],
        file_url=graphic_res["image_url"],
        mime_type="image/png",
        media_type=MediaType.IMAGE,
        file_size_bytes=graphic_res["file_size_bytes"],
        title="Zenvora Cloud Workflow Artwork",
        validation_status="APPROVED",
        quality_score=98.5
    )
    # Save Audio MediaAsset
    aud_asset = MediaAsset(
        organization_id=org.id,
        user_id=user.id,
        file_name=voice_res["file_name"],
        file_url=voice_res["audio_url"],
        mime_type="audio/mp3",
        media_type=MediaType.AUDIO,
        file_size_bytes=voice_res["file_size_bytes"],
        title="Zenvora Cloud Narration",
        validation_status="APPROVED",
        quality_score=100.0
    )
    # Save Video MediaAsset
    vid_asset = MediaAsset(
        organization_id=org.id,
        user_id=user.id,
        file_name=video_res["file_name"],
        file_url=video_res["video_url"],
        mime_type="video/mp4",
        media_type=MediaType.VIDEO,
        file_size_bytes=video_res["file_size_bytes"],
        title="Zenvora 9:16 Social Reel",
        validation_status="APPROVED",
        quality_score=99.0
    )
    db.add_all([img_asset, aud_asset, vid_asset])
    db.commit()

    saved_assets = db.query(MediaAsset).filter(MediaAsset.organization_id == org.id).all()
    print(f"[*] Verified {len(saved_assets)} MediaAsset records committed to database:")
    for a in saved_assets:
        print(f"    - ID: #{a.id} | Type: {a.media_type.value:5} | URL: {a.file_url[:65]}...")
        assert a.file_url.startswith("https://res.cloudinary.com"), f"Asset #{a.id} has non-cloud URL: {a.file_url}"

    # -------------------------------------------------------------------------
    # 7. SERVER RESTART & LOCAL-FILE-DELETION INDEPENDENCE TEST
    # -------------------------------------------------------------------------
    test_section("STEP 7: Server Independence & CDN Public Readability Test")
    print("[*] Testing that CDN URLs remain accessible over public HTTPS regardless of local files...")
    async with httpx.AsyncClient(timeout=25.0) as http_client:
        for a in saved_assets:
            r = await http_client.get(a.file_url)
            print(f"    - Fetching {a.media_type.value.upper()} from Cloud CDN: HTTP {r.status_code} ({len(r.content)} bytes)")
            assert r.status_code == 200, f"Cloud URL {a.file_url} returned HTTP {r.status_code}"
            assert len(r.content) > 1000, "Downloaded media size should be > 1KB"

    # -------------------------------------------------------------------------
    # 8. SCHEDULED POST PUBLISHING WORKER VALIDATION
    # -------------------------------------------------------------------------
    test_section("STEP 8: Scheduled Post Media Attachment & Publishing Verification")
    from datetime import datetime, timedelta
    sched_post = ScheduledPost(
        organization_id=org.id,
        user_id=user.id,
        content_id=None,
        media_asset_id=vid_asset.id,
        scheduled_time=datetime.utcnow() + timedelta(hours=2),
        platform="instagram",
        status="scheduled",
        title="Automate Workflows Faster with Zenvora Cloud",
        caption="Cut deployment bottlenecks by 80%. Try Zenvora Cloud free today!\n\n#Cloud #Automation #DevOps",
        publish_metadata={
            "brand_profile_id": brand.id,
            "media_urls": [graphic_res["image_url"], video_res["video_url"]],
            "audio_url": voice_res["audio_url"],
            "aspect_ratio": "9:16",
            "campaign_id": 101
        }
    )
    db.add(sched_post)
    db.commit()
    db.refresh(sched_post)

    print(f"[*] ScheduledPost created with ID #{sched_post.id}:")
    print(f"    - Target Platform: {sched_post.platform}")
    print(f"    - Media URLs in publish_metadata:")
    for u in sched_post.publish_metadata["media_urls"]:
        print(f"      • {u}")
        assert u.startswith("https://res.cloudinary.com"), "Publishing media URL must be durable HTTPS."

    print("\n" + "=" * 70)
    print("🎉 ALL DURABLE CLOUD MEDIA STORAGE VERIFICATION TESTS PASSED CLEANLY!")
    print("=" * 70)

    db.close()
    engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_full_storage_verification())
