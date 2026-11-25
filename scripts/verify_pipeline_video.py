import os
import sys
import json
import asyncio
import subprocess
import tempfile
from pathlib import Path

# Add backend directory to path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, backend_dir)

from app.db.session import SessionLocal
from app.models.brand_profile import BrandProfile
from app.services.daily_ai_provider import DailyAIProvider
from app.services.daily_campaign_service import verify_video, store_asset
from app.core.ffmpeg_utils import get_ffmpeg_executable
from app.integrations.storage.cloudinary_storage import CloudinaryStorageProvider


async def main():
    print("=================================================================")
    print("STARTING AUTONOMOUS PIPELINE VIDEO GENERATION VERIFICATION")
    print("=================================================================")

    db = SessionLocal()
    try:
        # Load verified brand profile
        brand = db.query(BrandProfile).filter(BrandProfile.id == 5).first()
        if not brand:
            brand = db.query(BrandProfile).first()

        snapshot = {
            "brand_profile_id": brand.id,
            "organization_id": brand.organization_id,
            "name": brand.name,
            "industry": brand.industry,
            "tagline": brand.tagline,
            "tone_of_voice": brand.tone_of_voice,
            "cta": brand.cta,
            "color_palette": brand.color_palette,
            "key_themes": brand.key_themes,
            "meta_info": brand.meta_info,
            "brand_guidelines": brand.brand_guidelines,
            "voice": getattr(brand, "voice", None)
        }
    finally:
        db.close()

    print(f"Target Brand: {snapshot['name']} ({snapshot['industry']})")
    print(f"Tone: {snapshot['tone_of_voice']}")
    print(f"CTA: {snapshot['cta']}")

    prompt = (
        "Cinematic 30s commercial showcase for handcrafted artisanal ceramics. "
        "Showcase macro textures, pottery wheel craftsmanship, elegant morning lifestyle, "
        "and modern ceramic dinnerware in warm ambient sunlight."
    )

    print("\n--- STEP 1: Calling DailyAIProvider.start_video ---")
    op_id, telemetry = await DailyAIProvider.start_video(
        prompt=prompt,
        snapshot=snapshot
    )
    print(f"Operation ID: {op_id}")
    print(f"Telemetry: {json.dumps(telemetry, indent=2)}")

    print("\n--- STEP 2: Calling DailyAIProvider.poll_video ---")
    video_bytes = await DailyAIProvider.poll_video(op_id)
    if not video_bytes:
        raise RuntimeError("poll_video returned no video bytes!")
    print(f"Retrieved video payload: {len(video_bytes):,} bytes ({len(video_bytes)/1024/1024:.2f} MB)")

    print("\n--- STEP 3: Calling verify_video(video_bytes) ---")
    tech_info = verify_video(video_bytes)
    print(f"Technical validation results: {json.dumps(tech_info, indent=2)}")

    # Assert duration is approximately 30 seconds
    duration = tech_info["duration_seconds"]
    assert 25.0 <= duration <= 35.0, f"Expected ~30s duration, got {duration}s"
    assert tech_info.get("has_audio") is True, "Expected active audio stream in video"
    print(f"SUCCESS: Video duration is {duration:.2f}s (target ~30s) with active audio stream!")

    print("\n--- STEP 4: Storing asset on Cloudinary CDN ---")
    ref_url, storage_info = store_asset(
        data=video_bytes,
        organization_id=snapshot["organization_id"],
        brand_id=snapshot["brand_profile_id"],
        run_id=999,
        asset_id=999,
        kind="video"
    )
    print(f"Cloudinary Delivery URL: {ref_url}")
    print(f"Storage Info: {json.dumps(storage_info, indent=2)}")

    print("\n--- STEP 5: FFprobe Technical Inspection on Cloudinary CDN Asset ---")
    ffmpeg_exe = get_ffmpeg_executable()
    probe_cmd = [ffmpeg_exe, "-i", ref_url]
    proc = subprocess.run(probe_cmd, stderr=subprocess.PIPE, text=True, timeout=20)
    probe_output = proc.stderr
    print("FFPROBE / FFMPEG OUTPUT:")
    print("-----------------------------------------------------------------")
    print(probe_output)
    print("-----------------------------------------------------------------")

    # Verify both streams
    has_video_stream = "Stream #" in probe_output and "Video:" in probe_output
    has_audio_stream = "Stream #" in probe_output and "Audio:" in probe_output
    print(f"Stream Inspection -> Video Stream Present: {has_video_stream}, Audio Stream Present: {has_audio_stream}")
    assert has_video_stream, "Probe failed to find Video stream!"
    assert has_audio_stream, "Probe failed to find Audio stream!"

    print("\n=================================================================")
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=================================================================")


if __name__ == "__main__":
    asyncio.run(main())

