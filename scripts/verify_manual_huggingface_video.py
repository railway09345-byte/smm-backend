"""Verify the manual Studio Hugging Face service creates a real Cloudinary clip."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import SessionLocal
from app.integrations.storage.cloudinary_storage import CloudinaryStorageProvider
from app.models.brand_profile import BrandProfile
from app.services.huggingface_video_service import HuggingFaceVideoService
from app.utils.enums import MediaType


async def main():
    db = SessionLocal()
    try:
        brand = db.query(BrandProfile).filter(BrandProfile.name == "Sutherland").first()
        if not brand:
            raise RuntimeError("Sutherland profile is required for this check")
        result = await HuggingFaceVideoService.generate_and_persist_video(
            "A short professional BPO operations visual, no text or logos.",
            brand=brand,
            aspect_ratio="9:16",
            duration_seconds=3,
            organization_id=brand.organization_id,
        )
    finally:
        db.close()
    public_id = result.get("cloudinary_public_id")
    if not result.get("video_url") or not public_id:
        raise RuntimeError("Manual Studio service did not return a Cloudinary video")
    removed = await asyncio.to_thread(CloudinaryStorageProvider.delete_file, public_id, MediaType.VIDEO)
    if not removed:
        raise RuntimeError("Temporary test video could not be deleted")
    print({"generated": True, "stored": True, "cleaned_up": True, "provider": result["provider"], "duration_seconds": result["duration_seconds"]})


if __name__ == "__main__":
    asyncio.run(main())
