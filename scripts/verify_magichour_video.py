"""Generate one short Magic Hour clip and validate Cloudinary delivery."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import SessionLocal
from app.models.generation_job import GenerationJob
from app.services.magic_hour_service import MagicHourVideoService
from app.utils.enums import MediaType


async def main():
    db = SessionLocal()
    try:
        image_job = db.query(GenerationJob).filter(GenerationJob.result_data.isnot(None)).order_by(GenerationJob.created_at.desc()).all()
        image_url = next((j.result_data.get("image_url") for j in image_job if isinstance(j.result_data, dict) and j.result_data.get("image_url")), None)
    finally:
        db.close()
    if not image_url:
        raise RuntimeError("A generated image is required for the Magic Hour verification.")
    result = await MagicHourVideoService.generate_and_persist_video(
        image_url, "Professional business operations scene with gentle natural camera movement, no text or logos.",
        duration_seconds=10,
    )
    url = result.get("video_url")
    if not url.startswith("https://"):
        raise RuntimeError("Magic Hour result was not stored in Cloudinary.")
    print({"generated": True, "provider": result["provider"], "duration_seconds": result["duration_seconds"]})


if __name__ == "__main__":
    asyncio.run(main())
