"""Verify a real LTX clip can be validated and delivered by Cloudinary, then remove it."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.integrations.storage.cloudinary_storage import CloudinaryStorageProvider
from app.services.daily_ai_provider import DailyAIProvider
from app.services.daily_campaign_service import store_asset, verify_video
from app.utils.enums import MediaType


async def main():
    operation, telemetry = await DailyAIProvider.start_video(
        "A three-second cinematic product reveal, no text or logos.",
        {"name": "Test Brand", "industry": "retail", "tone": "professional"},
    )
    video = await DailyAIProvider.poll_video(operation)
    technical = await asyncio.to_thread(verify_video, video)
    url, metadata = await asyncio.to_thread(store_asset, video, 999999, 999999, 999999, 999999, "video")
    public_id = metadata.get("public_id")
    if not url or not public_id:
        raise RuntimeError("Cloudinary did not return a durable delivery reference")
    removed = await asyncio.to_thread(CloudinaryStorageProvider.delete_file, public_id, MediaType.VIDEO)
    if not removed:
        raise RuntimeError("Temporary Cloudinary verification asset could not be removed")
    print({"generated": True, "delivered": True, "cleaned_up": True, "provider": telemetry["provider"], **technical})


if __name__ == "__main__":
    asyncio.run(main())
