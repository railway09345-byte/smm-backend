"""Request one small Hugging Face LTX clip through the daily-provider adapter."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.daily_ai_provider import DailyAIProvider
from app.services.daily_campaign_service import verify_video


async def main():
    operation, telemetry = await DailyAIProvider.start_video(
        "A three-second cinematic product reveal, no text or logos.",
        {"name": "Test Brand", "industry": "retail", "tone": "professional"},
    )
    video = await DailyAIProvider.poll_video(operation)
    technical = await asyncio.to_thread(verify_video, video)
    print({"generated": True, "bytes": len(video), "provider": telemetry["provider"], "model": telemetry["model"], **technical})


if __name__ == "__main__":
    asyncio.run(main())
