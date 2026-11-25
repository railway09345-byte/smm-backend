import asyncio
import os
from typing import Optional
from app.workers.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.generation_job import GenerationJob
from app.integrations.ai.text_provider import AITextProvider
from app.integrations.ai.image_provider import AIImageProvider
from app.utils.enums import JobStatus, GenerationType
from app.core.logging import logger


@celery_app.task(
    name="app.workers.generation_tasks.execute_generation_job",
    time_limit=900,        # 15 minutes hard timeout
    soft_time_limit=840     # 14 minutes soft timeout
)
def execute_generation_job(job_id: int):
    """
    Celery task to run long-running AI generations (Text, Image, Video).
    Uses clean per-task asyncio event loop to avoid Python 3.10+ worker thread crashes.
    """
    db = SessionLocal()
    job: Optional[GenerationJob] = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        job = db.query(GenerationJob).filter(GenerationJob.id == job_id).first()
        if not job:
            logger.warning(f"[CeleryGenTask] Job #{job_id} not found in database.")
            return

        job.status = JobStatus.PROCESSING
        db.commit()

        if job.job_type in [GenerationType.TEXT_CAPTION, GenerationType.TEXT_HASHTAGS, GenerationType.TEXT_ARTICLE]:
            result = loop.run_until_complete(
                AITextProvider.generate_caption(
                    prompt=job.prompt,
                    tone=(job.parameters or {}).get("tone", "engaging"),
                    target_platforms=(job.parameters or {}).get("platforms", []),
                    provider=(job.parameters or {}).get("provider", "openai")
                )
            )
            job.result_data = result
            job.tokens_used = result.get("tokens_used", 100) if isinstance(result, dict) else 100
            job.status = JobStatus.COMPLETED

        elif job.job_type == GenerationType.IMAGE_GENERATION:
            result = loop.run_until_complete(
                AIImageProvider.generate_image(
                    prompt=job.prompt,
                    aspect_ratio=(job.parameters or {}).get("aspect_ratio", "1:1"),
                    style=(job.parameters or {}).get("style", "modern")
                )
            )
            job.result_data = result
            job.status = JobStatus.COMPLETED

        elif job.job_type == GenerationType.VIDEO_GENERATION:
            from app.services.video_generation_service import VideoGenerationService
            logger.info(f"[CeleryGenTask] Executing video generation for Job #{job.id}...")
            payload_data = job.parameters or {}
            brand_id = payload_data.get("brand_profile_id")
            loop.run_until_complete(
                VideoGenerationService._execute_video_generation_background(
                    job_id=job.id,
                    payload_data=payload_data,
                    user_id=job.user_id,
                    organization_id=job.organization_id,
                    brand_id=brand_id
                )
            )
            db.refresh(job)
            logger.info(f"[CeleryGenTask] Video Job #{job.id} finished with status '{job.status}'.")

        db.commit()
    except Exception as e:
        logger.error(f"[CeleryGenTask] Generation task failed for Job #{job_id}: {str(e)}", exc_info=True)
        if job:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            db.commit()
    finally:
        try:
            loop.close()
        except Exception:
            pass
        db.close()
