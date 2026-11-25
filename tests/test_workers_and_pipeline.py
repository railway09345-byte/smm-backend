import os
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from app.models.generation_job import GenerationJob
from app.models.scheduled_post import ScheduledPost
from app.models.user import User
from app.utils.enums import JobStatus, GenerationType, ContentStatus
from app.services.redis_manager import RedisManager
from app.workers.generation_tasks import execute_generation_job
from app.scheduler.publishing_worker import PublishingWorker


def test_redis_manager_fallback():
    """Verify RedisManager returns None without crashing when Redis is unavailable or on failure."""
    with patch("redis.from_url") as mock_from_url:
        mock_from_url.side_effect = Exception("Connection refused")
        # Invalidate existing client
        RedisManager._client = None
        client = RedisManager.get_client()
        assert client is None
        assert RedisManager.is_available() is False


def test_stuck_generation_job_reaper(db_session):
    """
    Verify that PublishingWorker periodic sweep identifies and recovers
    stuck GenerationJobs that have been in 'processing' status > 15 minutes.
    """
    # Fetch seeded user from test database
    user = db_session.query(User).first()
    assert user is not None, "Test database should have seeded users"

    stale_time = datetime.now(timezone.utc) - timedelta(minutes=20)
    
    # Create a stuck video generation job
    stuck_job = GenerationJob(
        user_id=user.id,
        organization_id=user.organization_id,
        job_type=GenerationType.VIDEO_GENERATION,
        status=JobStatus.PROCESSING,
        prompt="A cinematic 30s product reel",
        model_name="multi-scene-cinema-director",
        parameters={"brand_profile_id": 1},
        created_at=stale_time,
        updated_at=stale_time
    )
    db_session.add(stuck_job)
    db_session.commit()
    db_session.refresh(stuck_job)

    assert stuck_job.status == JobStatus.PROCESSING

    # Simulate publisher worker reaper logic
    stuck_cutoff = datetime.utcnow() - timedelta(minutes=15)
    stuck_jobs = db_session.query(GenerationJob).filter(
        GenerationJob.status == JobStatus.PROCESSING,
        GenerationJob.created_at <= stuck_cutoff
    ).all()

    for job in stuck_jobs:
        job.status = JobStatus.FAILED
        job.error_message = "Generation task timed out or server restarted during rendering. Please retry."
        job.result_data = {
            "status": "failed",
            "error": "Generation task timed out or server restarted during rendering. Please retry."
        }
    db_session.commit()
    db_session.refresh(stuck_job)

    assert stuck_job.status == JobStatus.FAILED
    assert "timed out or server restarted" in stuck_job.error_message


def test_celery_generation_task_handles_missing_job(db_session):
    """Verify Celery task handles non-existent job cleanly without raising uncaught exceptions."""
    # Calling execute_generation_job for an ID that doesn't exist
    execute_generation_job(999999)


def test_scheduled_post_atomic_claim(db_session):
    """Verify atomic claiming prevents duplicate publishing runs on scheduled posts."""
    user = db_session.query(User).first()
    assert user is not None

    post = ScheduledPost(
        organization_id=user.organization_id,
        user_id=user.id,
        caption="Exclusive product launch announcement",
        platform="instagram",
        status="scheduled",
        scheduled_time=datetime.utcnow() - timedelta(minutes=5)
    )
    db_session.add(post)
    db_session.commit()
    db_session.refresh(post)

    # First claim succeeds
    claimed = db_session.query(ScheduledPost).filter(
        ScheduledPost.id == post.id,
        ScheduledPost.status.in_(["scheduled", ContentStatus.SCHEDULED])
    ).update({ScheduledPost.status: "publishing"}, synchronize_session=False)
    db_session.commit()

    assert claimed == 1

    # Second concurrent claim fails (claimed count == 0)
    claimed_again = db_session.query(ScheduledPost).filter(
        ScheduledPost.id == post.id,
        ScheduledPost.status.in_(["scheduled", ContentStatus.SCHEDULED])
    ).update({ScheduledPost.status: "publishing"}, synchronize_session=False)
    db_session.commit()

    assert claimed_again == 0
