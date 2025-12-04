"""
Startup tasks for the application.

Contains one-time initialization logic that runs when the app boots.
"""

import logging
from pathlib import Path

from sqlalchemy import select

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.job import Job
from app.workers.tasks.job_tasks import ingest_jobs_from_csv_async

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


async def _jobs_exist() -> bool:
    """Check if any jobs exist in the database."""
    async with AsyncSessionLocal() as session:
        query = select(Job.job_id).limit(1)
        result = await session.execute(query)
        return result.scalar_one_or_none() is not None


async def trigger_job_ingestion() -> None:
    """
    Trigger job ingestion from CSV if no jobs exist.

    Dispatches a Celery task to ingest jobs in the background.
    This is a non-blocking operation that returns immediately.
    """
    csv_path = Path(settings.JOB_INGEST_CSV_PATH)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    if not csv_path.exists():
        logger.warning("Job ingestion skipped; CSV not found at %s", csv_path)
        return

    try:
        if await _jobs_exist():
            logger.info("Job ingestion skipped; jobs already exist in database")
            return
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.error("Job ingestion skipped; DB check failed: %s", exc)
        return

    try:
        task = ingest_jobs_from_csv_async.delay(
            str(csv_path),
            settings.JOB_INGEST_BATCH_SIZE,
            settings.JOB_INGEST_LIMIT,
        )
        logger.info(
            "Scheduled job ingestion task %s from %s (batch=%s limit=%s)",
            task.id,
            csv_path,
            settings.JOB_INGEST_BATCH_SIZE,
            settings.JOB_INGEST_LIMIT,
        )
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.error("Failed to schedule job ingestion: %s", exc)
