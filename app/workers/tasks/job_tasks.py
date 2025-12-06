"""
Celery tasks for job processing.

Handles:
- Single job ingestion
- Bulk job ingestion
- CSV import
"""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import select
from app.core.celery_app import celery_app
from app.database import AsyncSessionLocal
from app.models.job import Job
from app.schemas.job import JobDescriptionSchema
from app.services.job.job_embedder import generate_job_embedding_async
from app.services.job.job_parser import parse_job_description

logger = logging.getLogger(__name__)


async def _process_single_job(
    job_id: str, job_title: str, job_description: str, job_data: Dict
) -> Dict:
    """
    Internal async function to process a single job.

    Args:
        job_id: Job UUID
        job_title: Job title
        job_description: Job description text
        job_data: Additional job data

    Returns:
        Dictionary with job_id and status
    """
    # Parse job description with LLM
    parsed_job = await parse_job_description(job_title, job_description)

    # Generate embedding
    embedding = await generate_job_embedding_async(parsed_job)

    # Save to database
    async with AsyncSessionLocal() as db:
        # Check for duplicates
        query = select(Job).where(
            Job.title == job_title,
            Job.description == job_description,
        )
        result = await db.execute(query)
        existing = result.scalar_one_or_none()

        if existing:
            logger.warning(f"Job already exists: {job_title}")
            return {"job_id": str(existing.job_id), "status": "duplicate"}

        # Create new job
        job = Job(
            job_id=uuid.UUID(job_id) if isinstance(job_id, str) else job_id,
            title=parsed_job.title or job_title,
            company=parsed_job.company or job_data.get("company"),
            employment_type=parsed_job.employment_type,
            date=parsed_job.date,
            description=parsed_job.description or job_description,
            location=(
                parsed_job.location.model_dump() if parsed_job.location else None
            ),
            remote=parsed_job.remote.value if parsed_job.remote else None,
            salary=parsed_job.salary,
            experience=parsed_job.experience,
            min_experience_years=parsed_job.min_experience_years,
            max_experience_years=parsed_job.max_experience_years,
            experience_level=parsed_job.experience_level,
            responsibilities=parsed_job.responsibilities,
            qualifications=parsed_job.qualifications,
            skills=(
                [skill.model_dump() for skill in parsed_job.skills]
                if parsed_job.skills
                else None
            ),
            embedding=embedding,
        )

        db.add(job)
        await db.commit()

    return {"job_id": job_id, "status": "success"}


async def _process_job_batch(jobs_data: List[Dict]) -> Dict:
    """
    Internal async function to process multiple jobs concurrently.

    Args:
        jobs_data: List of job dictionaries

    Returns:
        Dictionary with success/failure counts
    """
    tasks = []
    for job_data in jobs_data:
        job_id = job_data.get("job_id") or str(uuid.uuid4())
        job_title = job_data.get("title", "")
        job_description = job_data.get("description", "")

        task = _process_single_job(job_id, job_title, job_description, job_data)
        tasks.append(task)

    # Process all jobs concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successes and failures
    success_count = sum(
        1 for r in results if isinstance(r, dict) and r.get("status") == "success"
    )
    duplicate_count = sum(
        1 for r in results if isinstance(r, dict) and r.get("status") == "duplicate"
    )
    failed_count = sum(1 for r in results if isinstance(r, Exception))

    return {
        "total": len(jobs_data),
        "success": success_count,
        "duplicate": duplicate_count,
        "failed": failed_count,
    }


@celery_app.task(
    bind=True,
    max_retries=1,
    name="app.workers.tasks.job_tasks.ingest_jobs_from_csv",
)
def ingest_jobs_from_csv_async(
    self, csv_path: str, batch_size: int = 10, limit: Optional[int] = None
) -> Dict:
    """
    Ingest jobs from CSV file.

    Args:
        csv_path: Path to CSV file
        batch_size: Number of jobs to process concurrently
        limit: Maximum number of jobs to process (None for all)

    Returns:
        Dictionary with ingestion statistics
    """
    try:
        logger.info(f"Starting CSV import from: {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Found {len(df)} jobs in CSV")

        if limit:
            df = df.head(limit)
            logger.info(f"Processing limit of {limit} jobs")

        # Convert to job data dictionaries
        jobs_data = []
        for idx, row in df.iterrows():
            job_data = {
                "job_id": str(uuid.uuid4()),
                "title": str(row.get("job_title", "")),
                "description": str(row.get("job_description", "")),
            }
            jobs_data.append(job_data)

        # Process in batches
        result = asyncio.run(_process_csv_batches(jobs_data, batch_size))

        logger.info(f"CSV import complete: {result}")
        return result

    except Exception as e:
        logger.error(f"Error importing CSV: {e}")
        raise self.retry(exc=e, countdown=300)


async def _process_csv_batches(jobs_data: List[Dict], batch_size: int) -> Dict:
    """
    Process CSV jobs in batches.

    Args:
        jobs_data: List of job dictionaries
        batch_size: Batch size for concurrent processing

    Returns:
        Dictionary with statistics
    """
    total_success = 0
    total_duplicate = 0
    total_failed = 0

    # Split into batches
    batches = [
        jobs_data[i : i + batch_size] for i in range(0, len(jobs_data), batch_size)
    ]

    for batch_num, batch in enumerate(batches, 1):
        logger.info(f"Processing batch {batch_num}/{len(batches)}")
        result = await _process_job_batch(batch)

        total_success += result["success"]
        total_duplicate += result["duplicate"]
        total_failed += result["failed"]

    return {
        "total": len(jobs_data),
        "success": total_success,
        "duplicate": total_duplicate,
        "failed": total_failed,
    }
