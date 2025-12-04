"""
API routes for job-related endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_async_db
from app.models.job import Job
from app.schemas.job import JobDescription, JobDescriptionCreate
from app.services.job.job_embedder import generate_job_embedding


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("", response_model=List[JobDescription])
async def get_jobs(
    skip: int = Query(0, ge=0, description="Number of jobs to skip"),
    limit: int = Query(
        10, ge=1, le=100, description="Maximum number of jobs to return"
    ),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Get all jobs with pagination.

    Args:
        skip: Number of jobs to skip (for pagination)
        limit: Maximum number of jobs to return
        db: Database session

    Returns:
        List of job descriptions
    """
    query = select(Job).offset(skip).limit(limit)
    result = await db.execute(query)
    jobs = result.scalars().all()

    return [
        JobDescription(
            id=job.id,
            job_id=job.job_id,
            title=job.title,
            company=job.company,
            type=job.type,
            date=job.date,
            description=job.description,
            location=job.location,
            remote=job.remote,
            salary=job.salary,
            experience=job.experience,
            responsibilities=job.responsibilities,
            qualifications=job.qualifications,
            skills=job.skills,
        )
        for job in jobs
    ]


@router.get("/{job_id}", response_model=JobDescription)
async def get_job(job_id: str, db: AsyncSession = Depends(get_async_db)):
    """
    Get a single job by ID.

    Args:
        job_id: Unique job identifier
        db: Database session

    Returns:
        Job description

    Raises:
        HTTPException: If job not found
    """
    query = select(Job).where(Job.job_id == job_id)
    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobDescription(
        id=job.id,
        job_id=job.job_id,
        title=job.title,
        company=job.company,
        type=job.type,
        date=job.date,
        description=job.description,
        location=job.location,
        remote=job.remote,
        salary=job.salary,
        experience=job.experience,
        responsibilities=job.responsibilities,
        qualifications=job.qualifications,
        skills=job.skills,
    )


@router.post("", response_model=JobDescription, status_code=201)
async def create_job(
    job_data: JobDescriptionCreate, db: AsyncSession = Depends(get_async_db)
):
    """
    Create a new job posting.

    Args:
        job_data: Job description data
        db: Database session

    Returns:
        Created job description

    Raises:
        HTTPException: If job_id already exists
    """
    # Check if job_id already exists
    query = select(Job).where(Job.job_id == job_data.job_id)
    result = await db.execute(query)
    existing_job = result.scalar_one_or_none()

    if existing_job:
        raise HTTPException(
            status_code=400, detail=f"Job with ID {job_data.job_id} already exists"
        )

    # Generate embedding
    embedding = generate_job_embedding(job_data)

    # Create job
    job = Job(
        job_id=job_data.job_id,
        title=job_data.title,
        company=job_data.company,
        type=job_data.type,
        date=job_data.date,
        description=job_data.description,
        location=job_data.location.model_dump() if job_data.location else None,
        remote=job_data.remote,
        salary=job_data.salary,
        experience=job_data.experience,
        responsibilities=job_data.responsibilities,
        qualifications=job_data.qualifications,
        skills=(
            [skill.model_dump() for skill in job_data.skills]
            if job_data.skills
            else None
        ),
        embedding=embedding,
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    return JobDescription(
        id=job.id,
        job_id=job.job_id,
        title=job.title,
        company=job.company,
        type=job.type,
        date=job.date,
        description=job.description,
        location=job.location,
        remote=job.remote,
        salary=job.salary,
        experience=job.experience,
        responsibilities=job.responsibilities,
        qualifications=job.qualifications,
        skills=job.skills,
    )


@router.get("/stats/count")
async def get_job_count(db: AsyncSession = Depends(get_async_db)):
    """
    Get total number of jobs in the database.

    Args:
        db: Database session

    Returns:
        Dictionary with job count
    """
    query = select(func.count()).select_from(Job)
    result = await db.execute(query)
    count = result.scalar_one()

    return {"total_jobs": count}
