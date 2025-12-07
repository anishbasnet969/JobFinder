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
from app.services.job.job_parser import parse_job_description


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
            job_id=str(job.job_id),
            title=job.title,
            company=job.company,
            employment_type=job.employment_type,
            date=job.date,
            description=job.description,
            location=job.location,
            remote=job.remote,
            salary=job.salary,
            experience=job.experience,
            min_experience_years=job.min_experience_years,
            max_experience_years=job.max_experience_years,
            experience_level=job.experience_level,
            responsibilities=job.responsibilities,
            qualifications=job.qualifications,
            education=job.education,
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
        job_id=str(job.job_id),
        title=job.title,
        company=job.company,
        employment_type=job.employment_type,
        date=job.date,
        description=job.description,
        location=job.location,
        remote=job.remote,
        salary=job.salary,
        experience=job.experience,
        min_experience_years=job.min_experience_years,
        max_experience_years=job.max_experience_years,
        experience_level=job.experience_level,
        responsibilities=job.responsibilities,
        qualifications=job.qualifications,
        education=job.education,
        skills=job.skills,
    )


@router.post("", response_model=JobDescription, status_code=201)
async def create_job(
    job_data: JobDescriptionCreate, db: AsyncSession = Depends(get_async_db)
):
    """
    Create a new job posting job title and job description.

    The incoming `job_data` is parsed to produce a
    structured `JobDescriptionSchema` which we store in the database.
    """
    # Parse the job description to structured schema
    try:
        parsed_job = await parse_job_description(job_data.title, job_data.description)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to parse job description: {e}"
        )

    # Generate embedding from the parsed structured job
    embedding = generate_job_embedding(parsed_job)

    # Create job record (DB will generate job_id)
    job = Job(
        title=parsed_job.title,
        company=parsed_job.company,
        employment_type=parsed_job.employment_type,
        date=parsed_job.date,
        description=parsed_job.description,
        location=parsed_job.location.model_dump() if parsed_job.location else None,
        remote=parsed_job.remote.value if parsed_job.remote else None,
        salary=parsed_job.salary,
        experience=parsed_job.experience,
        min_experience_years=parsed_job.min_experience_years,
        max_experience_years=parsed_job.max_experience_years,
        experience_level=parsed_job.experience_level,
        responsibilities=parsed_job.responsibilities,
        qualifications=parsed_job.qualifications,
        education=(
            [edu.model_dump() for edu in parsed_job.education]
            if parsed_job.education
            else None
        ),
        skills=(
            [skill.model_dump() for skill in parsed_job.skills]
            if parsed_job.skills
            else None
        ),
        embedding=embedding,
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    return JobDescription(
        id=job.id,
        job_id=str(job.job_id),
        title=job.title,
        company=job.company,
        employment_type=job.employment_type,
        date=job.date,
        description=job.description,
        location=job.location,
        remote=job.remote,
        salary=job.salary,
        experience=job.experience,
        min_experience_years=job.min_experience_years,
        max_experience_years=job.max_experience_years,
        experience_level=job.experience_level,
        responsibilities=job.responsibilities,
        qualifications=job.qualifications,
        education=job.education,
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
