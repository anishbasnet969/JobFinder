"""
This module defines Pydantic models for job descriptions
according to the JSON Resume Job Description Schema (https://jsonresume.org/job-description-schema).
"""

from enum import Enum
from typing import List, Optional
from pydantic import Field
from app.schemas.base import CustomBaseModel
from app.schemas.common import Iso8601, Location


class Remote(Enum):
    Full = "Full"
    Hybrid = "Hybrid"
    None_ = "None"


class Education(CustomBaseModel):
    """Education requirements for a job position."""

    degree: Optional[str] = Field(
        None,
        description="Required degree level: High School, Associate, Bachelor, Master, PhD",
    )
    field: Optional[str] = Field(
        None,
        description="Required field of study, e.g. Computer Science, Engineering",
    )
    required: Optional[bool] = Field(
        None,
        description="Whether this education requirement is mandatory",
    )


class Skill(CustomBaseModel):
    name: Optional[str] = Field(None, description="e.g. Web Development")
    level: Optional[str] = Field(None, description="e.g. Master")
    keywords: Optional[List[str]] = Field(
        None, description="List some keywords pertaining to this skill"
    )


class JobDescriptionSchema(CustomBaseModel):
    """Base job description schema for creation/updates."""

    title: Optional[str] = Field(None, description="Job title, e.g. Web Developer")
    company: Optional[str] = Field(None, description="Company name, e.g. Microsoft")
    employment_type: Optional[str] = Field(
        None, description="Employment type: Full-time, Part-time, Contract, Internship"
    )
    date: Optional[Iso8601] = None
    description: Optional[str] = Field(
        None,
        description="Full job description text explaining the role and responsibilities",
    )
    location: Optional[Location] = None
    remote: Optional[Remote] = Field(
        None, description="the level of remote work available"
    )
    salary: Optional[str] = Field(
        None, description="Salary information, e.g. $60K-$90K"
    )

    # Experience requirements - comprehensive fields for accurate matching
    experience: Optional[str] = Field(
        None, description="Raw experience text, e.g., '5 to 15 Years'"
    )
    min_experience_years: Optional[int] = Field(
        None,
        description="Minimum years of experience required (e.g., 5 from '5 to 15 Years')",
    )
    max_experience_years: Optional[int] = Field(
        None, description="Maximum years of experience (e.g., 15 from '5 to 15 Years')"
    )
    experience_level: Optional[str] = Field(
        None,
        description="Experience level: Entry-level, Mid-level, Senior, Lead, Principal",
    )

    responsibilities: Optional[List[str]] = Field(
        None, description="List of job responsibilities"
    )
    qualifications: Optional[List[str]] = Field(
        None, description="Required qualifications (non-education related)"
    )
    education: Optional[List[Education]] = Field(
        None, description="Required education qualifications"
    )
    skills: Optional[List[Skill]] = Field(
        None, description="Required skills with proficiency levels"
    )


class JobDescriptionCreate(CustomBaseModel):
    """Simple schema for creating a job with just title and description.

    This schema is used for incoming create requests. The full structured
    job data will be produced by the parser service and stored in the DB.
    """

    title: str = Field(..., description="Job title")
    description: str = Field(..., description="Job description")


class JobDescription(JobDescriptionSchema):
    """Complete job description schema with database ID."""

    id: int = Field(..., description="Database primary key")
    job_id: str = Field(..., description="Unique identifier for the job")

    class Config:
        from_attributes = True
