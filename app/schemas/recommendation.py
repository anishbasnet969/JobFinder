"""
Schemas for job recommendations.
"""

from typing import List, Optional

from pydantic import Field

from app.schemas.base import CustomBaseModel


class MatchingFactors(CustomBaseModel):
    """Breakdown of matching scores for different factors."""

    skills_match: float = Field(
        ..., ge=0.0, le=1.0, description="Skills alignment score (0-1)"
    )
    experience_match: float = Field(
        ..., ge=0.0, le=1.0, description="Experience match score (0-1)"
    )
    education_match: float = Field(
        ..., ge=0.0, le=1.0, description="Education match score (0-1)"
    )
    semantic_similarity: float = Field(
        ..., ge=0.0, le=1.0, description="Overall semantic similarity score (0-1)"
    )


class JobRecommendation(CustomBaseModel):
    """Individual job recommendation with match details."""

    job_id: str = Field(..., description="Unique job identifier")
    job_title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    match_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall match score (0-1)"
    )
    matching_factors: MatchingFactors = Field(
        ..., description="Breakdown of individual matching factors"
    )
    matched_skills: List[str] = Field(
        default_factory=list,
        description="Skills from resume that match job requirements",
    )
    missing_skills: List[str] = Field(
        default_factory=list, description="Required skills not present in resume"
    )
    explanation: str = Field(
        ..., description="Human-readable explanation of why this job matches"
    )


class RecommendationResponse(CustomBaseModel):
    """Complete recommendation response for a candidate."""

    candidate_id: str = Field(..., description="Identifier for the candidate/CV")
    candidate_name: Optional[str] = Field(None, description="Candidate's name")
    recommendations: List[JobRecommendation] = Field(
        ..., description="List of recommended jobs, ranked by match score"
    )
