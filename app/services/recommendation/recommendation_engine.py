"""
Recommendation engine for matching resumes with job vacancies.
"""

import asyncio
import hashlib
import json
import logging
import re
from datetime import date
from typing import List, Tuple, Dict, Optional
from sqlalchemy import select, func, Float
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.core.cache import cache_manager
from app.models.job import Job
from app.schemas.resume import Resume
from app.schemas.job import (
    JobRecommendation,
    RecommendationResponse,
    MatchingFactors,
)
from app.services.job.job_embedder import generate_resume_embedding_async

logger = logging.getLogger(__name__)


def parse_iso8601_date(date_str: Optional[str]) -> Optional[date]:
    """
    Parse an ISO 8601 date string (YYYY, YYYY-MM, or YYYY-MM-DD) into a date object.

    Args:
        date_str: Date string in ISO 8601 format

    Returns:
        date object or None if parsing fails
    """
    if not date_str:
        return None

    try:
        parts = date_str.split("-")
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return date(year, month, day)
    except (ValueError, IndexError):
        return None


def calculate_months_between_dates(start: date, end: date) -> int:
    """
    Calculate the number of months between two dates.

    Args:
        start: Start date
        end: End date

    Returns:
        Number of months (minimum 0)
    """
    if end < start:
        return 0

    months = (end.year - start.year) * 12 + (end.month - start.month)
    # Add partial month if end day >= start day
    if end.day >= start.day:
        months += 1

    return max(0, months)


# Weights for different matching factors
WEIGHTS = {
    "skills": 0.25,
    "experience": 0.25,
    "education": 0.25,
    "semantic": 0.25,
}


def calculate_skills_match(
    resume: Resume, job_skills: Optional[List[Dict]]
) -> Tuple[float, List[str], List[str]]:
    """
    Calculate skills match score between resume and job.

    Args:
        resume: Resume object
        job_skills: List of skill dictionaries from job

    Returns:
        Tuple of (score, matched_skills, missing_skills)
    """
    if not job_skills or not resume.skills:
        return 0.0, [], []

    # Extract resume skills (skill names + keywords)
    resume_skill_set = set()
    if resume.skills:
        for skill in resume.skills:
            if skill.name:
                resume_skill_set.add(skill.name.lower())
            if skill.keywords:
                resume_skill_set.update([k.lower() for k in skill.keywords])

    # Extract job skills (skill names + keywords)
    job_skill_names = []
    job_skill_set = set()
    for skill_dict in job_skills:
        if isinstance(skill_dict, dict):
            if skill_dict.get("name"):
                name = skill_dict["name"].lower()
                job_skill_names.append(skill_dict["name"])
                job_skill_set.add(name)
            if skill_dict.get("keywords"):
                job_skill_set.update([k.lower() for k in skill_dict["keywords"]])

    if not job_skill_set:
        return 0.0, [], []

    # Calculate matches
    matched = resume_skill_set.intersection(job_skill_set)
    matched_skills = [s for s in job_skill_names if s.lower() in matched]

    missing = job_skill_set - resume_skill_set
    missing_skills = list(missing)

    # Score: ratio of matched skills to required skills
    score = len(matched) / len(job_skill_set) if job_skill_set else 0.0

    return min(score, 1.0), matched_skills, missing_skills


def calculate_experience_match(
    resume: Resume,
    job_experience: Optional[str],
    job_qualifications: Optional[List[str]],
) -> float:
    """
    Calculate experience match score.

    Args:
        resume: Resume object
        job_experience: Experience level from job (Entry-level, Mid-level, Senior, etc.)
        job_qualifications: Optional list of qualification strings which may contain
            experience requirements (e.g., "3+ years of experience in web development").

    Returns:
        Score between 0.0 and 1.0
    """
    # Return neutral score if no experience requirements provided
    if not job_experience and not job_qualifications:
        return 0.5

    # Calculate total months of experience from resume
    total_months = 0
    today = date.today()

    if resume.work:
        for work_item in resume.work:
            start_date = parse_iso8601_date(work_item.start_date)

            # Skip if no start date - we can't calculate experience without it
            if not start_date:
                continue

            # Use end_date if available, otherwise assume current position (use today)
            end_date = parse_iso8601_date(work_item.end_date)
            if not end_date:
                end_date = today

            months = calculate_months_between_dates(start_date, end_date)
            total_months += months

    # Convert total months to years (as a float for precision)
    resume_years = total_months / 12.0

    # Parse job experience requirement from experience field
    job_years_required = 0
    found_numeric = False

    # Check experience field first for numerical values
    if job_experience:
        experience_text = job_experience.lower()
        year_match = re.search(
            r"(\d+)[\+\-]?\s*(?:to)?\s*(\d+)?\s*years?", experience_text
        )
        if year_match:
            job_years_required = int(year_match.group(1))
            found_numeric = True

    # If no numeric value found in experience, check qualifications
    if not found_numeric and job_qualifications:
        qual_text = " ".join(job_qualifications).lower()
        year_match = re.search(r"(\d+)[\+\-]?\s*(?:to)?\s*(\d+)?\s*years?", qual_text)
        if year_match:
            job_years_required = int(year_match.group(1))
            found_numeric = True

    # If still no numeric value, check for experience levels
    if not found_numeric:
        # Check experience field for levels
        if job_experience:
            experience_text = job_experience.lower()
            if (
                "entry" in experience_text
                or "junior" in experience_text
                or "beginner" in experience_text
            ):
                job_years_required = 0
            elif "mid" in experience_text or "intermediate" in experience_text:
                job_years_required = 3
            elif "senior" in experience_text or "lead" in experience_text:
                job_years_required = 5
            elif (
                "principal" in experience_text
                or "staff" in experience_text
                or "expert" in experience_text
            ):
                job_years_required = 8

        # If experience field didn't provide level info, check qualifications
        if job_years_required == 0 and job_qualifications:
            qual_text = " ".join(job_qualifications).lower()

            if "entry" in qual_text or "junior" in qual_text or "beginner" in qual_text:
                job_years_required = 0
            elif "mid" in qual_text or "intermediate" in qual_text:
                job_years_required = 3
            elif "senior" in qual_text or "lead" in qual_text:
                job_years_required = 5
            elif "principal" in qual_text or "staff" in qual_text:
                job_years_required = 8

    # Calculate score based on how well resume experience matches requirement
    if resume_years >= job_years_required:
        # Has enough experience
        if resume_years <= job_years_required + 3:
            score = 1.0  # Perfect match
        else:
            score = 0.9  # Overqualified slightly
    else:
        # Doesn't have enough experience - penalize proportionally
        if job_years_required > 0:
            score = resume_years / job_years_required
        else:
            score = 1.0  # No requirement

    return min(score, 1.0)


def calculate_education_match(
    resume: Resume,
    job_education: Optional[List[Dict]],
    job_qualifications: Optional[List[str]] = None,
) -> float:
    """
    Calculate education match score.

    Args:
        resume: Resume object
        job_education: List of education requirement dictionaries from job
        job_qualifications: Optional list of qualification strings (fallback)

    Returns:
        Score between 0.0 and 1.0
    """
    # Check if we have education requirements
    has_education_requirements = job_education and len(job_education) > 0

    if not has_education_requirements and not job_qualifications:
        return 0.5  # Neutral score if no requirement

    if not resume.education:
        return 0.0  # No education info in resume

    # Extract education levels from resume
    resume_degrees = set()
    resume_fields = set()

    for edu in resume.education:
        if edu.study_type:
            resume_degrees.add(edu.study_type.lower())
        if edu.area:
            resume_fields.add(edu.area.lower())

    # Degree level hierarchy
    degree_levels = {
        "phd": 4,
        "doctorate": 4,
        "master": 3,
        "msc": 3,
        "mba": 3,
        "bachelor": 2,
        "bsc": 2,
        "bs": 2,
        "ba": 2,
        "associate": 1,
        "high school": 0,
        "diploma": 0,
    }

    # Get highest degree level from resume
    resume_level = 0
    for degree in resume_degrees:
        for key, value in degree_levels.items():
            if key in degree:
                resume_level = max(resume_level, value)
                break

    required_level = 0
    required_fields = set()
    has_required_education = False

    # Parse structured education requirements (preferred)
    if has_education_requirements:
        for edu_req in job_education:
            if isinstance(edu_req, dict):
                degree = (
                    edu_req.get("degree", "").lower() if edu_req.get("degree") else ""
                )
                field = edu_req.get("field", "").lower() if edu_req.get("field") else ""
                is_required = edu_req.get("required", True)  # Default to required

                if is_required:
                    has_required_education = True

                # Get required degree level
                for key, value in degree_levels.items():
                    if key in degree:
                        if is_required:
                            required_level = max(required_level, value)
                        break

                # Collect required fields
                if field:
                    required_fields.add(field)
    else:
        # Fallback: Parse from qualifications (legacy behavior)
        if job_qualifications:
            job_qual_text = " ".join(job_qualifications).lower()
            for key, value in degree_levels.items():
                if key in job_qual_text:
                    required_level = max(required_level, value)
                    has_required_education = True

    # Check field relevance
    field_match = False
    if required_fields:
        # Check if any resume field matches required fields
        for resume_field in resume_fields:
            for req_field in required_fields:
                if req_field in resume_field or resume_field in req_field:
                    field_match = True
                    break
            if field_match:
                break
    elif job_qualifications:
        # Fallback: check against qualifications text
        for field in resume_fields:
            if any(field in qual.lower() for qual in job_qualifications):
                field_match = True
                break

    # Calculate score based on degree level
    if resume_level >= required_level:
        base_score = 1.0
    elif resume_level == required_level - 1:
        base_score = 0.7
    else:
        base_score = 0.4

    # Bonus for field match
    if field_match:
        base_score = min(base_score + 0.2, 1.0)
    elif required_fields and has_required_education:
        # Slight penalty if specific field is required but not matched
        base_score = max(base_score - 0.1, 0.0)

    return base_score


def _build_resume_text(resume: Resume) -> str:
    """
    Build a comprehensive text representation of a resume for embedding.

    Args:
        resume: Resume object

    Returns:
        Combined text string
    """
    resume_text_parts = []
    if resume.basics:
        if resume.basics.summary:
            resume_text_parts.append(resume.basics.summary)
        if resume.basics.label:
            resume_text_parts.append(f"Role: {resume.basics.label}")

    if resume.work:
        for work in resume.work:
            if work.position:
                resume_text_parts.append(f"Position: {work.position}")
            if work.summary:
                resume_text_parts.append(work.summary)

    if resume.skills:
        skill_names = [s.name for s in resume.skills if s.name]
        if skill_names:
            resume_text_parts.append(f"Skills: {', '.join(skill_names)}")

    return "\n".join(resume_text_parts)


def _calculate_similarity_from_distance(distance: Optional[float]) -> float:
    """
    Convert cosine distance to similarity score.

    Args:
        distance: Cosine distance (0 to 2) or None

    Returns:
        Similarity score (0.0 to 1.0)
    """
    if distance is None:
        return 0.0

    # pgvector cosine_distance is [0, 2], where 0 is identical
    # Convert to similarity: similarity = 1 - (distance / 2)
    return max(0.0, min(1.0 - (distance / 2.0), 1.0))


def generate_explanation(
    matching_factors: MatchingFactors,
    matched_skills: List[str],
    resume: Resume,
    job_title: Optional[str],
) -> str:
    """
    Generate human-readable explanation for the match.

    Args:
        matching_factors: Breakdown of matching scores
        matched_skills: List of matched skills
        resume: Resume object
        job_title: Job title
        job_experience: Job experience requirement

    Returns:
        Explanation string
    """
    explanations = []

    # Determine strongest factor
    factors = {
        "skills": matching_factors.skills_match,
        "experience": matching_factors.experience_match,
        "education": matching_factors.education_match,
        "semantic": matching_factors.semantic_similarity,
    }
    strongest_factor = max(factors, key=factors.get)

    # Skills explanation
    if matching_factors.skills_match > 0.7:
        if matched_skills:
            skill_list = ", ".join(matched_skills)
            explanations.append(f"Strong skills match including {skill_list}")
    elif matching_factors.skills_match > 0.4:
        explanations.append("Moderate skills alignment")

    # Experience explanation
    if matching_factors.experience_match > 0.8:
        if resume.work and len(resume.work) > 0:
            years = len(resume.work) * 2  # Rough estimate
            explanations.append(f"Relevant experience (approximately {years} years)")

    # Education explanation
    if matching_factors.education_match > 0.7:
        if resume.education and len(resume.education) > 0:
            explanations.append("Educational background aligns well")

    # Semantic explanation
    if matching_factors.semantic_similarity > 0.7:
        explanations.append("Overall profile matches job requirements")

    # Combine explanations
    if not explanations:
        return "May be a potential match worth considering"

    explanation = ". ".join(explanations) + "."

    # Add summary based on strongest factor
    if strongest_factor == "skills":
        explanation = "Strong match based on technical skills. " + explanation
    elif strongest_factor == "experience":
        explanation = "Good match based on experience level. " + explanation
    elif strongest_factor == "semantic":
        explanation = "Excellent overall fit for this role. " + explanation

    return explanation


def _calculate_job_score(
    resume: Resume,
    job: Job,
    semantic_score: float,
) -> JobRecommendation:
    """
    Calculate the match score for a single job against a resume.

    Args:
        resume: Resume object
        job: Job model instance
        semantic_score: Pre-calculated semantic similarity score

    Returns:
        JobRecommendation with all match details
    """
    # Calculate individual factors (lightweight sync operations)
    skills_score, matched_skills, missing_skills = calculate_skills_match(
        resume, job.skills
    )
    experience_score = calculate_experience_match(
        resume, job.experience, job.qualifications
    )
    education_score = calculate_education_match(
        resume, job.education, job.qualifications
    )

    # Calculate weighted overall score
    overall_score = (
        WEIGHTS["skills"] * skills_score
        + WEIGHTS["experience"] * experience_score
        + WEIGHTS["education"] * education_score
        + WEIGHTS["semantic"] * semantic_score
    )

    matching_factors = MatchingFactors(
        skills_match=skills_score,
        experience_match=experience_score,
        education_match=education_score,
        semantic_similarity=semantic_score,
    )

    explanation = generate_explanation(
        matching_factors, matched_skills, resume, job.title
    )

    return JobRecommendation(
        job_id=str(job.job_id),
        job_title=job.title,
        company=job.company,
        match_score=overall_score,
        matching_factors=matching_factors,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        explanation=explanation,
    )


async def get_recommendations_for_resume(
    resume: Resume, db: AsyncSession, top_k: int = 10
) -> RecommendationResponse:
    """
    Get job recommendations for a resume.

    Uses caching to avoid re-computing recommendations for the same resume.
    Implements pre-filtering to limit detailed scoring to top candidates.

    Args:
        resume: Resume object
        db: Database session
        top_k: Number of top recommendations to return

    Returns:
        RecommendationResponse with ranked job matches
    """
    # Generate cache key from resume + top_k
    resume_dict = resume.model_dump()
    cache_key_data = json.dumps(resume_dict, sort_keys=True) + f":top_k={top_k}"
    cache_key_hash = hashlib.sha256(cache_key_data.encode()).hexdigest()[:16]
    cache_key = f"recommendations:{cache_key_hash}"

    # Try to get from cache
    cached_result = await cache_manager.get(cache_key)
    if cached_result is not None:
        logger.info(f"Recommendations cache HIT for hash {cache_key_hash}")
        return cached_result

    logger.info(f"Recommendations cache MISS for hash {cache_key_hash}")

    # 1. Generate resume embedding (ONCE) - this is also cached
    resume_text = _build_resume_text(resume)

    try:
        # Use the async version (cached)
        resume_embedding = await generate_resume_embedding_async(resume_text)
    except Exception as e:
        logger.error(f"Error generating resume embedding: {e}")
        resume_embedding = []

    # 2. PRE-FILTERING: Fetch top candidates from vector search
    # Instead of fetching ALL jobs, we fetch top N candidates based on semantic similarity
    # This reduces the number of jobs we need to do detailed scoring on
    pre_filter_limit = settings.VECTOR_SEARCH_CANDIDATES  # Default: 100

    if resume_embedding:
        # Fetch top candidates by vector similarity
        query = (
            select(
                Job, Job.embedding.cosine_distance(resume_embedding).label("distance")
            )
            .order_by(Job.embedding.cosine_distance(resume_embedding))
            .limit(pre_filter_limit)
        )
    else:
        # If no embedding, just fetch first N jobs
        query = select(Job, func.cast(0, Float).label("distance")).limit(
            pre_filter_limit
        )

    result = await db.execute(query)
    rows = result.all()  # Returns list of (Job, distance) tuples

    if not rows:
        raise ValueError("No jobs found in database")

    logger.info(f"Pre-filtered to {len(rows)} candidates for detailed scoring")

    # 3. Calculate detailed scores only for pre-filtered candidates
    job_scores = []
    for row in rows:
        job = row[0]
        distance = row[1]

        semantic_score = 0.0
        if resume_embedding:
            semantic_score = _calculate_similarity_from_distance(distance)

        recommendation = _calculate_job_score(resume, job, semantic_score)
        job_scores.append(recommendation)

    # Sort by overall score (descending)
    job_scores.sort(key=lambda x: x.match_score, reverse=True)

    # Return top K recommendations
    recommendations = job_scores[:top_k]

    # Extract candidate info
    candidate_id = "resume_" + (
        resume.basics.name if resume.basics and resume.basics.name else "unknown"
    )
    candidate_name = (
        resume.basics.name if resume.basics and resume.basics.name else None
    )

    response = RecommendationResponse(
        candidate_id=candidate_id,
        candidate_name=candidate_name,
        recommendations=recommendations,
    )

    # Cache the result
    await cache_manager.set(cache_key, response, ttl=settings.CACHE_RECOMMENDATIONS_TTL)

    logger.info(f"Generated {len(recommendations)} recommendations")

    return response
