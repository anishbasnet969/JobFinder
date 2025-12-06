"""
Recommendation engine for matching resumes with job vacancies.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import date
from typing import List, Tuple, Dict, Optional
from sqlalchemy import select, func, Float
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings
from app.core.cache import cache_manager
from app.core.metrics import metrics_tracker, TimingContext
from app.models.job import Job
from app.schemas.resume import Resume
from app.schemas.recommendation import (
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
                resume_skill_set.update(k.lower() for k in skill.keywords)

    # Extract job skills (skill names + keywords)
    job_skill_names = []  # Original casing for reporting
    job_skill_set = set()
    for skill_dict in job_skills:
        if isinstance(skill_dict, dict):
            if skill_dict.get("name"):
                job_skill_names.append(skill_dict["name"])
                job_skill_set.add(skill_dict["name"].lower())
            if skill_dict.get("keywords"):
                job_skill_names.extend(skill_dict["keywords"])
                job_skill_set.update(k.lower() for k in skill_dict["keywords"])

    if not job_skill_set:
        return 0.0, [], []

    # Calculate exact matches using set intersection
    matched = resume_skill_set.intersection(job_skill_set)
    matched_skills = [s for s in job_skill_names if s.lower() in matched]

    # Calculate missing skills
    missing = job_skill_set - matched
    missing_skills = list(missing)[:10]  # Limit to top 10 missing skills

    # Score: ratio of matched skills to required skills
    score = len(matched) / len(job_skill_set)

    return min(score, 1.0), matched_skills, missing_skills


def calculate_experience_match(
    resume: Resume,
    job_experience: Optional[str],
    job_qualifications: Optional[List[str]],
    min_experience_years: Optional[int] = None,
    max_experience_years: Optional[int] = None,
    experience_level: Optional[str] = None,
) -> float:
    """
    Calculate experience match score.

    Args:
        resume: Resume object
        job_experience: Raw experience text from job (e.g., "5 to 15 Years")
        job_qualifications: Optional list of qualification strings
        min_experience_years: Pre-parsed minimum years required
        max_experience_years: Pre-parsed maximum years (for overqualification check)
        experience_level: Pre-parsed level (Entry-level, Mid-level, Senior, Lead, Principal)

    Returns:
        Score between 0.0 and 1.0
    """
    # Return neutral score if no experience requirements provided
    if (
        not job_experience
        and not job_qualifications
        and min_experience_years is None
        and experience_level is None
    ):
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

    # Parse job experience requirement
    job_min_years = 0
    job_max_years = None
    found_numeric = False

    # Use pre-parsed min/max years if available
    if min_experience_years is not None:
        job_min_years = min_experience_years
        found_numeric = True
    if max_experience_years is not None:
        job_max_years = max_experience_years

    # Fallback: Check experience field for numerical values
    if not found_numeric and job_experience:
        experience_text = job_experience.lower()
        year_match = re.search(r"(\d+)\s*(?:to|\-)\s*(\d+)\s*years?", experience_text)
        if year_match:
            job_min_years = int(year_match.group(1))
            job_max_years = int(year_match.group(2))
            found_numeric = True
        else:
            # Try single number pattern
            single_match = re.search(r"(\d+)\+?\s*years?", experience_text)
            if single_match:
                job_min_years = int(single_match.group(1))
                found_numeric = True

    # If no numeric value found, check qualifications
    if not found_numeric and job_qualifications:
        qual_text = " ".join(job_qualifications).lower()
        year_match = re.search(r"(\d+)\s*(?:to|\-)\s*(\d+)\s*years?", qual_text)
        if year_match:
            job_min_years = int(year_match.group(1))
            job_max_years = int(year_match.group(2))
            found_numeric = True
        else:
            single_match = re.search(r"(\d+)\+?\s*years?", qual_text)
            if single_match:
                job_min_years = int(single_match.group(1))
                found_numeric = True

    # If still no numeric value, use experience_level or detect from text
    if not found_numeric:
        level = experience_level.lower() if experience_level else ""
        if not level and job_experience:
            level = job_experience.lower()

        if "entry" in level or "junior" in level or "beginner" in level:
            job_min_years = 0
            job_max_years = 2
        elif "mid" in level or "intermediate" in level:
            job_min_years = 3
            job_max_years = 5
        elif "senior" in level:
            job_min_years = 5
            job_max_years = 8
        elif "lead" in level:
            job_min_years = 8
            job_max_years = 12
        elif "principal" in level or "staff" in level or "expert" in level:
            job_min_years = 10
            job_max_years = None  # No upper limit

    # Calculate score based on how well resume experience matches requirement
    if resume_years >= job_min_years:
        if job_max_years is not None and resume_years > job_max_years:
            # Overqualified - apply slight penalty based on how much over
            excess_years = resume_years - job_max_years
            if excess_years <= 3:
                score = 0.9  # Slightly overqualified
            elif excess_years <= 6:
                score = 0.8  # Moderately overqualified
            else:
                score = 0.7  # Significantly overqualified
        else:
            score = 1.0  # Perfect fit within range
    else:
        # Underqualified - penalize proportionally
        if job_min_years > 0:
            score = resume_years / job_min_years
        else:
            score = 1.0  # No minimum requirement

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

    # Degree level hierarchy - expanded to include common abbreviations
    degree_levels = {
        # Level 4 - Doctorate
        "phd": 4,
        "doctorate": 4,
        "doctoral": 4,
        # Level 3 - Master's
        "master": 3,
        "msc": 3,
        "ms": 3,
        "mba": 3,
        "m.tech": 3,
        "mtech": 3,
        "mca": 3,
        "m.e.": 3,
        "m.s.": 3,
        # Level 2 - Bachelor's
        "bachelor": 2,
        "bsc": 2,
        "bs": 2,
        "ba": 2,
        "b.tech": 2,
        "btech": 2,
        "bca": 2,
        "b.e.": 2,
        "b.s.": 2,
        "undergraduate": 2,
        # Level 1 - Associate
        "associate": 1,
        # Level 0 - High School / Diploma
        "high school": 0,
        "diploma": 0,
        "ged": 0,
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


async def generate_explanation(
    matching_factors: MatchingFactors,
    matched_skills: List[str],
    missing_skills: List[str],
    resume: Resume,
    job: Job,
) -> str:
    """
    Generate human-readable explanation for the match using LLM.

    Args:
        matching_factors: Breakdown of matching scores
        matched_skills: List of matched skills
        missing_skills: List of missing skills
        resume: Resume object
        job: Job model instance

    Returns:
        Explanation string
    """
    if not settings.GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY not set")
        return "May be a potential match worth considering."

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            google_api_key=settings.GOOGLE_API_KEY,
        )

        # Build resume summary
        resume_summary_parts = []
        if resume.basics and resume.basics.name:
            resume_summary_parts.append(f"Name: {resume.basics.name}")
        if resume.basics and resume.basics.label:
            resume_summary_parts.append(f"Current Role: {resume.basics.label}")
        if resume.work:
            recent_work = resume.work[0] if resume.work else None
            if recent_work:
                work_info = f"Recent Experience: {recent_work.position or 'N/A'} at {recent_work.name or 'N/A'}"
                resume_summary_parts.append(work_info)
        if resume.skills:
            skill_names = [s.name for s in resume.skills if s.name][:10]
            if skill_names:
                resume_summary_parts.append(f"Key Skills: {', '.join(skill_names)}")
        if resume.education:
            edu = resume.education[0] if resume.education else None
            if edu:
                edu_info = f"Education: {edu.study_type or ''} in {edu.area or 'N/A'} from {edu.institution or 'N/A'}"
                resume_summary_parts.append(edu_info)

        resume_summary = "\n".join(resume_summary_parts)

        # Build job summary
        job_summary_parts = [f"Job Title: {job.title or 'N/A'}"]
        if job.company:
            job_summary_parts.append(f"Company: {job.company}")
        if job.experience:
            job_summary_parts.append(f"Experience Required: {job.experience}")
        if job.skills:
            job_skill_names = []
            for s in job.skills[:10]:
                if isinstance(s, dict) and s.get("name"):
                    job_skill_names.append(s["name"])
            if job_skill_names:
                job_summary_parts.append(
                    f"Required Skills: {', '.join(job_skill_names)}"
                )

        job_summary = "\n".join(job_summary_parts)

        # Build match context
        matched_skills_str = (
            ", ".join(matched_skills) if matched_skills else "None identified"
        )
        missing_skills_str = (
            ", ".join(missing_skills[:5]) if missing_skills else "None identified"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert job matching system generating explanations based on scoring factors.
                    
                    The recommendation engine evaluates candidates using three key factors:
                    1. Skills Match - How well the candidate's technical and professional skills align with job requirements
                    2. Experience Match - How the candidate's work history and years of experience meet the role's requirements
                    3. Education Match - How the candidate's educational qualifications fit the job's requirements
                    
                    Additionally, an Overall Fit score indicates general profile alignment.
                    
                    Generate a brief explanation (2-3 sentences) that specifically addresses the three key factors.
                    Mention how each factor contributes to the match, highlighting stronger areas.
                    If matched skills are provided, reference specific skills that align.
                    Write in third person, referring to "the candidate" or "this candidate".
                    Be objective and factual.
                    
                    IMPORTANT: Do not use bullet points or lists. Return ONLY the explanation text.""",
                ),
                (
                    "user",
                    """Candidate Profile:
                {resume_summary}

                Job Opportunity:
                {job_summary}

                Scoring Factors:
                - Skills Match: {skills_match}
                - Experience Match: {experience_match}
                - Education Match: {education_match}
                - Overall Fit: {overall_fit}

                Matched Skills: {matched_skills}
                Skills Gap: {missing_skills}""",
                ),
            ]
        )

        chain = prompt | llm

        result = await chain.ainvoke(
            {
                "resume_summary": resume_summary,
                "job_summary": job_summary,
                "skills_match": f"{matching_factors.skills_match:.0%}",
                "experience_match": f"{matching_factors.experience_match:.0%}",
                "education_match": f"{matching_factors.education_match:.0%}",
                "overall_fit": f"{matching_factors.semantic_similarity:.0%}",
                "matched_skills": matched_skills_str,
                "missing_skills": missing_skills_str,
            }
        )

        return result.content.strip()

    except Exception as e:
        logger.warning(f"Error generating LLM explanation: {e}")
        return "May be a potential match worth considering."


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
        resume,
        job.experience,
        job.qualifications,
        min_experience_years=getattr(job, "min_experience_years", None),
        max_experience_years=getattr(job, "max_experience_years", None),
        experience_level=getattr(job, "experience_level", None),
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

    return JobRecommendation(
        job_id=str(job.job_id),
        job_title=job.title,
        company=job.company,
        match_score=overall_score,
        matching_factors=matching_factors,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        explanation="",  # Will be generated in batch
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
    start_time = time.time()

    # Generate cache key from resume + top_k
    resume_dict = resume.model_dump(mode="json")
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

    # Track database query time
    async with TimingContext("database_vector_search"):
        result = await db.execute(query)
        rows = result.all()  # Returns list of (Job, distance) tuples

    if not rows:
        raise ValueError("No jobs found in database")

    logger.info(f"Pre-filtered to {len(rows)} candidates for detailed scoring")

    # Build a map of job_id to Job for explanation generation
    jobs_map: Dict[str, Job] = {}

    # 3. Calculate detailed scores only for pre-filtered candidates
    job_scores = []
    for row in rows:
        job = row[0]
        distance = row[1]

        jobs_map[str(job.job_id)] = job

        semantic_score = 0.0
        if resume_embedding:
            semantic_score = _calculate_similarity_from_distance(distance)

        recommendation = _calculate_job_score(resume, job, semantic_score)
        job_scores.append(recommendation)

    # Sort by overall score (descending)
    job_scores.sort(key=lambda x: x.match_score, reverse=True)

    # Return top K recommendations
    recommendations = job_scores[:top_k]

    # 4. Generate LLM explanations for top recommendations in batch
    if settings.GOOGLE_API_KEY:

        async def generate_single_explanation(rec: JobRecommendation) -> None:
            job = jobs_map.get(rec.job_id)
            if job:
                rec.explanation = await generate_explanation(
                    rec.matching_factors,
                    rec.matched_skills,
                    rec.missing_skills,
                    resume,
                    job,
                )

        await asyncio.gather(
            *[generate_single_explanation(rec) for rec in recommendations]
        )

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

    # Record timing metric
    duration_ms = (time.time() - start_time) * 1000
    await metrics_tracker.record_timing("recommendation_generation", duration_ms)

    logger.info(
        f"Generated {len(recommendations)} recommendations in {duration_ms:.2f}ms"
    )

    return response
