"""
Reranker for job recommendations using Cohere API with fallback support.

This module provides intelligent reranking using Cohere's specialized reranker model.
Falls back to original ranking on rate limits or API issues.
"""

import logging
from typing import List, Tuple

import cohere

from app.config import settings
from app.models.job import Job
from app.schemas.resume import Resume

logger = logging.getLogger(__name__)

# Initialize Cohere client
try:
    if settings.COHERE_API_KEY:
        cohere_client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        logger.info("Cohere client initialized successfully")
    else:
        logger.warning("COHERE_API_KEY not configured. Reranking will be skipped.")
        cohere_client = None
except Exception as e:
    logger.warning(
        f"Failed to initialize Cohere client: {e}. Reranking will be skipped."
    )
    cohere_client = None


def _build_resume_query(resume: Resume) -> str:
    """
    Build a text query from the resume for semantic matching.
    """
    parts = []
    if resume.basics:
        if resume.basics.label:
            parts.append(f"Role: {resume.basics.label}")
        if resume.basics.summary:
            parts.append(f"Summary: {resume.basics.summary[:400]}")

    if resume.skills:
        # Collect top skills with keywords
        all_skills = []
        for s in resume.skills:
            if s.name:
                all_skills.append(s.name)
            if s.keywords:
                all_skills.extend(s.keywords)

        # Deduplicate while preserving order
        seen = set()
        skills = []
        for skill in all_skills:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                skills.append(skill)

        if skills:
            parts.append(f"Skills: {', '.join(skills)}")

    if resume.work:
        # Latest work experience
        work = resume.work[0]
        if work.position and work.name:
            parts.append(f"Recent: {work.position} at {work.name}")
            if work.summary:
                parts.append(f"Experience: {work.summary[:200]}")

    return " | ".join(parts)


def _job_to_text(job: Job) -> str:
    """
    Convert a Job object to a text summary for the cross-encoder.
    """
    parts = [job.title or "Untitled"]

    if job.company:
        parts.append(f"at {job.company}")

    if job.skills:
        all_skills = []
        for s in job.skills:
            if isinstance(s, dict):
                if s.get("name"):
                    all_skills.append(s["name"])
                if s.get("keywords"):
                    all_skills.extend(s["keywords"])

        # Deduplicate
        seen = set()
        skill_names = []
        for skill in all_skills:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                skill_names.append(skill)

        if skill_names:
            parts.append(f"Skills: {', '.join(skill_names)}")

    if job.description:
        # Cross-encoders work best with concise text (~512 tokens usually)
        desc = job.description[:400].replace("\n", " ")
        parts.append(f"Desc: {desc}")

    return " ".join(parts)


async def rerank_candidates(
    resume: Resume, candidates: List[Tuple[Job, float]], top_k: int = 25
) -> List[Tuple[Job, float]]:
    """
    Rerank candidates based on resume match using Cohere API.
    Falls back to original ranking on rate limit or API errors.

    Args:
        resume: The candidate's resume
        candidates: List of (Job, distance) tuples from vector search
        top_k: Number of candidates to keep after reranking

    Returns:
        List of reranked (Job, distance) tuples by relevance
    """
    if not candidates:
        return []

    if not settings.RERANKING_ENABLED:
        return candidates[:top_k]

    try:
        # Check if Cohere client is available
        if cohere_client is None:
            logger.debug("Cohere client not available, skipping reranking")
            return candidates[:top_k]

        # Build query from resume
        query = _build_resume_query(resume)

        # Build documents for reranking
        documents = []
        for i, (job, _) in enumerate(candidates):
            text_content = _job_to_text(job)
            documents.append(text_content)

        try:
            # Call Cohere rerank API
            response = cohere_client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=documents,
                top_n=top_k,
            )

            # Map results back to original candidates
            reranked = []
            for result in response.results:
                idx = result.index
                if idx < len(candidates):
                    reranked.append(candidates[idx])

            logger.info(
                f"Cohere reranked {len(candidates)} -> {len(reranked)} candidates"
            )
            return reranked

        except cohere.errors.RateLimitError:
            # Rate limit hit - fall back to original ranking
            logger.warning("Cohere rate limit hit, falling back to original ranking")
            return candidates[:top_k]
        except cohere.errors.TooManyRequestsError:
            # Too many requests - fall back to original ranking
            logger.warning("Cohere too many requests, falling back to original ranking")
            return candidates[:top_k]

    except Exception as e:
        logger.error(f"Error during Cohere reranking: {e}")
        # Fallback to original order
        return candidates[:top_k]
