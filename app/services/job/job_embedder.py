"""
Service for generating embeddings for job descriptions.
"""

import asyncio
import logging
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings
from app.schemas.job import JobDescriptionSchema
from app.core.cache import cache_result

logger = logging.getLogger(__name__)


def generate_job_embedding(job: JobDescriptionSchema) -> List[float]:
    """
    Generates a vector embedding for a job description.

    Args:
        job: JobDescriptionSchema object containing job information

    Returns:
        List of floats representing the embedding vector (3072 dimensions for gemini-embedding-001)
    """
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL, google_api_key=settings.GOOGLE_API_KEY
    )

    # Create a comprehensive text representation of the job for embedding
    text_parts = []

    if job.title:
        text_parts.append(f"Title: {job.title}")

    if job.company:
        text_parts.append(f"Company: {job.company}")

    if job.description:
        text_parts.append(f"Description: {job.description}")

    if job.responsibilities:
        text_parts.append(f"Responsibilities: {', '.join(job.responsibilities)}")

    if job.qualifications:
        text_parts.append(f"Qualifications: {', '.join(job.qualifications)}")

    if job.skills:
        # Extract skill names and keywords
        skill_texts = []
        for skill in job.skills:
            if skill.name:
                skill_texts.append(skill.name)
            if skill.keywords:
                skill_texts.extend(skill.keywords)
        if skill_texts:
            text_parts.append(f"Skills: {', '.join(skill_texts)}")

    if job.experience:
        text_parts.append(f"Experience: {job.experience}")

    # Combine all parts into a single text
    combined_text = "\n".join(text_parts)

    try:
        # Generate embedding
        embedding = embeddings.embed_query(combined_text)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise e


def generate_resume_embedding(resume_text: str) -> List[float]:
    """
    Generates a vector embedding for a resume text.

    Args:
        resume_text: Combined text representation of a resume

    Returns:
        List of floats representing the embedding vector

    Raises:
        ValueError: If GOOGLE_API_KEY is not set
        Exception: If embedding generation fails
    """
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL, google_api_key=settings.GOOGLE_API_KEY
    )

    try:
        embedding = embeddings.embed_query(resume_text)
        return embedding
    except Exception as e:
        print(f"Error generating resume embedding: {e}")
        raise e


def _build_job_text(job: JobDescriptionSchema) -> str:
    """
    Build a comprehensive text representation of a job for embedding.

    Args:
        job: JobDescriptionSchema object containing job information

    Returns:
        Combined text string for embedding
    """
    text_parts = []

    if job.title:
        text_parts.append(f"Title: {job.title}")

    if job.company:
        text_parts.append(f"Company: {job.company}")

    if job.description:
        text_parts.append(f"Description: {job.description}")

    if job.responsibilities:
        text_parts.append(f"Responsibilities: {', '.join(job.responsibilities)}")

    if job.qualifications:
        text_parts.append(f"Qualifications: {', '.join(job.qualifications)}")

    if job.skills:
        skill_texts = []
        for skill in job.skills:
            if skill.name:
                skill_texts.append(skill.name)
            if skill.keywords:
                skill_texts.extend(skill.keywords)
        if skill_texts:
            text_parts.append(f"Skills: {', '.join(skill_texts)}")

    if job.experience:
        text_parts.append(f"Experience: {job.experience}")

    return "\n".join(text_parts)


async def generate_job_embedding_async(job: JobDescriptionSchema) -> List[float]:
    """
    Async version: Generates a vector embedding for a job description.

    Args:
        job: JobDescriptionSchema object containing job information

    Returns:
        List of floats representing the embedding vector (3072 dimensions for gemini-embedding-001)
    """
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL, google_api_key=settings.GOOGLE_API_KEY
    )

    combined_text = _build_job_text(job)

    try:
        embedding = await embeddings.aembed_query(combined_text)
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise e


@cache_result(ttl=604800, key_prefix="resume_embedding")
async def generate_resume_embedding_async(resume_text: str) -> List[float]:
    """
    Async version: Generates a vector embedding for a resume text.

    Cached for 7 days to avoid expensive re-computation.

    Args:
        resume_text: Combined text representation of a resume

    Returns:
        List of floats representing the embedding vector

    Raises:
        ValueError: If GOOGLE_API_KEY is not set
        Exception: If embedding generation fails
    """
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL, google_api_key=settings.GOOGLE_API_KEY
    )

    try:
        embedding = await embeddings.aembed_query(resume_text)
        return embedding
    except Exception as e:
        logger.error(f"Error generating resume embedding: {e}")
        raise e
