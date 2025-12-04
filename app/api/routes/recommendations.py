"""
API routes for recommendation endpoints.
"""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_async_db
from app.schemas.resume import Resume
from app.schemas.job import RecommendationResponse
from app.services.recommendation.recommendation_engine import (
    get_recommendations_for_resume,
)
from app.services.resume.resume_pdf_extractor import extract_text_from_pdf
from app.services.resume.resume_parser import extract_resume_data


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


@router.post("/for-resume", response_model=RecommendationResponse)
async def get_recommendations(
    file: UploadFile = File(..., description="Resume PDF file to analyze"),
    top_k: int = Query(
        10, ge=1, le=50, description="Number of top recommendations to return"
    ),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Get job recommendations for a resume by uploading a PDF file.

    Args:
        file: Uploaded resume PDF file
        top_k: Number of top recommendations to return
        db: Database session

    Returns:
        Recommendation response with ranked job matches

    Raises:
        HTTPException: If file processing fails, no jobs found, or error during recommendation
    """
    # Validate file type
    logger.info(f"Received resume file: {file.filename}")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.warning(f"Invalid file type received: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a PDF resume.",
        )

    # Create temporary file to save uploaded content
    temp_file_path = None
    try:
        # Create a temporary file with .pdf extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file_path = temp_file.name
            # Write uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
            logger.debug(f"Saved resume to temporary file: {temp_file_path}")

        # Extract text from PDF
        logger.info("Extracting text from PDF...")
        resume_text = extract_text_from_pdf(temp_file_path)
        if not resume_text or not resume_text.strip():
            logger.error("Failed to extract text from PDF - empty content")
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF. Please ensure the PDF contains readable text.",
            )
        logger.info(f"Successfully extracted {len(resume_text)} characters from PDF")
        logger.debug(f"Extracted text preview: {resume_text[:500]}...")

        # Parse resume using LLM
        logger.info("Parsing resume content with LLM...")
        try:
            resume = await extract_resume_data(resume_text)
            logger.info(f"Successfully parsed resume for: {resume.basics.name if resume.basics else 'Unknown'}")
        except Exception as e:
            logger.exception(f"Error parsing resume content: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error parsing resume content: {str(e)}"
            )

        # Get recommendations
        logger.info(f"Generating top {top_k} job recommendations...")
        recommendations = await get_recommendations_for_resume(resume, db, top_k=top_k)
        logger.info(f"Successfully generated {len(recommendations.recommendations)} recommendations")
        return recommendations

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Value error during recommendation: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error generating recommendations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating recommendations: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file_path}: {e}")
