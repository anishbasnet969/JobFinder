"""
Service for parsing job descriptions using LLM.
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.schemas.job import JobDescriptionSchema

logger = logging.getLogger(__name__)


async def parse_job_description(job_title: str, job_text: str) -> JobDescriptionSchema:
    """Extract structured job data asynchronously using Gemini."""
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=settings.GOOGLE_API_KEY,
    )

    parser = PydanticOutputParser(pydantic_object=JobDescriptionSchema)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert job description parser. Extract structured information from the job description text and format it according to the JSON Resume Job Description Schema.

                CRITICAL INSTRUCTIONS:
                1. Extract ALL available information from the job description text
                2. For location: Extract city, country code, region if mentioned
                3. For skills: Create a structured list with name, level (if mentioned), and relevant keywords
                4. For responsibilities: Extract as a list of distinct responsibility items
                5. For qualifications: Extract education and experience requirements as a list
                6. For employment_type: Identify Full-time, Part-time, Contract, Internship, etc.
                7. For remote: Determine Remote, Hybrid, or On-site based on the description
                8. For experience: Store the raw experience text (e.g., "5 to 15 Years")
                9. For min_experience_years: Extract MINIMUM years as integer (e.g., "5 to 15 Years" -> 5, "3+ years" -> 3)
                10. For max_experience_years: Extract MAXIMUM years as integer (e.g., "5 to 15 Years" -> 15, "3+ years" -> null)
                11. For experience_level: Determine level - Entry-level (0-2 yrs), Mid-level (3-5 yrs), Senior (5-8 yrs), Lead (8+ yrs), Principal (10+ yrs)
                12. For salary: Extract salary information if mentioned
                13. Use the provided job title if not clearly stated in description
                14. Extract company name if mentioned

                DO NOT infer information that isn't explicitly or implicitly stated in the text.
                
                IMPORTANT: Return ONLY valid JSON. Do not include markdown code blocks, backticks, or any other formatting.
                Return ONLY the raw JSON object.""",
            ),
            (
                "user",
                "Job Title: {job_title}\n\nJob Description:\n{job_text}\n\n{format_instructions}",
            ),
        ]
    )

    chain = prompt | llm | parser

    try:
        result = await chain.ainvoke(
            {
                "job_title": job_title,
                "job_text": job_text,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return result
    except Exception as e:
        logger.warning(
            f"Error parsing job description with LLM: {e}. Using fallback schema."
        )
        raise e
