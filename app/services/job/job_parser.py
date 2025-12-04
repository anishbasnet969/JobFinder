"""
Service for parsing job descriptions using LLM.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.schemas.job import JobDescriptionSchema


async def parse_job_description(job_title: str, job_text: str) -> JobDescriptionSchema:
    """Extract structured job data asynchronously using Gemini."""
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
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
                6. For type: Identify Full-time, Part-time, Contract, Internship, etc.
                7. For remote: Determine Remote, Hybrid, or On-site based on the description
                8. For experience: Extract experience level (Entry-level, Mid-level, Senior) or years required
                9. For salary: Extract salary information if mentioned
                10. Use the provided job title if not clearly stated in description
                11. Extract company name if mentioned

                DO NOT infer information that isn't explicitly or implicitly stated in the text.
                Return ONLY the JSON.""",
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
        print(f"Error parsing job description with LLM: {e}")
        raise e
