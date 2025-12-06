import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.schemas.resume import Resume
from app.config import settings
from app.core import generate_hash
from app.core.cache import cache_manager
from app.core.metrics import metrics_tracker
import logging

logger = logging.getLogger(__name__)


async def extract_resume_data(text: str) -> Resume:
    """
    Extracts structured resume data from text using Gemini.

    Uses caching to avoid re-parsing identical resume text.
    Cache key is based on hash of the resume text.

    Args:
        text: The text content of the resume.

    Returns:
        A Resume object containing the extracted data.
    """
    start_time = time.time()

    # Generate cache key from text hash
    text_hash = generate_hash(text)
    cache_key = f"resume_parsed:{text_hash}"

    # Try to get from cache
    cached_result = await cache_manager.get(cache_key)
    if cached_result is not None:
        logger.info(f"Resume parsing cache HIT for hash {text_hash[:8]}")
        return cached_result

    logger.info(f"Resume parsing cache MISS for hash {text_hash[:8]}")

    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0, google_api_key=settings.GOOGLE_API_KEY
    )

    parser = PydanticOutputParser(pydantic_object=Resume)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert CV parser. Extract information from the following resume text and format it according to the JSON Resume schema.

                CRITICAL INSTRUCTIONS:
                1. For the 'basics' section: Extract ALL available information (name, email, phone, location, etc.). Do NOT leave fields as null if the information is present in the resume.
                2. For certificates/courses: Only fill the 'issuer' field if it is explicitly mentioned for that specific certificate. If no issuer is mentioned for a certificate, leave it as null. DO NOT copy the issuer from other certificates.
                3. Do NOT infer the issuer. For example, if the certificate is "Data Science", do NOT assume it is from "Coursera". Only use the issuer if it is explicitly stated in the text next to the certificate.
                4. Extract all data accurately from the text provided.
                5. For dates: Use ISO 8601 format (YYYY, YYYY-MM, or YYYY-MM-DD). For current/ongoing positions where end_date is "Present" or similar, set end_date to null.

                Return ONLY the JSON.""",
            ),
            ("user", "{text}\n\n{format_instructions}"),
        ]
    )

    chain = prompt | llm | parser

    try:
        result = await chain.ainvoke(
            {"text": text, "format_instructions": parser.get_format_instructions()}
        )

        # Cache the result for 24 hours
        await cache_manager.set(cache_key, result, ttl=settings.CACHE_RESUME_TTL)

        # Record timing metric
        duration_ms = (time.time() - start_time) * 1000
        await metrics_tracker.record_timing("resume_parsing", duration_ms)

        logger.info(f"Resume parsed and cached in {duration_ms:.2f}ms")

        return result
    except Exception as e:
        logger.error(f"Error extracting data with LLM: {e}")
        raise e
