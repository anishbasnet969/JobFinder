from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.schemas.resume import Resume
from app.config import settings


def extract_resume_data(text: str) -> Resume:
    """
    Extracts structured resume data from text using Gemini.

    Args:
        text: The text content of the resume.

    Returns:
        A Resume object containing the extracted data.
    """
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

                Return ONLY the JSON.""",
            ),
            ("user", "{text}\n\n{format_instructions}"),
        ]
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke(
            {"text": text, "format_instructions": parser.get_format_instructions()}
        )
        return result
    except Exception as e:
        print(f"Error extracting data with LLM: {e}")
        raise e
