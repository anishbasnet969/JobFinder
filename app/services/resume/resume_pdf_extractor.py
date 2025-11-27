import pymupdf
from typing import Optional


def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """
    Args:
        file_path: Absolute path to the PDF file.

    Returns:
        Extracted text as a string, or None if extraction fails.
    """
    try:
        doc = pymupdf.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return None
