from pathlib import Path
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.resume.resume_pdf_extractor import extract_text_from_pdf
from app.services.resume.resume_parser import extract_resume_data


def process_file(file_path: str, output_dir: str):
    print(f"Processing {file_path}...")
    text = extract_text_from_pdf(file_path)
    if not text:
        print(f"Failed to extract text from {file_path}")
        return

    try:
        resume = extract_resume_data(text)
        output_path = Path(output_dir) / (Path(file_path).stem + ".json")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(resume.model_dump_json(by_alias=True, indent=2))
        print(f"Saved parsed data to {output_path}")
    except Exception as e:
        print(f"Failed to parse {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Parse CVs from PDF to JSON.")
    parser.add_argument("input", help="Path to PDF file or directory containing PDFs")
    parser.add_argument(
        "--output", default="parsed_resumes", help="Output directory for JSON files"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)

    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            process_file(str(input_path), str(output_dir))
        else:
            print("Input file must be a PDF")
    elif input_path.is_dir():
        for file_path in input_path.glob("*.pdf"):
            process_file(str(file_path), str(output_dir))
    else:
        print("Invalid input path")


if __name__ == "__main__":
    main()
