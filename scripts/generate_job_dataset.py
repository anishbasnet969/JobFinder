"""
Script to generate a clean job descriptions dataset using LLM synthesis.

Reads job data from datasets/Job/job_descriptions.csv, uses Gemini to synthesize
comprehensive job descriptions, and outputs a new CSV with job_id, job_title, job_description.
"""

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings

# LLM prompt template for job description synthesis
SYNTHESIS_PROMPT = """You are a professional job description writer. Given the following job details, 
create a comprehensive, well-structured job description suitable for a job portal.

Job Details:
- Title: {job_title}
- Role: {role}
- Company: {company}
- Location: {location}, {country}
- Work Type: {work_type}
- Experience Required: {experience}
- Education/Qualifications: {qualifications}
- Salary Range: {salary_range}

Key Responsibilities:
{responsibilities}

Required Skills:
{skills}

Benefits:
{benefits}

Original Description:
{job_description}

Generate a professional job description with these sections:
1. **About the Role** - A compelling 2-3 sentence overview
2. **Key Responsibilities** - 4-6 bullet points
3. **Required Qualifications** - Experience level, education requirements
4. **Required Skills** - Technical and soft skills as bullet points  
5. **Compensation & Benefits** - Salary range and benefits
6. **Location & Work Arrangement** - Where and how (remote/on-site)

Make the description detailed and structured so an AI can easily parse and extract:
- Required years of experience
- Required education level
- List of required skills
- Job responsibilities
- Salary information
- Work type (full-time, part-time, etc.)

Output ONLY the job description text, no additional commentary."""


def clean_value(val) -> str:
    """Clean a value, converting NaN to empty string."""
    if pd.isna(val) or val == "nan":
        return ""
    return str(val).strip()


def extract_job_data(row) -> dict:
    """Extract relevant fields from a CSV row."""
    return {
        "job_id": clean_value(row.get("Job Id", "")),
        "job_title": clean_value(row.get("Job Title", "")),
        "role": clean_value(row.get("Role", "")),
        "company": clean_value(row.get("Company", "")),
        "location": clean_value(row.get("location", "")),
        "country": clean_value(row.get("Country", "")),
        "work_type": clean_value(row.get("Work Type", "")),
        "experience": clean_value(row.get("Experience", "")),
        "qualifications": clean_value(row.get("Qualifications", "")),
        "salary_range": clean_value(row.get("Salary Range", "")),
        "responsibilities": clean_value(row.get("Responsibilities", "")),
        "skills": clean_value(row.get("skills", "")),
        "benefits": clean_value(row.get("Benefits", "")),
        "job_description": clean_value(row.get("Job Description", "")),
    }


async def synthesize_job_description(
    llm: ChatGoogleGenerativeAI,
    job_data: dict,
    job_index: int,
) -> Optional[str]:
    """
    Use LLM to synthesize a comprehensive job description.

    Args:
        llm: The LLM instance
        job_data: Dictionary with job fields
        job_index: Row index for logging

    Returns:
        Synthesized job description or None if failed
    """
    try:
        prompt = SYNTHESIS_PROMPT.format(**job_data)
        response = await llm.ainvoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"  [X] Error synthesizing job [{job_index}]: {e}")
        return None


async def process_batch(
    llm: ChatGoogleGenerativeAI,
    batch: list[tuple[int, dict]],
    batch_num: int,
    total_batches: int,
) -> list[tuple[str, str, str]]:
    """
    Process a batch of jobs concurrently.

    Args:
        llm: The LLM instance
        batch: List of (index, job_data) tuples
        batch_num: Current batch number
        total_batches: Total number of batches

    Returns:
        List of (job_id, job_title, job_description) tuples
    """
    print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} jobs...")

    tasks = [synthesize_job_description(llm, job_data, idx) for idx, job_data in batch]

    results = await asyncio.gather(*tasks)

    processed = []
    success_count = 0

    for (idx, job_data), description in zip(batch, results):
        if description:
            processed.append(
                (
                    job_data["job_id"],
                    job_data["job_title"],
                    description,
                )
            )
            success_count += 1
        else:
            # Use original description as fallback
            if job_data["job_description"]:
                processed.append(
                    (
                        job_data["job_id"],
                        job_data["job_title"],
                        job_data["job_description"],
                    )
                )

    print(
        f"  [OK] Batch {batch_num}: {success_count}/{len(batch)} synthesized successfully"
    )
    return processed


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate job description dataset using LLM synthesis."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="datasets/Job/job_descriptions.csv",
        help="Path to source CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/Job/job_title_descriptions.csv",
        help="Path for output CSV file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Maximum number of rows to process (default: 2000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of jobs to process concurrently (default: 50)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of rows to skip from the beginning",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    csv_path = project_root / args.csv
    output_path = project_root / args.output

    # Validate source file
    if not csv_path.exists():
        print(f"Error: Source CSV not found at {csv_path}")
        return

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize LLM
    if not settings.GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not set in environment")
        return

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.3,
    )

    # Load source data
    print(f"Loading jobs from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} rows in source CSV")

    # Apply skip if specified
    if args.skip > 0:
        df = df.iloc[args.skip :]
        print(f"Skipped first {args.skip} rows")

    # Apply limit if specified
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df)} rows")

    # Extract job data
    jobs_to_process = []
    for idx, row in df.iterrows():
        job_data = extract_job_data(row)
        if job_data["job_id"] and job_data["job_title"]:
            jobs_to_process.append((idx, job_data))

    print(f"Jobs to process: {len(jobs_to_process)}")
    print(f"Batch size: {args.batch_size}")

    # Split into batches
    batches = [
        jobs_to_process[i : i + args.batch_size]
        for i in range(0, len(jobs_to_process), args.batch_size)
    ]
    total_batches = len(batches)

    # Process all batches
    all_results = []
    for batch_num, batch in enumerate(batches, 1):
        results = await process_batch(llm, batch, batch_num, total_batches)
        all_results.extend(results)

        # Small delay between batches to avoid rate limiting
        if batch_num < total_batches:
            await asyncio.sleep(1)

    # Write output CSV
    print(f"\nWriting {len(all_results)} rows to {output_path}...")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["job_id", "job_title", "job_description"])
        for job_id, job_title, job_description in all_results:
            writer.writerow([job_id, job_title, job_description])

    # Summary
    print(f"\n{'='*60}")
    print("Dataset generation complete!")
    print(f"Rows processed: {len(jobs_to_process)}")
    print(f"Rows written: {len(all_results)}")
    print(f"Output file: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
