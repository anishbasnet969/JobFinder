"""
Script to parse job descriptions from CSV and save as JSON files.

Reads job data from datasets/Job/job_title_des.csv, parses each job description
using LLM, and stores the parsed JSON in the parsed_jobs directory.

Supports concurrent processing of multiple jobs using asyncio.
"""

from pathlib import Path
import argparse
import sys
import os
import asyncio
import aiofiles
import pandas as pd
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.job.job_parser import parse_job_description


async def process_job(
    job_title: str,
    job_description: str,
    job_index: int,
    output_dir: Path,
) -> bool:
    """
    Process a single job description and save as JSON.

    Args:
        job_title: The job title
        job_description: The job description text
        job_index: Index of the job (used for filename)
        output_dir: Directory to save the JSON file

    Returns:
        True if successful, False otherwise
    """
    print(f"[{job_index}] Processing: {job_title[:50]}...")

    try:
        # Parse job description using LLM
        parsed_job = await parse_job_description(job_title, job_description)

        # Create output filename
        # Use index for unique filename since titles may repeat
        output_path = output_dir / f"job_{job_index}.json"

        # Save to JSON file asynchronously
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(parsed_job.model_dump_json(by_alias=True, indent=2))

        print(f"  ✓ Saved to {output_path}")
        return True

    except Exception as e:
        print(f"  ✗ Failed to parse job [{job_index}]: {e}")
        return False


async def process_jobs_batch(
    jobs: list[tuple[int, str, str]],
    output_dir: Path,
    batch_num: int,
    total_batches: int,
) -> tuple[int, int]:
    """
    Process a batch of jobs concurrently.

    Args:
        jobs: List of tuples (index, title, description)
        output_dir: Directory to save the JSON files
        batch_num: Current batch number (for logging)
        total_batches: Total number of batches (for logging)

    Returns:
        Tuple of (success_count, error_count)
    """
    print(
        f"\n[Batch {batch_num}/{total_batches}] Processing {len(jobs)} jobs concurrently..."
    )

    tasks = [
        process_job(title, description, idx, output_dir)
        for idx, title, description in jobs
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = sum(1 for r in results if r is True)
    error_count = len(results) - success_count

    return success_count, error_count


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parse job descriptions from CSV to JSON files."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="datasets/Job/job_title_des.csv",
        help="Path to CSV file containing job data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="parsed_jobs",
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of jobs to process",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of jobs to skip from the beginning",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing from specific row index (default: 0)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=500,
        help="Stop processing at specific row index (exclusive, default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of jobs to process concurrently in each batch (default: 100)",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    csv_path = project_root / args.csv
    output_dir = project_root / args.output

    # Validate CSV file exists
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Read CSV file
    print(f"Loading jobs from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} jobs in CSV")

    # Apply index-based filtering if specified
    if args.start_index is not None or args.end_index is not None:
        start = args.start_index if args.start_index is not None else 0
        end = args.end_index if args.end_index is not None else len(df)
        df = df.iloc[start:end]
        print(f"Processing rows {start} to {end} ({len(df)} jobs)")
    else:
        # Apply skip and limit
        if args.skip > 0:
            df = df.iloc[args.skip :]
            print(f"Skipping first {args.skip} jobs")

        if args.limit:
            df = df.head(args.limit)
            print(f"Processing limit of {args.limit} jobs")

    # Collect jobs to process
    jobs_to_process = []

    for idx, row in df.iterrows():
        job_title = str(row.get("Job Title", ""))
        job_description = str(row.get("Job Description", ""))

        # Skip if either field is empty
        if (
            not job_title
            or not job_description
            or job_title == "nan"
            or job_description == "nan"
        ):
            print(f"[{idx}] Skipping empty job")
            continue

        jobs_to_process.append((idx, job_title, job_description))

    print(f"Jobs to process: {len(jobs_to_process)}")
    print(f"Batch size: {args.batch_size}")

    # Split into batches
    batches = [
        jobs_to_process[i : i + args.batch_size]
        for i in range(0, len(jobs_to_process), args.batch_size)
    ]
    total_batches = len(batches)

    # Process batches sequentially, jobs within each batch concurrently
    total_success = 0
    total_errors = 0

    for batch_num, batch in enumerate(batches, 1):
        success_count, error_count = await process_jobs_batch(
            batch, output_dir, batch_num, total_batches
        )
        total_success += success_count
        total_errors += error_count

    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully parsed: {total_success} jobs")
    print(f"Errors: {total_errors} jobs")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
