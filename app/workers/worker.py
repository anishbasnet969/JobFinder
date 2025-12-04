"""
Worker entry point for running Celery workers.

Usage:
    python -m app.workers.worker

Or with Celery CLI:
    celery -A app.core.celery_app worker --loglevel=info -Q job_queue
"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.celery_app import celery_app

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting JobFinder Celery worker...")
    logger.info(f"Broker: {celery_app.conf.broker_url}")
    logger.info(f"Backend: {celery_app.conf.result_backend}")
    logger.info("Queues: job_queue")

    celery_app.worker_main(
        [
            "worker",
            "--loglevel=info",
            "--queues=job_queue",
            "--concurrency=4",
            "--max-tasks-per-child=100",
        ]
    )


if __name__ == "__main__":
    main()
