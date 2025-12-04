"""
Celery application configuration for async task processing.
"""

from celery import Celery

from app.config import settings, configure_logging

# Ensure centralized logging is configured for CLI-launched workers
configure_logging()

celery_app = Celery(
    "jobfinder",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND or settings.REDIS_URL,
    include=[
        "app.workers.tasks.job_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=None,
    task_soft_time_limit=None,
    result_expires=3600,
    result_backend_transport_options={"master_name": "mymaster"},
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_routes={
        "app.workers.tasks.job_tasks.*": {"queue": "job_queue"},
    },
    task_default_rate_limit="100/m",
)

if settings.CELERY_RESULT_BACKEND:
    celery_app.conf.result_backend = settings.CELERY_RESULT_BACKEND
celery_app.conf.beat_schedule = {
    # Example: Clean up old cache entries every day
    # "cleanup-cache": {
    #     "task": "app.workers.tasks.maintenance.cleanup_cache",
    #     "schedule": crontab(hour=2, minute=0),  # Run at 2 AM daily
    # },
}
