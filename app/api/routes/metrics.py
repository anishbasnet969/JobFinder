"""
API routes for performance metrics endpoints.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from app.core.metrics import metrics_tracker
from app.schemas.metrics import MetricDetailResponse, MetricsResponse, TimingStats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("", response_model=MetricsResponse)
async def get_all_metrics():
    """
    Get all performance metrics.

    Returns timing statistics for:
    - CV parsing (resume_parsing)
    - Recommendation generation (recommendation_generation)
    - Database queries (database_vector_search)
    """
    try:
        metrics = await metrics_tracker.get_all_metrics()

        # Convert timing metrics to TimingStats objects
        timing_stats = {}
        for name, stats in metrics.get("timing", {}).items():
            timing_stats[name] = TimingStats(**stats)

        return MetricsResponse(timing=timing_stats)

    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {str(e)}")


@router.get("/timing/{metric_name}", response_model=MetricDetailResponse)
async def get_timing_metric(metric_name: str):
    """
    Get detailed timing statistics for a specific metric.

    Args:
        metric_name: Name of the metric (e.g., "resume_parsing", "recommendation_generation", "database_vector_search")

    Returns:
        Detailed timing statistics including percentiles
    """
    try:
        stats = await metrics_tracker.get_timing_stats(metric_name)

        if stats.get("count", 0) == 0:
            raise HTTPException(
                status_code=404, detail=f"No data found for metric: {metric_name}"
            )

        return MetricDetailResponse(
            metric_name=metric_name,
            stats=TimingStats(**stats),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching timing metric {metric_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching metric: {str(e)}")


@router.delete("")
async def reset_metrics(
    pattern: str = Query(
        default="metrics:*",
        description="Key pattern to reset (use 'metrics:*' for all metrics)",
    )
) -> Dict[str, Any]:
    """
    Reset metrics matching the specified pattern.

    WARNING: This will delete all matching metrics data.

    Args:
        pattern: Redis key pattern to reset

    Returns:
        Number of keys deleted
    """
    try:
        deleted = await metrics_tracker.reset_metrics(pattern)
        logger.info(f"Reset {deleted} metrics keys matching pattern: {pattern}")
        return {
            "deleted_keys": deleted,
            "pattern": pattern,
        }

    except Exception as e:
        logger.error(f"Error resetting metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error resetting metrics: {str(e)}"
        )


@router.get("/summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Get a summary of key performance metrics.

    Returns a simplified view of:
    - CV parsing time (avg)
    - Recommendation generation time (avg)
    - Database query performance (avg)
    """
    try:
        metrics = await metrics_tracker.get_all_metrics()

        # Get key timing metrics
        resume_parsing = metrics.get("timing", {}).get("resume_parsing", {})
        recommendation = metrics.get("timing", {}).get("recommendation_generation", {})
        db_query = metrics.get("timing", {}).get("database_vector_search", {})

        return {
            "cv_parsing": {
                "avg_ms": resume_parsing.get("avg", 0),
                "count": resume_parsing.get("count", 0),
            },
            "recommendation_generation": {
                "avg_ms": recommendation.get("avg", 0),
                "count": recommendation.get("count", 0),
            },
            "database_query": {
                "avg_ms": db_query.get("avg", 0),
                "count": db_query.get("count", 0),
            },
        }

    except Exception as e:
        logger.error(f"Error fetching metrics summary: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching metrics summary: {str(e)}"
        )
