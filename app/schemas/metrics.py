"""
Schemas for performance metrics.
"""

from typing import Dict

from pydantic import Field

from app.schemas.base import CustomBaseModel


class TimingStats(CustomBaseModel):
    """Timing statistics for a metric."""

    count: int = Field(description="Total number of recorded measurements")
    avg: float = Field(description="Average duration in milliseconds")
    min: float = Field(description="Minimum duration in milliseconds")
    max: float = Field(description="Maximum duration in milliseconds")


class MetricsResponse(CustomBaseModel):
    """Complete metrics response."""

    timing: Dict[str, TimingStats] = Field(
        description="Timing metrics for various operations"
    )


class MetricDetailResponse(CustomBaseModel):
    """Detailed response for a single metric."""

    metric_name: str
    stats: TimingStats
