"""
Performance metrics tracking system.

Tracks:
- CV/Resume parsing time
- Recommendation generation time
- Database query performance
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Performance metrics tracker using Redis for storage.

    Stores timing metrics in Redis sorted sets with percentile calculations.
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self._client: Optional[redis.Redis] = None

    @property
    async def client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = await redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
            )
        return self._client

    async def record_timing(self, metric_name: str, duration_ms: float) -> None:
        """
        Record timing metric (asynchronous).

        Args:
            metric_name: Name of the metric (e.g., "resume_parsing", "recommendation_generation")
            duration_ms: Duration in milliseconds
        """
        try:
            client = await self.client
            timestamp = datetime.utcnow().isoformat()
            key = f"metrics:timing:{metric_name}"

            # Store in sorted set with timestamp as score for ordering
            await client.zadd(key, {f"{timestamp}:{duration_ms}": time.time()})

            # Keep only last 1000 entries to prevent unbounded growth
            await client.zremrangebyrank(key, 0, -1001)

            # Update total count for this metric
            await client.incr(f"metrics:count:{metric_name}")

            # Update running sum for average calculation
            await client.incrbyfloat(f"metrics:sum:{metric_name}", duration_ms)

            logger.debug(f"Recorded timing metric {metric_name}: {duration_ms:.2f}ms")

        except Exception as e:
            logger.warning(f"Failed to record timing metric {metric_name}: {e}")

    async def get_timing_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get timing statistics for a metric (asynchronous).

        Args:
            metric_name: Name of the metric

        Returns:
            Dictionary with count, avg, min, max values
        """
        try:
            client = await self.client
            key = f"metrics:timing:{metric_name}"
            values = await client.zrange(key, 0, -1)

            if not values:
                return {
                    "count": 0,
                    "avg": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                }

            # Extract durations from "timestamp:duration" format
            durations = []
            for v in values:
                try:
                    # Handle format "2024-01-01T12:00:00.000000:123.45"
                    parts = v.rsplit(":", 1)
                    if len(parts) == 2:
                        durations.append(float(parts[-1]))
                except (ValueError, IndexError):
                    continue

            if not durations:
                return {
                    "count": 0,
                    "avg": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                }

            durations.sort()
            count = len(durations)
            avg = sum(durations) / count
            min_val = durations[0]
            max_val = durations[-1]

            return {
                "count": count,
                "avg": round(avg, 2),
                "min": round(min_val, 2),
                "max": round(max_val, 2),
            }

        except Exception as e:
            logger.warning(f"Failed to get timing stats for {metric_name}: {e}")
            return {
                "count": 0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

    async def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics in a structured format.

        Returns:
            Dictionary containing timing stats
        """
        try:
            client = await self.client
            result = {
                "timing": {},
            }

            # Get all timing metrics
            timing_keys = await client.keys("metrics:timing:*")
            for key in timing_keys:
                metric_name = key.replace("metrics:timing:", "")
                result["timing"][metric_name] = await self.get_timing_stats(metric_name)

            return result

        except Exception as e:
            logger.warning(f"Failed to get all metrics: {e}")
            return {"timing": {}}

    async def reset_metrics(self, pattern: str = "metrics:*") -> int:
        """
        Reset all metrics matching pattern (asynchronous).

        Args:
            pattern: Key pattern to reset

        Returns:
            Number of keys deleted
        """
        try:
            client = await self.client
            keys = await client.keys(pattern)
            if keys:
                deleted = await client.delete(*keys)
                logger.info(f"Reset {deleted} metrics keys")
                return deleted
            return 0
        except Exception as e:
            logger.warning(f"Failed to reset metrics: {e}")
            return 0

    async def close(self) -> None:
        """Close the Redis client and release resources."""
        if self._client is None:
            return

        try:
            await self._client.close()
        except Exception as e:
            logger.debug(f"Error closing metrics redis client: {e}")
        finally:
            self._client = None


# Global metrics tracker instance
metrics_tracker = MetricsTracker()


class TimingContext:
    """
    Context manager for tracking timing of code blocks.

    Example:
        async with TimingContext("database_query"):
            result = await db.execute(query)
    """

    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.start_time: Optional[float] = None

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            await metrics_tracker.record_timing(self.metric_name, duration_ms)
            logger.debug(f"{self.metric_name} completed in {duration_ms:.2f}ms")
        return False
