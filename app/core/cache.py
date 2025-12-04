"""
Redis cache manager with decorators for easy integration.

Provides caching functionality for expensive operations like:
- Resume parsing (LLM calls)
- Job parsing (LLM calls)
- Recommendation results
- Embeddings
"""

import hashlib
import logging
import pickle
from functools import wraps
import inspect
from typing import Any, Callable, Optional, Union

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger(__name__)


class CacheManager:

    def __init__(self):
        """Initialize Redis connections."""
        self._client: Optional[redis.Redis] = None

    @property
    async def client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = await redis.from_url(
                settings.REDIS_URL,
                decode_responses=False,
            )
        return self._client

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            client = await self.client
            value = await client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for no expiration)

        Returns:
            True if successful, False otherwise
        """
        try:
            client = await self.client
            serialized = pickle.dumps(value)
            if ttl:
                await client.setex(key, ttl, serialized)
            else:
                await client.set(key, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if successful, False otherwise
        """
        try:
            client = await self.client
            await client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "resume_parsed:*")

        Returns:
            Number of keys deleted
        """
        try:
            client = await self.client
            keys = await client.keys(pattern)
            if keys:
                return await client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache clear pattern error for {pattern}: {e}")
            return 0

    async def close(self) -> None:
        """Close the Redis client and release resources."""
        if not self._client:
            return

        try:
            # Most redis.asyncio clients provide close() which may be a
            # coroutine or a regular function. Call it and await if needed.
            res = self._client.close()
            if inspect.isawaitable(res):
                await res

            # If client exposes wait_closed, await it to ensure teardown.
            if hasattr(self._client, "wait_closed"):
                wait_closed = getattr(self._client, "wait_closed")
                if inspect.iscoroutinefunction(wait_closed):
                    await wait_closed()
                else:
                    maybe = wait_closed()
                    if inspect.isawaitable(maybe):
                        await maybe
        except Exception as e:
            logger.debug(f"Error closing redis client: {e}")
        finally:
            self._client = None


# Global cache manager instance
cache_manager = CacheManager()


def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Args:
        prefix: Key prefix
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Cache key string
    """
    # Create a deterministic string from args and kwargs
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_string = ":".join(key_parts)

    # Hash the key string for consistent length
    key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]

    return f"{prefix}:{key_hash}"


def generate_hash(content: Union[str, bytes]) -> str:
    """
    Generate hash for content (useful for PDF hashing, text hashing).

    Args:
        content: Content to hash

    Returns:
        SHA256 hash string
    """
    if isinstance(content, str):
        content = content.encode()
    return hashlib.sha256(content).hexdigest()


def cache_result(
    ttl: Optional[int] = None,
    key_prefix: str = "cache",
    key_func: Optional[Callable] = None,
):
    """
    Decorator to cache function results.

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        key_func: Optional function to generate cache key from args

    Example:
        @cache_result(ttl=3600, key_prefix="resume_parsed")
        async def parse_resume_async(text: str):
            return await expensive_operation(text)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = generate_cache_key(key_prefix, *args, **kwargs)

            # Try to get from cache
            cached_value = await cache_manager.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache HIT for {cache_key}")
                return cached_value

            # Cache miss - execute function
            logger.debug(f"Cache MISS for {cache_key}")
            result = await func(*args, **kwargs)

            # Store in cache
            await cache_manager.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator
