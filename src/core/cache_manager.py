"""Advanced caching manager with multi-level caching strategy."""

import json
import hashlib
from typing import Any, Optional, Union
from functools import wraps

from redis.asyncio import Redis
import structlog

logger = structlog.get_logger(__name__)


class CacheManager:
    """Multi-level cache manager with Redis and in-memory caching."""

    def __init__(self, redis_client: Redis, default_ttl: int = 3600):
        """Initialize cache manager.

        Args:
            redis_client: Redis client instance
            default_ttl: Default TTL in seconds
        """
        self.redis = redis_client
        self.default_ttl = default_ttl
        self._memory_cache: dict[str, tuple[Any, float]] = {}
        self._max_memory_items = 1000

    def _generate_key(self, namespace: str, **kwargs: Any) -> str:
        """Generate cache key from namespace and parameters.

        Args:
            namespace: Cache namespace
            **kwargs: Key-value pairs for cache key

        Returns:
            Generated cache key
        """
        # Sort kwargs for consistent key generation
        sorted_items = sorted(kwargs.items())
        key_data = f"{namespace}:" + ":".join(
            f"{k}={v}" for k, v in sorted_items
        )

        # Hash if key is too long
        if len(key_data) > 200:
            key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
            return f"{namespace}:hash:{key_hash}"

        return key_data

    async def get(
        self,
        namespace: str,
        use_memory: bool = True,
        **kwargs: Any
    ) -> Optional[Any]:
        """Get value from cache.

        Args:
            namespace: Cache namespace
            use_memory: Whether to use memory cache
            **kwargs: Cache key parameters

        Returns:
            Cached value or None
        """
        key = self._generate_key(namespace, **kwargs)

        # Try memory cache first
        if use_memory and key in self._memory_cache:
            value, _ = self._memory_cache[key]
            logger.debug("memory_cache_hit", key=key)
            return value

        # Try Redis
        try:
            cached = await self.redis.get(key)
            if cached:
                value = json.loads(cached)
                logger.debug("redis_cache_hit", key=key)

                # Store in memory cache
                if use_memory:
                    self._memory_cache[key] = (value, 0)
                    self._cleanup_memory_cache()

                return value
        except Exception as e:
            logger.error("cache_get_error", key=key, error=str(e))

        logger.debug("cache_miss", key=key)
        return None

    async def set(
        self,
        namespace: str,
        value: Any,
        ttl: Optional[int] = None,
        use_memory: bool = True,
        **kwargs: Any
    ) -> bool:
        """Set value in cache.

        Args:
            namespace: Cache namespace
            value: Value to cache
            ttl: Time to live in seconds
            use_memory: Whether to use memory cache
            **kwargs: Cache key parameters

        Returns:
            True if successful
        """
        key = self._generate_key(namespace, **kwargs)
        ttl = ttl or self.default_ttl

        try:
            # Store in Redis
            serialized = json.dumps(value)
            await self.redis.setex(key, ttl, serialized)

            # Store in memory cache
            if use_memory:
                self._memory_cache[key] = (value, 0)
                self._cleanup_memory_cache()

            logger.debug("cache_set", key=key, ttl=ttl)
            return True

        except Exception as e:
            logger.error("cache_set_error", key=key, error=str(e))
            return False

    async def delete(self, namespace: str, **kwargs: Any) -> bool:
        """Delete value from cache.

        Args:
            namespace: Cache namespace
            **kwargs: Cache key parameters

        Returns:
            True if successful
        """
        key = self._generate_key(namespace, **kwargs)

        try:
            # Delete from Redis
            await self.redis.delete(key)

            # Delete from memory cache
            self._memory_cache.pop(key, None)

            logger.debug("cache_delete", key=key)
            return True

        except Exception as e:
            logger.error("cache_delete_error", key=key, error=str(e))
            return False

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace.

        Args:
            namespace: Cache namespace to clear

        Returns:
            Number of keys deleted
        """
        pattern = f"{namespace}:*"

        try:
            # Get all keys matching pattern
            keys = []
            async for key in self.redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info(
                    "namespace_cleared",
                    namespace=namespace,
                    count=deleted
                )
                return deleted

            return 0

        except Exception as e:
            logger.error(
                "clear_namespace_error",
                namespace=namespace,
                error=str(e)
            )
            return 0

    def _cleanup_memory_cache(self):
        """Clean up memory cache if it exceeds max items."""
        if len(self._memory_cache) > self._max_memory_items:
            # Remove oldest 20% of items
            items_to_remove = len(self._memory_cache) - int(
                self._max_memory_items * 0.8
            )

            for key in list(self._memory_cache.keys())[:items_to_remove]:
                del self._memory_cache[key]

            logger.debug(
                "memory_cache_cleanup",
                removed=items_to_remove
            )


def cached(
    namespace: str,
    ttl: Optional[int] = None,
    key_params: Optional[list[str]] = None
):
    """Decorator for caching function results.

    Args:
        namespace: Cache namespace
        ttl: Time to live in seconds
        key_params: List of parameter names to use in cache key

    Example:
        @cached("user_data", ttl=3600, key_params=["user_id"])
        async def get_user(user_id: str):
            return await fetch_user(user_id)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get cache manager from self
            if not hasattr(self, 'cache_manager'):
                # No cache manager, execute function
                return await func(self, *args, **kwargs)

            cache_manager: CacheManager = self.cache_manager

            # Build cache key from parameters
            cache_kwargs = {}
            if key_params:
                # Get function signature
                import inspect
                sig = inspect.signature(func)
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()

                for param in key_params:
                    if param in bound.arguments:
                        cache_kwargs[param] = str(bound.arguments[param])

            # Try to get from cache
            cached_value = await cache_manager.get(
                namespace,
                **cache_kwargs
            )

            if cached_value is not None:
                return cached_value

            # Execute function
            result = await func(self, *args, **kwargs)

            # Cache result
            await cache_manager.set(
                namespace,
                result,
                ttl=ttl,
                **cache_kwargs
            )

            return result

        return wrapper
    return decorator
