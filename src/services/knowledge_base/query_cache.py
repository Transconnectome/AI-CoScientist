"""Query result caching service for RAG performance optimization."""

import hashlib
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
from dataclasses import asdict

from src.core.redis import get_redis_client
from src.services.knowledge_base.search import SearchResult


class QueryCache:
    """Cache layer for RAG query results."""

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_cache_size: int = 10000,
        prefix: str = "rag_query"
    ):
        """Initialize query cache.

        Args:
            ttl_seconds: Time-to-live for cached results
            max_cache_size: Maximum number of entries in cache
            prefix: Redis key prefix for cache entries
        """
        self.ttl_seconds = ttl_seconds
        self.max_cache_size = max_cache_size
        self.prefix = prefix
        self._local_cache: Dict[str, tuple[List[SearchResult], float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    async def get(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "semantic"
    ) -> Optional[List[SearchResult]]:
        """Get cached query results.

        Args:
            query: Search query
            top_k: Number of results
            filters: Optional metadata filters
            search_type: Type of search (semantic, keyword, hybrid)

        Returns:
            Cached search results if available, None otherwise
        """
        cache_key = self._generate_cache_key(query, top_k, filters, search_type)

        # Check local cache first (in-memory)
        if cache_key in self._local_cache:
            results, timestamp = self._local_cache[cache_key]
            if datetime.now().timestamp() - timestamp < self.ttl_seconds:
                self._cache_hits += 1
                return results
            else:
                # Expired, remove from local cache
                del self._local_cache[cache_key]

        # Check Redis cache
        try:
            redis = await get_redis_client()
            cached_data = await redis.get(cache_key)

            if cached_data:
                results_dict = json.loads(cached_data)
                results = [
                    SearchResult(**result_data)
                    for result_data in results_dict
                ]

                # Store in local cache for faster subsequent access
                self._local_cache[cache_key] = (results, datetime.now().timestamp())
                self._manage_local_cache_size()

                self._cache_hits += 1
                return results
        except Exception as e:
            # If Redis fails, log but don't crash
            print(f"Redis cache read error: {e}")

        self._cache_misses += 1
        return None

    async def set(
        self,
        query: str,
        top_k: int,
        results: List[SearchResult],
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "semantic"
    ) -> None:
        """Cache query results.

        Args:
            query: Search query
            top_k: Number of results
            results: Search results to cache
            filters: Optional metadata filters
            search_type: Type of search (semantic, keyword, hybrid)
        """
        cache_key = self._generate_cache_key(query, top_k, filters, search_type)

        # Store in local cache
        self._local_cache[cache_key] = (results, datetime.now().timestamp())
        self._manage_local_cache_size()

        # Store in Redis cache
        try:
            redis = await get_redis_client()
            results_dict = [asdict(result) for result in results]
            await redis.set(
                cache_key,
                json.dumps(results_dict, default=str),
                ex=self.ttl_seconds
            )
        except Exception as e:
            # If Redis fails, log but don't crash
            print(f"Redis cache write error: {e}")

    async def invalidate(
        self,
        query: Optional[str] = None,
        search_type: Optional[str] = None
    ) -> int:
        """Invalidate cache entries.

        Args:
            query: Specific query to invalidate (None = invalidate all)
            search_type: Specific search type to invalidate

        Returns:
            Number of entries invalidated
        """
        count = 0

        if query is None:
            # Clear all cache
            self._local_cache.clear()

            try:
                redis = await get_redis_client()
                # Delete all keys matching prefix
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor,
                        match=f"{self.prefix}:*",
                        count=100
                    )
                    if keys:
                        await redis.delete(*keys)
                        count += len(keys)
                    if cursor == 0:
                        break
            except Exception as e:
                print(f"Redis cache invalidation error: {e}")
        else:
            # Invalidate specific query
            # Need to iterate through all variations of top_k and filters
            # For simplicity, clear local cache entries matching query
            keys_to_remove = [
                k for k in self._local_cache.keys()
                if query in k
            ]
            for key in keys_to_remove:
                del self._local_cache[key]
                count += 1

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            self._cache_hits / total_requests
            if total_requests > 0
            else 0.0
        )

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "local_cache_size": len(self._local_cache),
            "max_cache_size": self.max_cache_size,
            "ttl_seconds": self.ttl_seconds
        }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._cache_hits = 0
        self._cache_misses = 0

    def _generate_cache_key(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        search_type: str
    ) -> str:
        """Generate cache key from query parameters.

        Args:
            query: Search query
            top_k: Number of results
            filters: Optional metadata filters
            search_type: Type of search

        Returns:
            Cache key string
        """
        # Create deterministic representation
        key_data = {
            "query": query.lower().strip(),
            "top_k": top_k,
            "filters": filters or {},
            "search_type": search_type
        }

        # Generate hash
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]

        return f"{self.prefix}:{search_type}:{key_hash}"

    def _manage_local_cache_size(self) -> None:
        """Manage local cache size by removing oldest entries."""
        if len(self._local_cache) > self.max_cache_size:
            # Remove oldest 10% of entries
            remove_count = self.max_cache_size // 10
            sorted_items = sorted(
                self._local_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            for key, _ in sorted_items[:remove_count]:
                del self._local_cache[key]
