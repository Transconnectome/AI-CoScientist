"""Unit tests for query cache optimization."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.services.knowledge_base.query_cache import QueryCache
from src.services.knowledge_base.search import SearchResult


@pytest.fixture
def query_cache():
    """Create query cache fixture."""
    return QueryCache(ttl_seconds=60, max_cache_size=100)


@pytest.fixture
def sample_search_results():
    """Create sample search results."""
    return [
        SearchResult(
            document_id="doc1",
            title="Test Paper 1",
            abstract="This is a test abstract about machine learning",
            score=0.95,
            metadata={"year": 2023},
            highlights=["machine learning"]
        ),
        SearchResult(
            document_id="doc2",
            title="Test Paper 2",
            abstract="Another paper about deep learning",
            score=0.85,
            metadata={"year": 2023},
            highlights=["deep learning"]
        )
    ]


class TestQueryCache:
    """Test cases for QueryCache."""

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, query_cache):
        """Test cache key generation is deterministic."""
        key1 = query_cache._generate_cache_key(
            "machine learning",
            top_k=10,
            filters=None,
            search_type="semantic"
        )

        key2 = query_cache._generate_cache_key(
            "machine learning",
            top_k=10,
            filters=None,
            search_type="semantic"
        )

        assert key1 == key2

    @pytest.mark.asyncio
    async def test_cache_key_differs_for_different_queries(self, query_cache):
        """Test cache keys differ for different queries."""
        key1 = query_cache._generate_cache_key(
            "machine learning",
            top_k=10,
            filters=None,
            search_type="semantic"
        )

        key2 = query_cache._generate_cache_key(
            "deep learning",
            top_k=10,
            filters=None,
            search_type="semantic"
        )

        assert key1 != key2

    @pytest.mark.asyncio
    async def test_cache_miss(self, query_cache):
        """Test cache miss returns None."""
        result = await query_cache.get(
            query="machine learning",
            top_k=10,
            filters=None,
            search_type="semantic"
        )

        assert result is None
        assert query_cache._cache_misses == 1
        assert query_cache._cache_hits == 0

    @pytest.mark.asyncio
    @patch('src.services.knowledge_base.query_cache.get_redis_client')
    async def test_cache_set_and_get(self, mock_redis, query_cache, sample_search_results):
        """Test setting and getting from cache."""
        # Mock Redis
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = None
        mock_redis_instance.set.return_value = True
        mock_redis.return_value = mock_redis_instance

        # Set cache
        await query_cache.set(
            query="machine learning",
            top_k=10,
            results=sample_search_results,
            filters=None,
            search_type="semantic"
        )

        # Get from cache (should hit local cache)
        result = await query_cache.get(
            query="machine learning",
            top_k=10,
            filters=None,
            search_type="semantic"
        )

        assert result is not None
        assert len(result) == 2
        assert result[0].document_id == "doc1"
        assert query_cache._cache_hits == 1

    @pytest.mark.asyncio
    async def test_cache_local_storage(self, query_cache, sample_search_results):
        """Test local cache storage works without Redis."""
        # Set cache (will store locally even if Redis fails)
        with patch('src.services.knowledge_base.query_cache.get_redis_client', side_effect=Exception("Redis error")):
            await query_cache.set(
                query="machine learning",
                top_k=10,
                results=sample_search_results,
                filters=None,
                search_type="semantic"
            )

        # Get from local cache
        result = await query_cache.get(
            query="machine learning",
            top_k=10,
            filters=None,
            search_type="semantic"
        )

        assert result is not None
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, sample_search_results):
        """Test cache entries expire after TTL."""
        # Create cache with 1-second TTL
        cache = QueryCache(ttl_seconds=1)

        # Set cache
        await cache.set(
            query="machine learning",
            top_k=10,
            results=sample_search_results,
            filters=None,
            search_type="semantic"
        )

        # Immediate get should hit
        result = await cache.get(
            query="machine learning",
            top_k=10,
            filters=None,
            search_type="semantic"
        )
        assert result is not None

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should miss after expiration
        result = await cache.get(
            query="machine learning",
            top_k=10,
            filters=None,
            search_type="semantic"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_size_management(self, sample_search_results):
        """Test local cache size management."""
        cache = QueryCache(max_cache_size=5)

        # Add more entries than max_cache_size
        for i in range(10):
            await cache.set(
                query=f"query_{i}",
                top_k=10,
                results=sample_search_results,
                filters=None,
                search_type="semantic"
            )

        # Cache should not exceed max size
        assert len(cache._local_cache) <= 5

    @pytest.mark.asyncio
    async def test_cache_stats(self, query_cache, sample_search_results):
        """Test cache statistics tracking."""
        # Set cache
        await query_cache.set(
            query="machine learning",
            top_k=10,
            results=sample_search_results,
            filters=None,
            search_type="semantic"
        )

        # One hit, one miss
        await query_cache.get("machine learning", 10, None, "semantic")  # Hit
        await query_cache.get("deep learning", 10, None, "semantic")  # Miss

        stats = query_cache.get_stats()

        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 0.5
        assert stats["local_cache_size"] == 1

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, query_cache, sample_search_results):
        """Test cache invalidation."""
        # Set cache
        await query_cache.set(
            query="machine learning",
            top_k=10,
            results=sample_search_results,
            filters=None,
            search_type="semantic"
        )

        # Verify cached
        result = await query_cache.get("machine learning", 10, None, "semantic")
        assert result is not None

        # Invalidate all
        count = await query_cache.invalidate()
        assert count >= 1

        # Should miss after invalidation
        result = await query_cache.get("machine learning", 10, None, "semantic")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_with_filters(self, query_cache, sample_search_results):
        """Test cache respects filters."""
        filters1 = {"year_min": 2020}
        filters2 = {"year_min": 2023}

        # Set with filter 1
        await query_cache.set(
            query="machine learning",
            top_k=10,
            results=sample_search_results,
            filters=filters1,
            search_type="semantic"
        )

        # Get with same filter should hit
        result = await query_cache.get(
            "machine learning", 10, filters1, "semantic"
        )
        assert result is not None

        # Get with different filter should miss
        result = await query_cache.get(
            "machine learning", 10, filters2, "semantic"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_stats_reset(self, query_cache, sample_search_results):
        """Test resetting cache statistics."""
        # Generate some stats
        await query_cache.set("query", 10, sample_search_results, None, "semantic")
        await query_cache.get("query", 10, None, "semantic")

        assert query_cache._cache_hits > 0

        # Reset stats
        query_cache.reset_stats()

        assert query_cache._cache_hits == 0
        assert query_cache._cache_misses == 0
