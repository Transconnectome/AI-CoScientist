"""Integration tests for optimized RAG search."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from src.services.knowledge_base.search_optimized import (
    OptimizedKnowledgeBaseSearch,
    QueryPreprocessor,
    SearchMetrics
)
from src.services.knowledge_base.vector_store_optimized import OptimizedVectorStore
from src.services.knowledge_base.embedding_optimized import OptimizedEmbeddingService
from src.services.knowledge_base.query_cache import QueryCache
from src.services.knowledge_base.search import SearchResult


@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    session = AsyncMock()
    return session


@pytest.fixture
def mock_vector_store():
    """Create mock optimized vector store."""
    store = AsyncMock(spec=OptimizedVectorStore)

    # Mock query response
    store.query.return_value = {
        'ids': [['doc1', 'doc2', 'doc3']],
        'distances': [[0.1, 0.2, 0.3]],
        'metadatas': [[
            {'title': 'Paper 1', 'year': 2023},
            {'title': 'Paper 2', 'year': 2023},
            {'title': 'Paper 3', 'year': 2022}
        ]],
        'documents': [[
            'Abstract about machine learning',
            'Abstract about deep learning',
            'Abstract about neural networks'
        ]]
    }

    return store


@pytest.fixture
def mock_embedding_service():
    """Create mock optimized embedding service."""
    service = AsyncMock(spec=OptimizedEmbeddingService)

    # Mock embedding generation
    async def mock_encode_async(text, use_cache=True):
        return np.random.rand(384)

    service.encode_async = mock_encode_async

    return service


@pytest.fixture
def query_cache():
    """Create query cache fixture."""
    return QueryCache(ttl_seconds=60)


@pytest.fixture
def optimized_search(mock_vector_store, mock_embedding_service, mock_db_session, query_cache):
    """Create optimized search service fixture."""
    return OptimizedKnowledgeBaseSearch(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        db=mock_db_session,
        cache=query_cache
    )


class TestQueryPreprocessor:
    """Test cases for QueryPreprocessor."""

    def test_normalize_query(self):
        """Test query normalization."""
        preprocessor = QueryPreprocessor()

        normalized = preprocessor.normalize("  Machine   Learning  ")
        assert normalized == "machine learning"

        normalized = preprocessor.normalize("Test@Query#Here!")
        assert normalized == "testqueryhere"

    def test_remove_stopwords(self):
        """Test stopword removal."""
        preprocessor = QueryPreprocessor()

        filtered = preprocessor.remove_stopwords("the machine learning is")
        assert "the" not in filtered
        assert "is" not in filtered
        assert "machine" in filtered
        assert "learning" in filtered

    def test_expand_query(self):
        """Test query expansion."""
        preprocessor = QueryPreprocessor()

        variations = preprocessor.expand_query("the machine learning algorithm")
        assert len(variations) >= 1
        assert "the machine learning algorithm" in variations


class TestOptimizedKnowledgeBaseSearch:
    """Integration tests for optimized search."""

    @pytest.mark.asyncio
    async def test_semantic_search_no_cache(self, optimized_search):
        """Test semantic search without cache."""
        results, metrics = await optimized_search.semantic_search(
            query="machine learning",
            top_k=10,
            use_cache=False
        )

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert isinstance(metrics, SearchMetrics)
        assert metrics.cache_hit is False
        assert metrics.total_results == 3
        assert metrics.query_time_ms > 0

    @pytest.mark.asyncio
    async def test_semantic_search_with_cache(self, optimized_search):
        """Test semantic search with caching."""
        # First call - cache miss
        results1, metrics1 = await optimized_search.semantic_search(
            query="machine learning",
            top_k=10,
            use_cache=True
        )

        assert metrics1.cache_hit is False
        assert len(results1) == 3

        # Second call - cache hit
        results2, metrics2 = await optimized_search.semantic_search(
            query="machine learning",
            top_k=10,
            use_cache=True
        )

        assert metrics2.cache_hit is True
        assert len(results2) == 3
        assert metrics2.query_time_ms < metrics1.query_time_ms

    @pytest.mark.asyncio
    async def test_semantic_search_with_filters(self, optimized_search):
        """Test semantic search with metadata filters."""
        filters = {"year_min": 2023}

        results, metrics = await optimized_search.semantic_search(
            query="machine learning",
            top_k=10,
            filters=filters,
            use_cache=False
        )

        assert len(results) == 3
        assert isinstance(metrics, SearchMetrics)

    @pytest.mark.asyncio
    async def test_semantic_search_min_score_filtering(self, optimized_search, mock_vector_store):
        """Test results are filtered by minimum score."""
        # Adjust mock to return lower similarity scores
        mock_vector_store.query.return_value = {
            'ids': [['doc1', 'doc2', 'doc3']],
            'distances': [[0.1, 0.6, 0.7]],  # 0.9, 0.4, 0.3 similarity
            'metadatas': [[
                {'title': 'Paper 1'},
                {'title': 'Paper 2'},
                {'title': 'Paper 3'}
            ]],
            'documents': [['Abstract 1', 'Abstract 2', 'Abstract 3']]
        }

        results, _ = await optimized_search.semantic_search(
            query="test",
            top_k=10,
            min_score=0.5,
            use_cache=False
        )

        # Only doc1 (0.9 similarity) should pass min_score filter
        assert len(results) == 1
        assert results[0].document_id == 'doc1'

    @pytest.mark.asyncio
    async def test_keyword_search(self, optimized_search, mock_db_session):
        """Test keyword search functionality."""
        # Mock SQL query result
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            Mock(id='doc1', title='Paper 1', abstract='Machine learning abstract', rank=0.95),
            Mock(id='doc2', title='Paper 2', abstract='Deep learning abstract', rank=0.85)
        ]
        mock_db_session.execute.return_value = mock_result

        results, metrics = await optimized_search.keyword_search(
            query="machine learning",
            top_k=10,
            use_cache=False
        )

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert isinstance(metrics, SearchMetrics)

    @pytest.mark.asyncio
    async def test_hybrid_search(self, optimized_search, mock_db_session):
        """Test hybrid search combining semantic and keyword."""
        # Mock SQL query result for keyword search
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            Mock(id='doc1', title='Paper 1', abstract='Abstract 1', rank=0.95)
        ]
        mock_db_session.execute.return_value = mock_result

        results, metrics = await optimized_search.hybrid_search(
            query="machine learning",
            top_k=10,
            semantic_weight=0.7,
            use_cache=False
        )

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert isinstance(metrics, SearchMetrics)

    @pytest.mark.asyncio
    async def test_performance_improvement_with_cache(self, optimized_search):
        """Test that caching improves performance."""
        # First call - no cache
        start = time.time()
        await optimized_search.semantic_search(
            "machine learning", top_k=10, use_cache=True
        )
        first_call_time = time.time() - start

        # Second call - with cache
        start = time.time()
        await optimized_search.semantic_search(
            "machine learning", top_k=10, use_cache=True
        )
        second_call_time = time.time() - start

        # Cached call should be faster
        assert second_call_time < first_call_time

    @pytest.mark.asyncio
    async def test_get_comprehensive_stats(self, optimized_search):
        """Test getting comprehensive performance statistics."""
        # Perform some searches
        await optimized_search.semantic_search("query1", top_k=10, use_cache=True)
        await optimized_search.semantic_search("query1", top_k=10, use_cache=True)
        await optimized_search.semantic_search("query2", top_k=10, use_cache=True)

        stats = optimized_search.get_stats()

        assert "search" in stats
        assert "cache" in stats
        assert "embedding" in stats
        assert "vector_store" in stats

        assert stats["search"]["total_queries"] == 3
        assert stats["search"]["cache_hits"] == 1
        assert stats["search"]["cache_hit_rate"] > 0

    @pytest.mark.asyncio
    async def test_clear_all_caches(self, optimized_search):
        """Test clearing all caches."""
        # Perform search to populate caches
        await optimized_search.semantic_search(
            "machine learning", top_k=10, use_cache=True
        )

        # Clear caches
        result = await optimized_search.clear_all_caches()

        assert "query_cache_cleared" in result
        assert "embedding_cache_cleared" in result

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, optimized_search):
        """Test handling multiple concurrent searches."""
        tasks = [
            optimized_search.semantic_search(f"query{i}", top_k=10, use_cache=False)
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(len(r[0]) > 0 for r in results)  # All have results
        assert all(isinstance(r[1], SearchMetrics) for r in results)  # All have metrics

    @pytest.mark.asyncio
    async def test_query_normalization_consistency(self, optimized_search):
        """Test query normalization provides consistent cache hits."""
        # Different formats of same query
        queries = [
            "Machine Learning",
            "machine learning",
            "  MACHINE  LEARNING  "
        ]

        results_list = []
        for query in queries:
            results, metrics = await optimized_search.semantic_search(
                query, top_k=10, use_cache=True
            )
            results_list.append((results, metrics))

        # First should miss, rest should hit
        assert results_list[0][1].cache_hit is False
        assert results_list[1][1].cache_hit is True
        assert results_list[2][1].cache_hit is True

    @pytest.mark.asyncio
    async def test_filter_variations_create_different_cache_keys(self, optimized_search):
        """Test different filters create different cache keys."""
        # Same query, different filters
        results1, metrics1 = await optimized_search.semantic_search(
            "machine learning",
            top_k=10,
            filters={"year_min": 2020},
            use_cache=True
        )

        results2, metrics2 = await optimized_search.semantic_search(
            "machine learning",
            top_k=10,
            filters={"year_min": 2023},
            use_cache=True
        )

        # Both should be cache misses (different filters)
        assert metrics1.cache_hit is False
        assert metrics2.cache_hit is False
