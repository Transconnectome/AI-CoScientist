"""Unit tests for optimized embedding service."""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.services.knowledge_base.embedding_optimized import OptimizedEmbeddingService


@pytest.fixture
def embedding_service():
    """Create optimized embedding service fixture."""
    with patch('src.services.knowledge_base.embedding_optimized.SentenceTransformer'):
        service = OptimizedEmbeddingService(
            model_name="test-model",
            cache_ttl=60,
            batch_size=4,
            max_batch_wait_ms=50
        )
        return service


@pytest.fixture
def mock_model():
    """Create mock sentence transformer model."""
    model = MagicMock()
    model.encode.return_value = np.random.rand(384)  # Mock 384-dim embedding
    model.get_sentence_embedding_dimension.return_value = 384
    return model


class TestOptimizedEmbeddingService:
    """Test cases for OptimizedEmbeddingService."""

    def test_lazy_model_loading(self, embedding_service, mock_model):
        """Test model is lazy loaded."""
        assert embedding_service._model is None

        with patch('src.services.knowledge_base.embedding_optimized.SentenceTransformer', return_value=mock_model):
            model = embedding_service.model
            assert model is not None
            assert embedding_service._model is not None

    @pytest.mark.asyncio
    async def test_encode_async_single_text(self, embedding_service, mock_model):
        """Test encoding single text asynchronously."""
        embedding_service._model = mock_model
        mock_model.encode.return_value = np.array([0.1] * 384)

        with patch.object(embedding_service, '_add_to_batch') as mock_batch:
            mock_batch.return_value = np.array([0.1] * 384)

            result = await embedding_service.encode_async("test text", use_cache=False)

            assert result is not None
            assert isinstance(result, np.ndarray)
            mock_batch.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.services.knowledge_base.embedding_optimized.get_redis_client')
    async def test_encode_async_with_cache(self, mock_redis, embedding_service, mock_model):
        """Test encoding with cache."""
        embedding_service._model = mock_model

        # Mock Redis
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = None
        mock_redis_instance.set.return_value = True
        mock_redis.return_value = mock_redis_instance

        # First call should miss cache and encode
        with patch.object(embedding_service, '_add_to_batch') as mock_batch:
            mock_batch.return_value = np.array([0.1] * 384)

            result1 = await embedding_service.encode_async("test text", use_cache=True)

            assert result1 is not None
            assert embedding_service._cache_misses == 1

        # Second call should hit local cache
        result2 = await embedding_service.encode_async("test text", use_cache=True)

        assert result2 is not None
        assert embedding_service._cache_hits == 1

    @pytest.mark.asyncio
    async def test_encode_batch(self, embedding_service, mock_model):
        """Test batch encoding."""
        embedding_service._model = mock_model

        texts = ["text1", "text2", "text3"]
        mock_model.encode.return_value = np.array([[0.1] * 384] * 3)

        result = await embedding_service.encode_batch(texts, use_cache=False)

        assert result.shape == (3, 384)
        assert embedding_service._batch_processed == 1

    @pytest.mark.asyncio
    @patch('src.services.knowledge_base.embedding_optimized.get_redis_client')
    async def test_encode_batch_with_cache(self, mock_redis, embedding_service, mock_model):
        """Test batch encoding with cache."""
        embedding_service._model = mock_model

        # Mock Redis
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = None
        mock_redis_instance.set.return_value = True
        mock_redis.return_value = mock_redis_instance

        texts = ["text1", "text2", "text3"]
        mock_model.encode.return_value = np.array([[0.1] * 384] * 3)

        # First call - all cache misses
        result1 = await embedding_service.encode_batch(texts, use_cache=True)

        assert result1.shape == (3, 384)
        assert embedding_service._cache_misses == 3

        # Second call - all cache hits
        result2 = await embedding_service.encode_batch(texts, use_cache=True)

        assert result2.shape == (3, 384)
        assert embedding_service._cache_hits == 3

    @pytest.mark.asyncio
    async def test_batch_accumulation(self, embedding_service, mock_model):
        """Test batch accumulation and processing."""
        embedding_service._model = mock_model
        mock_model.encode.return_value = np.array([[0.1] * 384] * 4)

        # Send 4 requests (batch_size=4)
        tasks = [
            embedding_service.encode_async(f"text{i}", use_cache=False)
            for i in range(4)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        assert all(isinstance(r, np.ndarray) for r in results)

        # Should have processed as one batch
        assert embedding_service._batch_processed >= 1

    @pytest.mark.asyncio
    async def test_batch_timer_processing(self, embedding_service, mock_model):
        """Test batch processing after timer delay."""
        embedding_service._model = mock_model
        mock_model.encode.return_value = np.array([[0.1] * 384] * 2)

        # Send 2 requests (less than batch_size=4)
        # Should process after max_batch_wait_ms delay
        tasks = [
            embedding_service.encode_async(f"text{i}", use_cache=False)
            for i in range(2)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 2
        assert all(isinstance(r, np.ndarray) for r in results)

    def test_cache_key_generation(self, embedding_service):
        """Test cache key generation is deterministic."""
        key1 = embedding_service._generate_cache_key("test text")
        key2 = embedding_service._generate_cache_key("test text")

        assert key1 == key2

    def test_cache_key_normalization(self, embedding_service):
        """Test cache key respects text normalization."""
        key1 = embedding_service._generate_cache_key("Test Text")
        key2 = embedding_service._generate_cache_key("test text")

        # Should be same after normalization
        assert key1 == key2

    @pytest.mark.asyncio
    @patch('src.services.knowledge_base.embedding_optimized.get_redis_client')
    async def test_clear_cache(self, mock_redis, embedding_service, mock_model):
        """Test clearing all caches."""
        embedding_service._model = mock_model

        # Mock Redis
        mock_redis_instance = AsyncMock()
        mock_redis_instance.scan.return_value = (0, [])
        mock_redis.return_value = mock_redis_instance

        # Add some entries to local cache
        embedding_service._embedding_cache["key1"] = np.array([0.1] * 384)
        embedding_service._embedding_cache["key2"] = np.array([0.2] * 384)

        # Clear cache
        count = await embedding_service.clear_cache()

        assert len(embedding_service._embedding_cache) == 0
        assert count >= 2

    def test_get_stats(self, embedding_service):
        """Test getting performance statistics."""
        embedding_service._cache_hits = 10
        embedding_service._cache_misses = 5
        embedding_service._total_embeddings = 20
        embedding_service._batch_processed = 3

        stats = embedding_service.get_stats()

        assert stats["cache_hits"] == 10
        assert stats["cache_misses"] == 5
        assert stats["hit_rate"] == 10 / 15
        assert stats["total_embeddings"] == 20
        assert stats["batches_processed"] == 3

    def test_local_cache_size_limit(self, embedding_service):
        """Test local cache size is limited."""
        # Add more than 10000 entries
        for i in range(11000):
            embedding_service._embedding_cache[f"key{i}"] = np.array([0.1] * 384)

        # Trigger cache management
        embedding_service._cache_embedding.__wrapped__(
            embedding_service,
            "test",
            np.array([0.1] * 384)
        )

        # Should be under limit
        assert len(embedding_service._embedding_cache) <= 10000

    @pytest.mark.asyncio
    async def test_encode_list_of_texts(self, embedding_service, mock_model):
        """Test encoding list of texts."""
        embedding_service._model = mock_model

        texts = ["text1", "text2"]

        with patch.object(embedding_service, '_add_to_batch') as mock_batch:
            mock_batch.side_effect = [
                np.array([0.1] * 384),
                np.array([0.2] * 384)
            ]

            result = await embedding_service.encode_async(texts, use_cache=False)

            assert result.shape == (2, 384)
            assert mock_batch.call_count == 2

    def test_get_embedding_dimension(self, embedding_service, mock_model):
        """Test getting embedding dimension."""
        embedding_service._model = mock_model

        dim = embedding_service.get_embedding_dimension()

        assert dim == 384
        mock_model.get_sentence_embedding_dimension.assert_called_once()
