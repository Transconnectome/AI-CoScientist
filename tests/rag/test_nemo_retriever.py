"""
Tests for NeMo Retriever service (EmbedQA + RerankQA).

Following TDD methodology (RED → GREEN → REFACTOR):
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Improve code quality
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from src.services.nemo_retriever import (
    NeMoEmbedder,
    NeMoReranker,
    NeMoRetrievalPipeline,
    EmbeddingResult,
    RerankResult,
)


class TestNeMoEmbedder:
    """Tests for NeMo Embedding service"""

    @pytest.fixture
    def embedder(self):
        """Create NeMoEmbedder instance with mocked HTTP client"""
        with patch('httpx.AsyncClient'):
            return NeMoEmbedder(
                base_url="http://localhost:8001/v1",
                model="nvidia/llama-3.2-nv-embedqa-1b-v2"
            )

    @pytest.mark.asyncio
    async def test_embedder_initialization(self, embedder):
        """Test embedder initialization"""
        assert embedder.base_url == "http://localhost:8001/v1"
        assert embedder.model == "nvidia/llama-3.2-nv-embedqa-1b-v2"
        assert embedder.cache_ttl == 3600
        assert embedder.client is not None

    @pytest.mark.asyncio
    async def test_embed_query_single(self, embedder):
        """Test single query embedding"""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "embedding": [0.1, 0.2, 0.3] * 341 + [0.1],  # 1024 dimensions
                    "tokens": 10
                }
            ],
            "model": "nvidia/llama-3.2-nv-embedqa-1b-v2"
        }
        mock_response.raise_for_status = MagicMock()

        embedder.client.post = AsyncMock(return_value=mock_response)

        # Test
        embedding = await embedder.embed_query("What is machine learning?")

        # Assertions
        assert isinstance(embedding, list)
        assert len(embedding) == 1024
        assert all(isinstance(x, float) for x in embedding)

        # Verify API call
        embedder.client.post.assert_called_once()
        call_args = embedder.client.post.call_args
        assert "embeddings" in call_args[0][0]
        assert call_args[1]["json"]["input_type"] == "query"

    @pytest.mark.asyncio
    async def test_embed_documents_batch(self, embedder):
        """Test batch document embedding"""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1] * 1024, "tokens": 50},
                {"embedding": [0.2] * 1024, "tokens": 60},
                {"embedding": [0.3] * 1024, "tokens": 70},
            ],
            "model": "nvidia/llama-3.2-nv-embedqa-1b-v2"
        }
        mock_response.raise_for_status = MagicMock()

        embedder.client.post = AsyncMock(return_value=mock_response)

        # Test
        documents = [
            "Document 1 content",
            "Document 2 content",
            "Document 3 content"
        ]
        embeddings = await embedder.embed_documents(documents)

        # Assertions
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(emb) == 1024 for emb in embeddings)

        # Verify API call
        embedder.client.post.assert_called_once()
        call_args = embedder.client.post.call_args
        assert call_args[1]["json"]["input_type"] == "passage"

    @pytest.mark.asyncio
    async def test_embedding_caching(self, embedder):
        """Test embedding caching with Redis"""
        # Mock Redis client
        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # First call - cache miss
        embedder.redis_client = mock_redis

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 1024, "tokens": 10}],
            "model": "nvidia/llama-3.2-nv-embedqa-1b-v2"
        }
        mock_response.raise_for_status = MagicMock()
        embedder.client.post = AsyncMock(return_value=mock_response)

        # First call - should hit API and cache
        embedding1 = await embedder.embed_query("Test query")

        # Verify cache write
        assert mock_redis.setex.called

        # Second call - mock cache hit
        import pickle
        cached_result = EmbeddingResult(
            embedding=[0.1] * 1024,
            model="nvidia/llama-3.2-nv-embedqa-1b-v2",
            tokens=10
        )
        mock_redis.get.return_value = pickle.dumps(cached_result)

        embedding2 = await embedder.embed_query("Test query")

        # Should use cached result (API called only once)
        assert embedder.client.post.call_count == 1
        assert embedding2 == [0.1] * 1024

    @pytest.mark.asyncio
    async def test_embed_empty_input(self, embedder):
        """Test handling of empty input"""
        # Empty list should return empty result
        embedder.client.post = AsyncMock()

        result = await embedder.embed_texts([], input_type="query")

        assert result == []
        embedder.client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_error_handling(self, embedder):
        """Test error handling for API failures"""
        # Mock HTTP error
        import httpx
        embedder.client.post = AsyncMock(
            side_effect=httpx.HTTPError("Connection failed")
        )

        # Should raise exception
        with pytest.raises(httpx.HTTPError):
            await embedder.embed_query("Test query")


class TestNeMoReranker:
    """Tests for NeMo Reranking service"""

    @pytest.fixture
    def reranker(self):
        """Create NeMoReranker instance with mocked HTTP client"""
        with patch('httpx.AsyncClient'):
            return NeMoReranker(
                base_url="http://localhost:8002/v1",
                model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
                top_k=5
            )

    @pytest.mark.asyncio
    async def test_reranker_initialization(self, reranker):
        """Test reranker initialization"""
        assert reranker.base_url == "http://localhost:8002/v1"
        assert reranker.model == "nvidia/llama-3.2-nv-rerankqa-1b-v2"
        assert reranker.top_k == 5
        assert reranker.client is not None

    @pytest.mark.asyncio
    async def test_rerank_documents(self, reranker):
        """Test document reranking"""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.87},
                {"index": 1, "relevance_score": 0.72},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        reranker.client.post = AsyncMock(return_value=mock_response)

        # Test
        query = "What is deep learning?"
        documents = [
            "Document about neural networks",
            "Document about statistics",
            "Document about deep learning algorithms"
        ]

        results = await reranker.rerank(query, documents, top_k=3)

        # Assertions
        assert isinstance(results, list)
        assert len(results) == 3

        # Should be sorted by relevance score (descending)
        assert results[0].relevance_score >= results[1].relevance_score
        assert results[1].relevance_score >= results[2].relevance_score

        # Verify API call
        reranker.client.post.assert_called_once()
        call_args = reranker.client.post.call_args
        assert "rerank" in call_args[0][0]
        assert call_args[1]["json"]["query"] == query
        assert call_args[1]["json"]["passages"] == documents

    @pytest.mark.asyncio
    async def test_rerank_with_metadata(self, reranker):
        """Test reranking with document metadata"""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.90},
                {"index": 1, "relevance_score": 0.75},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        reranker.client.post = AsyncMock(return_value=mock_response)

        # Test with metadata
        documents = ["Doc 1", "Doc 2"]
        metadata = [
            {"paper_id": "001", "title": "Paper 1"},
            {"paper_id": "002", "title": "Paper 2"}
        ]

        results = await reranker.rerank("Query", documents, metadata=metadata)

        # Assertions
        assert len(results) == 2
        assert results[0].metadata == metadata[0]
        assert results[1].metadata == metadata[1]

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self, reranker):
        """Test handling of empty document list"""
        results = await reranker.rerank("Query", [])

        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_top_k_limit(self, reranker):
        """Test top_k limiting"""
        # Mock response with 10 results
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": i, "relevance_score": 1.0 - (i * 0.05)}
                for i in range(10)
            ]
        }
        mock_response.raise_for_status = MagicMock()

        reranker.client.post = AsyncMock(return_value=mock_response)

        # Request top 3
        documents = [f"Doc {i}" for i in range(10)]
        results = await reranker.rerank("Query", documents, top_k=3)

        # Should return only top 3
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_compress_documents_langchain_compatible(self, reranker):
        """Test LangChain-compatible document compression"""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.92},
                {"index": 0, "relevance_score": 0.88},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        reranker.client.post = AsyncMock(return_value=mock_response)

        # Test with LangChain-style documents
        documents = [
            {
                "page_content": "Content 1",
                "metadata": {"source": "paper1.pdf"}
            },
            {
                "page_content": "Content 2",
                "metadata": {"source": "paper2.pdf"}
            }
        ]

        compressed = await reranker.compress_documents(
            query="Test query",
            documents=documents,
            top_k=2
        )

        # Assertions
        assert len(compressed) == 2
        assert "page_content" in compressed[0]
        assert "metadata" in compressed[0]
        assert "relevance_score" in compressed[0]["metadata"]
        assert "reranked" in compressed[0]["metadata"]


class TestNeMoRetrievalPipeline:
    """Tests for complete NeMo Retrieval Pipeline"""

    @pytest.fixture
    def pipeline(self):
        """Create retrieval pipeline with mocked components"""
        with patch('httpx.AsyncClient'):
            embedder = NeMoEmbedder()
            reranker = NeMoReranker()

            # Mock vector store
            mock_vector_store = MagicMock()
            mock_vector_store.similarity_search_by_vector = AsyncMock()

            return NeMoRetrievalPipeline(
                embedder=embedder,
                reranker=reranker,
                vector_store=mock_vector_store
            )

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline.embedder is not None
        assert pipeline.reranker is not None
        assert pipeline.vector_store is not None

    @pytest.mark.asyncio
    async def test_retrieve_and_rerank_flow(self, pipeline):
        """Test complete retrieve → rerank flow"""
        # Mock embedder
        pipeline.embedder.embed_query = AsyncMock(
            return_value=[0.1] * 1024
        )

        # Mock vector store retrieval
        pipeline.vector_store.similarity_search_by_vector = AsyncMock(
            return_value=[
                {"page_content": f"Document {i}", "metadata": {"id": i}}
                for i in range(10)
            ]
        )

        # Mock reranker
        mock_reranked = [
            {
                "page_content": f"Document {i}",
                "metadata": {"id": i, "relevance_score": 0.9 - (i * 0.1), "reranked": True}
            }
            for i in range(5)
        ]
        pipeline.reranker.compress_documents = AsyncMock(
            return_value=mock_reranked
        )

        # Test pipeline
        query = "Test query"
        docs, scores = await pipeline.retrieve_and_rerank(
            query=query,
            top_k_retrieve=10,
            top_k_rerank=5
        )

        # Assertions
        assert len(docs) == 5
        assert len(scores) == 5
        assert all(score >= 0 and score <= 1 for score in scores)

        # Verify pipeline flow
        pipeline.embedder.embed_query.assert_called_once_with(query)
        pipeline.vector_store.similarity_search_by_vector.assert_called_once()
        pipeline.reranker.compress_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_without_vector_store(self):
        """Test pipeline error when vector store not configured"""
        with patch('httpx.AsyncClient'):
            embedder = NeMoEmbedder()
            reranker = NeMoReranker()

            pipeline = NeMoRetrievalPipeline(
                embedder=embedder,
                reranker=reranker,
                vector_store=None  # No vector store
            )

            # Should raise error
            with pytest.raises(ValueError, match="Vector store not configured"):
                await pipeline.retrieve_and_rerank("Query")

    @pytest.mark.asyncio
    async def test_pipeline_close(self, pipeline):
        """Test pipeline cleanup"""
        pipeline.embedder.close = AsyncMock()
        pipeline.reranker.close = AsyncMock()

        await pipeline.close()

        pipeline.embedder.close.assert_called_once()
        pipeline.reranker.close.assert_called_once()


class TestNeMoRetrievalIntegration:
    """Integration tests for NeMo Retrieval components"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embedding_dimensions(self):
        """Test that embeddings have correct dimensions"""
        # This would require actual API running
        # Marked with @pytest.mark.integration for optional execution
        pass

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_reranking_improves_relevance(self):
        """Test that reranking improves relevance ordering"""
        # This would require actual API running
        # Marked with @pytest.mark.integration for optional execution
        pass

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_retrieval(self):
        """Test complete retrieval pipeline end-to-end"""
        # This would require actual API and vector store
        # Marked with @pytest.mark.integration for optional execution
        pass
