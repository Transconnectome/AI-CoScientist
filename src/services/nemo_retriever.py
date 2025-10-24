"""
NeMo Retriever Service

Provides embeddings and reranking capabilities using NVIDIA NeMo Retriever models:
- Llama 3.2 EmbedQA 1B for document embeddings
- Llama 3.2 RerankQA 1B for result reranking
"""

import os
import httpx
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import pickle
import logging

import numpy as np
from redis import Redis

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""
    embedding: List[float]
    model: str
    tokens: int


@dataclass
class RerankResult:
    """Result from reranking operation."""
    index: int
    document: str
    relevance_score: float
    metadata: Optional[Dict[str, Any]] = None


class NeMoEmbedder:
    """
    NeMo Retriever Embedding Service.

    Uses Llama 3.2 EmbedQA 1B for generating embeddings optimized for Q&A tasks.
    """

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        redis_client: Optional[Redis] = None,
        cache_ttl: int = 3600
    ):
        """
        Initialize NeMo Embedder.

        Args:
            base_url: Base URL for embedding service (default: env EMBEDDER_BASE_URL)
            model: Model identifier (default: env EMBEDDER_MODEL)
            redis_client: Optional Redis client for caching
            cache_ttl: Cache TTL in seconds (default: 1 hour)
        """
        self.base_url = base_url or os.getenv(
            "EMBEDDER_BASE_URL",
            "http://localhost:8001/v1"
        )
        self.model = model or os.getenv(
            "EMBEDDER_MODEL",
            "nvidia/llama-3.2-nv-embedqa-1b-v2"
        )
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
        self.client = httpx.AsyncClient(timeout=30.0)

        logger.info(f"Initialized NeMo Embedder: {self.base_url}, model={self.model}")

    async def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            List of floats representing the embedding vector
        """
        result = await self.embed_texts([text], input_type="query")
        return result[0].embedding

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        results = await self.embed_texts(texts, input_type="passage")
        return [r.embedding for r in results]

    async def embed_texts(
        self,
        texts: List[str],
        input_type: str = "query"
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            input_type: Type of input - "query" or "passage"

        Returns:
            List of EmbeddingResult objects
        """
        # Check cache first
        cached_results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached = await self._get_cached_embedding(text, input_type)
            if cached:
                cached_results.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # If all cached, return immediately
        if not uncached_texts:
            results = [None] * len(texts)
            for i, result in cached_results:
                results[i] = result
            return results

        # Generate embeddings for uncached texts
        try:
            response = await self.client.post(
                f"{self.base_url}/embeddings",
                json={
                    "input": uncached_texts,
                    "model": self.model,
                    "input_type": input_type
                }
            )
            response.raise_for_status()
            data = response.json()

            # Parse response
            new_results = []
            for embedding_data in data["data"]:
                result = EmbeddingResult(
                    embedding=embedding_data["embedding"],
                    model=data.get("model", self.model),
                    tokens=embedding_data.get("tokens", 0)
                )
                new_results.append(result)

            # Cache new results
            for text, result in zip(uncached_texts, new_results):
                await self._cache_embedding(text, input_type, result)

            # Combine cached and new results
            all_results = [None] * len(texts)
            for i, result in cached_results:
                all_results[i] = result
            for i, result in zip(uncached_indices, new_results):
                all_results[i] = result

            return all_results

        except httpx.HTTPError as e:
            logger.error(f"Embedding request failed: {e}")
            raise

    async def _get_cached_embedding(
        self,
        text: str,
        input_type: str
    ) -> Optional[EmbeddingResult]:
        """Get cached embedding if available."""
        if not self.redis_client:
            return None

        cache_key = self._make_cache_key(text, input_type)
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return pickle.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")

        return None

    async def _cache_embedding(
        self,
        text: str,
        input_type: str,
        result: EmbeddingResult
    ) -> None:
        """Cache embedding result."""
        if not self.redis_client:
            return

        cache_key = self._make_cache_key(text, input_type)
        try:
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                pickle.dumps(result)
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def _make_cache_key(self, text: str, input_type: str) -> str:
        """Generate cache key for embedding."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"nemo:embed:{input_type}:{text_hash}"

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class NeMoReranker:
    """
    NeMo Retriever Reranking Service.

    Uses Llama 3.2 RerankQA 1B for reranking retrieved documents.
    """

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        top_k: int = None
    ):
        """
        Initialize NeMo Reranker.

        Args:
            base_url: Base URL for reranking service (default: env RERANKER_BASE_URL)
            model: Model identifier (default: env RERANKER_MODEL)
            top_k: Number of top results to return (default: env RERANKER_TOP_K or 5)
        """
        self.base_url = base_url or os.getenv(
            "RERANKER_BASE_URL",
            "http://localhost:8002/v1"
        )
        self.model = model or os.getenv(
            "RERANKER_MODEL",
            "nvidia/llama-3.2-nv-rerankqa-1b-v2"
        )
        self.top_k = top_k or int(os.getenv("RERANKER_TOP_K", "5"))
        self.client = httpx.AsyncClient(timeout=30.0)

        logger.info(f"Initialized NeMo Reranker: {self.base_url}, model={self.model}")

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[RerankResult]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (default: self.top_k)
            metadata: Optional metadata for each document

        Returns:
            List of RerankResult objects, sorted by relevance (highest first)
        """
        if not documents:
            return []

        top_k = top_k or self.top_k

        try:
            response = await self.client.post(
                f"{self.base_url}/rerank",
                json={
                    "query": query,
                    "passages": documents,
                    "model": self.model,
                    "top_n": min(top_k, len(documents))
                }
            )
            response.raise_for_status()
            data = response.json()

            # Parse results
            results = []
            for result_data in data["results"]:
                index = result_data["index"]
                result = RerankResult(
                    index=index,
                    document=documents[index],
                    relevance_score=result_data["relevance_score"],
                    metadata=metadata[index] if metadata else None
                )
                results.append(result)

            # Sort by relevance score (descending)
            results.sort(key=lambda x: x.relevance_score, reverse=True)

            return results[:top_k]

        except httpx.HTTPError as e:
            logger.error(f"Reranking request failed: {e}")
            raise

    async def compress_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank and compress documents (LangChain-compatible interface).

        Args:
            query: Search query
            documents: List of document dicts with 'page_content' and 'metadata'
            top_k: Number of top results to return

        Returns:
            List of reranked documents with relevance scores added to metadata
        """
        texts = [doc.get("page_content", "") for doc in documents]
        metadata_list = [doc.get("metadata", {}) for doc in documents]

        rerank_results = await self.rerank(
            query=query,
            documents=texts,
            top_k=top_k,
            metadata=metadata_list
        )

        # Convert back to document format
        compressed_docs = []
        for result in rerank_results:
            doc = {
                "page_content": result.document,
                "metadata": {
                    **(result.metadata or {}),
                    "relevance_score": result.relevance_score,
                    "reranked": True
                }
            }
            compressed_docs.append(doc)

        return compressed_docs

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class NeMoRetrievalPipeline:
    """
    Complete NeMo Retrieval Pipeline.

    Combines embedding and reranking for end-to-end retrieval.
    """

    def __init__(
        self,
        embedder: NeMoEmbedder,
        reranker: NeMoReranker,
        vector_store=None
    ):
        """
        Initialize retrieval pipeline.

        Args:
            embedder: NeMo embedder instance
            reranker: NeMo reranker instance
            vector_store: Vector store for similarity search (e.g., ChromaDB)
        """
        self.embedder = embedder
        self.reranker = reranker
        self.vector_store = vector_store

        logger.info("Initialized NeMo Retrieval Pipeline")

    async def retrieve_and_rerank(
        self,
        query: str,
        top_k_retrieve: int = 10,
        top_k_rerank: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve documents and rerank them.

        Args:
            query: Search query
            top_k_retrieve: Number of documents to retrieve initially
            top_k_rerank: Number of documents to return after reranking

        Returns:
            Tuple of (reranked_documents, relevance_scores)
        """
        # Step 1: Embed query
        query_embedding = await self.embedder.embed_query(query)

        # Step 2: Retrieve candidates from vector store
        if self.vector_store:
            candidates = await self.vector_store.similarity_search_by_vector(
                query_embedding,
                k=top_k_retrieve
            )
        else:
            raise ValueError("Vector store not configured")

        # Step 3: Rerank candidates
        reranked = await self.reranker.compress_documents(
            query=query,
            documents=candidates,
            top_k=top_k_rerank
        )

        # Extract relevance scores
        scores = [doc["metadata"]["relevance_score"] for doc in reranked]

        logger.info(
            f"Retrieval pipeline: {top_k_retrieve} retrieved → {len(reranked)} reranked"
        )

        return reranked, scores

    async def close(self):
        """Close all clients."""
        await self.embedder.close()
        await self.reranker.close()
