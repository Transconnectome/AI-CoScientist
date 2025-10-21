"""Optimized embedding service with batching and caching."""

from typing import List, Dict, Optional
import numpy as np
import asyncio
from sentence_transformers import SentenceTransformer
import hashlib
import json

from src.core.config import settings
from src.core.redis import get_redis_client


class OptimizedEmbeddingService:
    """Optimized embedding service with caching and batch processing."""

    def __init__(
        self,
        model_name: str | None = None,
        cache_ttl: int = 86400,  # 24 hours
        batch_size: int = 32,
        max_batch_wait_ms: int = 100
    ):
        """Initialize optimized embedding service.

        Args:
            model_name: Name of the sentence transformer model
            cache_ttl: Time-to-live for cached embeddings in seconds
            batch_size: Default batch size for processing
            max_batch_wait_ms: Maximum wait time for batch accumulation
        """
        self.model_name = model_name or settings.embedding_model
        self.cache_ttl = cache_ttl
        self.batch_size = batch_size
        self.max_batch_wait_ms = max_batch_wait_ms

        self._model: Optional[SentenceTransformer] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._pending_batch: List[tuple[str, asyncio.Future]] = []
        self._batch_lock = asyncio.Lock()
        self._batch_timer: Optional[asyncio.Task] = None

        # Performance metrics
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_embeddings = 0
        self._batch_processed = 0

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def encode_async(
        self,
        text: str | List[str],
        use_cache: bool = True
    ) -> np.ndarray:
        """Generate embeddings asynchronously with batching.

        Args:
            text: Single text or list of texts to encode
            use_cache: Whether to use embedding cache

        Returns:
            Numpy array of embeddings
        """
        # Handle single text
        if isinstance(text, str):
            if use_cache:
                # Check cache first
                cached = await self._get_cached_embedding(text)
                if cached is not None:
                    self._cache_hits += 1
                    return cached

                self._cache_misses += 1

            # Add to batch queue
            embedding = await self._add_to_batch(text)

            if use_cache:
                # Cache the result
                await self._cache_embedding(text, embedding)

            self._total_embeddings += 1
            return embedding

        # Handle list of texts
        else:
            tasks = [self.encode_async(t, use_cache=use_cache) for t in text]
            embeddings = await asyncio.gather(*tasks)
            return np.array(embeddings)

    async def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        use_cache: bool = True
    ) -> np.ndarray:
        """Generate embeddings for batch of texts efficiently.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            use_cache: Whether to use embedding cache

        Returns:
            Numpy array of embeddings
        """
        batch_size = batch_size or self.batch_size

        # Check cache for all texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        if use_cache:
            for i, text in enumerate(texts):
                cached = await self._get_cached_embedding(text)
                if cached is not None:
                    embeddings.append((i, cached))
                    self._cache_hits += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self._cache_misses += 1
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Process uncached texts in batches
        if uncached_texts:
            loop = asyncio.get_event_loop()
            new_embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    uncached_texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )
            )

            # Cache new embeddings
            if use_cache:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    await self._cache_embedding(text, embedding)

            # Add to results
            for i, embedding in zip(uncached_indices, new_embeddings):
                embeddings.append((i, embedding))

            self._batch_processed += 1

        # Sort by original index
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings])

        self._total_embeddings += len(texts)
        return result

    async def _add_to_batch(self, text: str) -> np.ndarray:
        """Add text to batch queue and wait for processing.

        Args:
            text: Text to encode

        Returns:
            Embedding for the text
        """
        future: asyncio.Future = asyncio.Future()

        async with self._batch_lock:
            self._pending_batch.append((text, future))

            # Start batch timer if not already running
            if self._batch_timer is None or self._batch_timer.done():
                self._batch_timer = asyncio.create_task(
                    self._process_batch_after_delay()
                )

            # Process immediately if batch is full
            if len(self._pending_batch) >= self.batch_size:
                if self._batch_timer and not self._batch_timer.done():
                    self._batch_timer.cancel()
                await self._process_batch()

        # Wait for result
        return await future

    async def _process_batch_after_delay(self):
        """Process batch after delay."""
        try:
            await asyncio.sleep(self.max_batch_wait_ms / 1000)
            async with self._batch_lock:
                await self._process_batch()
        except asyncio.CancelledError:
            pass

    async def _process_batch(self):
        """Process pending batch of texts."""
        if not self._pending_batch:
            return

        # Get batch
        batch = self._pending_batch
        self._pending_batch = []

        # Extract texts and futures
        texts = [text for text, _ in batch]
        futures = [future for _, future in batch]

        try:
            # Generate embeddings
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(texts, convert_to_numpy=True)
            )

            # Set results
            for future, embedding in zip(futures, embeddings):
                if not future.done():
                    future.set_result(embedding)

            self._batch_processed += 1

        except Exception as e:
            # Set exception for all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    async def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text.

        Args:
            text: Text to look up

        Returns:
            Cached embedding or None
        """
        # Check local cache
        cache_key = self._generate_cache_key(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Check Redis cache
        try:
            redis = await get_redis_client()
            cached_data = await redis.get(f"embedding:{cache_key}")
            if cached_data:
                embedding = np.frombuffer(
                    cached_data,
                    dtype=np.float32
                )
                # Store in local cache
                self._embedding_cache[cache_key] = embedding
                return embedding
        except Exception as e:
            print(f"Redis embedding cache read error: {e}")

        return None

    async def _cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Cache embedding for text.

        Args:
            text: Text that was encoded
            embedding: Generated embedding
        """
        cache_key = self._generate_cache_key(text)

        # Store in local cache
        self._embedding_cache[cache_key] = embedding

        # Limit local cache size
        if len(self._embedding_cache) > 10000:
            # Remove random 10% to keep memory bounded
            keys_to_remove = list(self._embedding_cache.keys())[:1000]
            for key in keys_to_remove:
                del self._embedding_cache[key]

        # Store in Redis cache
        try:
            redis = await get_redis_client()
            await redis.set(
                f"embedding:{cache_key}",
                embedding.astype(np.float32).tobytes(),
                ex=self.cache_ttl
            )
        except Exception as e:
            print(f"Redis embedding cache write error: {e}")

    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text.

        Args:
            text: Text to generate key for

        Returns:
            Cache key string
        """
        # Normalize text
        text_normalized = text.lower().strip()

        # Generate hash including model name for versioning
        key_data = f"{self.model_name}:{text_normalized}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    async def clear_cache(self) -> int:
        """Clear all cached embeddings.

        Returns:
            Number of cache entries cleared
        """
        count = len(self._embedding_cache)
        self._embedding_cache.clear()

        try:
            redis = await get_redis_client()
            cursor = 0
            while True:
                cursor, keys = await redis.scan(
                    cursor,
                    match="embedding:*",
                    count=100
                )
                if keys:
                    await redis.delete(*keys)
                    count += len(keys)
                if cursor == 0:
                    break
        except Exception as e:
            print(f"Redis cache clear error: {e}")

        return count

    def get_stats(self) -> Dict[str, any]:
        """Get performance statistics.

        Returns:
            Dictionary with statistics
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
            "hit_rate": hit_rate,
            "total_embeddings": self._total_embeddings,
            "batches_processed": self._batch_processed,
            "local_cache_size": len(self._embedding_cache),
            "model_name": self.model_name
        }

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()
