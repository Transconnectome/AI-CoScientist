"""Embedding service for text vectorization."""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.config import settings


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self, model_name: str | None = None):
        """Initialize embedding service."""
        self.model_name = model_name or settings.embedding_model
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, text: str | List[str]) -> np.ndarray:
        """Generate embeddings for text."""
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """Generate embeddings for batch of texts."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

    async def encode_async(self, text: str | List[str]) -> np.ndarray:
        """Generate embeddings asynchronously."""
        # Run in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode, text)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()
