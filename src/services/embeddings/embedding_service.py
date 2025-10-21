"""
Embedding Service for RAG integration

Provides unified interface for generating text embeddings
with support for multiple providers (OpenAI, Anthropic/Voyage).
"""

import os
from typing import List, Optional
import openai
from openai import AsyncOpenAI


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize embedding service.

        Args:
            provider: Embedding provider ("openai" or "voyage")
            api_key: API key (optional, will use env var if not provided)
        """
        self.provider = provider

        if provider == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")

            self.client = AsyncOpenAI(api_key=api_key)
            self.model = "text-embedding-ada-002"
            self.max_tokens = 8191
            self.dimension = 1536

        elif provider == "voyage":
            # Voyage AI for Anthropic users
            api_key = api_key or os.getenv("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError("VOYAGE_API_KEY not found")

            try:
                import voyageai
                self.client = voyageai.Client(api_key=api_key)
                self.model = "voyage-large-2"
                self.max_tokens = 16000
                self.dimension = 1536
            except ImportError:
                raise ImportError("voyageai package not installed. Run: pip install voyageai")

        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Truncate if too long
        text = text[:self.max_tokens * 4]  # Rough char estimate

        if self.provider == "openai":
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding

        elif self.provider == "voyage":
            result = self.client.embed(
                texts=[text],
                model=self.model
            )
            return result.embeddings[0]

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Truncate texts
            batch = [t[:self.max_tokens * 4] for t in batch]

            if self.provider == "openai":
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [d.embedding for d in response.data]

            elif self.provider == "voyage":
                result = self.client.embed(
                    texts=batch,
                    model=self.model
                )
                batch_embeddings = result.embeddings

            embeddings.extend(batch_embeddings)

        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "provider": self.provider,
            "model": self.model,
            "dimension": self.dimension,
            "max_tokens": self.max_tokens
        }
