"""LLM response caching."""

import hashlib
from typing import Optional

from redis.asyncio import Redis

from src.services.llm.types import LLMConfig


class LLMCache:
    """Cache layer for LLM responses."""

    def __init__(self, redis_client: Redis, ttl: int = 3600):
        """Initialize LLM cache."""
        self.redis = redis_client
        self.ttl = ttl

    async def get(self, prompt: str, config: LLMConfig) -> Optional[str]:
        """Get cached response."""
        cache_key = self._generate_key(prompt, config)
        cached = await self.redis.get(cache_key)
        return cached.decode() if cached else None

    async def set(
        self,
        prompt: str,
        config: LLMConfig,
        response: str,
        ttl: Optional[int] = None
    ) -> None:
        """Cache response."""
        cache_key = self._generate_key(prompt, config)
        await self.redis.setex(
            cache_key,
            ttl or self.ttl,
            response
        )

    async def delete(self, prompt: str, config: LLMConfig) -> None:
        """Delete cached response."""
        cache_key = self._generate_key(prompt, config)
        await self.redis.delete(cache_key)

    def _generate_key(self, prompt: str, config: LLMConfig) -> str:
        """Generate cache key from prompt and config."""
        key_data = f"{prompt}:{config.model}:{config.temperature}:{config.max_tokens}"
        hash_digest = hashlib.sha256(key_data.encode()).hexdigest()
        return f"llm_cache:{hash_digest}"
