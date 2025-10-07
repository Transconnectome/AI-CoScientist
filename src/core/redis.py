"""Redis connection and caching utilities."""

from typing import Optional, Any
import json
import pickle

from redis.asyncio import Redis, ConnectionPool

from src.core.config import settings


class RedisClient:
    """Redis client wrapper."""

    def __init__(self) -> None:
        """Initialize Redis client."""
        self.pool = ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=False
        )
        self._client: Optional[Redis] = None

    @property
    def client(self) -> Redis:
        """Get Redis client instance."""
        if self._client is None:
            self._client = Redis(connection_pool=self.pool)
        return self._client

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        value = await self.client.get(key)
        if value is None:
            return None
        try:
            return pickle.loads(value)
        except Exception:
            return value.decode() if isinstance(value, bytes) else value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Set value in cache with optional TTL."""
        try:
            serialized = pickle.dumps(value)
        except Exception:
            serialized = str(value).encode()

        if ttl is not None:
            await self.client.setex(key, ttl, serialized)
        else:
            await self.client.set(key, serialized)

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        await self.client.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return bool(await self.client.exists(key))

    async def get_json(self, key: str) -> Optional[dict]:
        """Get JSON value from cache."""
        value = await self.client.get(key)
        if value is None:
            return None
        return json.loads(value)

    async def set_json(
        self,
        key: str,
        value: dict,
        ttl: Optional[int] = None
    ) -> None:
        """Set JSON value in cache."""
        serialized = json.dumps(value)
        if ttl is not None:
            await self.client.setex(key, ttl, serialized)
        else:
            await self.client.set(key, serialized)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            await self.pool.disconnect()


# Global Redis client
redis_client = RedisClient()


async def get_redis() -> Redis:
    """Get Redis client dependency."""
    return redis_client.client
