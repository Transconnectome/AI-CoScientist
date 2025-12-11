"""
Intelligent Cache System for RAG Operations

Implementation for: Smart caching with TTL and invalidation strategies
Created: 2025-12-05

Acceptance Criteria:
- Query result caching with semantic similarity
- Smart TTL based on content type and freshness
- Intelligent cache invalidation and warming
- Performance metrics and optimization

This module provides intelligent caching for RAG operations with semantic
similarity matching, adaptive TTL, and proactive cache management.
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor

# External dependencies with fallbacks
try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Core dependencies
from ..rag.unified_rag_orchestrator import QueryContext, RAGResponse, RAGStrategy

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache strategy types"""
    EXACT_MATCH = "exact_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    HYBRID = "hybrid"

class ContentType(Enum):
    """Content types for TTL determination"""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    TEMPORAL = "temporal"
    RESEARCH = "research"
    GENERAL = "general"

class InvalidationReason(Enum):
    """Cache invalidation reasons"""
    EXPIRED = "expired"
    MANUAL = "manual"
    CONTENT_UPDATE = "content_update"
    POOR_PERFORMANCE = "poor_performance"
    SEMANTIC_DRIFT = "semantic_drift"

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    query: str
    query_context: QueryContext
    response: RAGResponse
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: int  # seconds
    content_type: ContentType
    embedding: Optional[List[float]] = None
    quality_score: float = 0.0
    performance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)

    def time_until_expiry(self) -> int:
        """Get seconds until expiry"""
        expiry_time = self.created_at + timedelta(seconds=self.ttl)
        return max(0, int((expiry_time - datetime.now()).total_seconds()))

    def update_access(self):
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1

@dataclass
class CachePerformance:
    """Cache performance metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    semantic_matches: int = 0
    exact_matches: int = 0
    invalidations: int = 0
    avg_response_time: float = 0.0
    cache_size: int = 0
    hit_rate: float = 0.0
    semantic_hit_rate: float = 0.0
    memory_usage: int = 0

class CacheBackend(ABC):
    """Abstract cache backend interface"""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key"""
        pass

    @abstractmethod
    async def set(self, entry: CacheEntry) -> bool:
        """Set cache entry"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cache entry"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass

    @abstractmethod
    async def get_all_keys(self) -> List[str]:
        """Get all cache keys"""
        pass

class RedisBackend(CacheBackend):
    """Redis-based cache backend"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", prefix: str = "rag_cache:"):
        self.redis_url = redis_url
        self.prefix = prefix
        self.client: Optional[aioredis.Redis] = None
        self._initialized = False

    async def _ensure_connection(self):
        """Ensure Redis connection"""
        if not self._initialized:
            try:
                self.client = aioredis.from_url(self.redis_url, decode_responses=False)
                await self.client.ping()
                self._initialized = True
                logger.info("Connected to Redis cache backend")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise

    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry to bytes"""
        try:
            # Convert dataclass to dict, handle datetime serialization
            data = asdict(entry)
            data['created_at'] = entry.created_at.isoformat()
            data['last_accessed'] = entry.last_accessed.isoformat()
            data['query_context'] = entry.query_context.__dict__
            data['response'] = entry.response.__dict__
            return json.dumps(data).encode('utf-8')
        except Exception as e:
            logger.error(f"Failed to serialize cache entry: {e}")
            raise

    def _deserialize_entry(self, data: bytes) -> CacheEntry:
        """Deserialize cache entry from bytes"""
        try:
            entry_dict = json.loads(data.decode('utf-8'))

            # Handle datetime deserialization
            entry_dict['created_at'] = datetime.fromisoformat(entry_dict['created_at'])
            entry_dict['last_accessed'] = datetime.fromisoformat(entry_dict['last_accessed'])

            # Reconstruct objects
            query_context = QueryContext(**entry_dict['query_context'])
            response = RAGResponse(**entry_dict['response'])

            entry_dict['query_context'] = query_context
            entry_dict['response'] = response
            entry_dict['content_type'] = ContentType(entry_dict['content_type'])

            return CacheEntry(**entry_dict)
        except Exception as e:
            logger.error(f"Failed to deserialize cache entry: {e}")
            raise

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key"""
        await self._ensure_connection()
        try:
            data = await self.client.get(f"{self.prefix}{key}")
            if data:
                return self._deserialize_entry(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get cache entry: {e}")
            return None

    async def set(self, entry: CacheEntry) -> bool:
        """Set cache entry with TTL"""
        await self._ensure_connection()
        try:
            data = self._serialize_entry(entry)
            await self.client.setex(
                f"{self.prefix}{entry.key}",
                entry.ttl,
                data
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set cache entry: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete cache entry"""
        await self._ensure_connection()
        try:
            result = await self.client.delete(f"{self.prefix}{key}")
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete cache entry: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache entries"""
        await self._ensure_connection()
        try:
            keys = await self.client.keys(f"{self.prefix}*")
            if keys:
                await self.client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    async def get_all_keys(self) -> List[str]:
        """Get all cache keys"""
        await self._ensure_connection()
        try:
            keys = await self.client.keys(f"{self.prefix}*")
            return [key.decode('utf-8').replace(self.prefix, "") for key in keys]
        except Exception as e:
            logger.error(f"Failed to get cache keys: {e}")
            return []

class MemoryBackend(CacheBackend):
    """In-memory cache backend for fallback"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key"""
        async with self._lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired():
                entry.update_access()
                return entry
            elif entry:
                # Remove expired entry
                del self._cache[key]
            return None

    async def set(self, entry: CacheEntry) -> bool:
        """Set cache entry"""
        async with self._lock:
            try:
                # Evict if at max size
                if len(self._cache) >= self.max_size:
                    await self._evict_lru()

                self._cache[entry.key] = entry
                return True
            except Exception as e:
                logger.error(f"Failed to set cache entry: {e}")
                return False

    async def _evict_lru(self):
        """Evict least recently used entry"""
        if not self._cache:
            return

        lru_key = min(self._cache.keys(),
                     key=lambda k: self._cache[k].last_accessed)
        del self._cache[lru_key]

    async def delete(self, key: str) -> bool:
        """Delete cache entry"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> bool:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            return True

    async def get_all_keys(self) -> List[str]:
        """Get all cache keys"""
        async with self._lock:
            return list(self._cache.keys())

class SemanticMatcher:
    """Semantic similarity matching for cache lookup"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.85):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize embedding model"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Initialized semantic matcher: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize semantic matcher: {e}")

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get text embedding"""
        try:
            if self.model:
                embedding = self.model.encode([text])[0]
                return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
        return None

    async def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            if NUMPY_AVAILABLE and SKLEARN_AVAILABLE:
                sim = cosine_similarity([embedding1], [embedding2])[0][0]
                return float(sim)
            else:
                # Simple dot product similarity
                dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
                magnitude1 = sum(a * a for a in embedding1) ** 0.5
                magnitude2 = sum(b * b for b in embedding2) ** 0.5
                return dot_product / (magnitude1 * magnitude2)
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0

    async def find_similar_queries(
        self,
        query_embedding: List[float],
        cache_entries: List[CacheEntry]
    ) -> List[Tuple[CacheEntry, float]]:
        """Find semantically similar cached queries"""
        similar_entries = []

        for entry in cache_entries:
            if entry.embedding:
                similarity = await self.calculate_similarity(query_embedding, entry.embedding)
                if similarity >= self.similarity_threshold:
                    similar_entries.append((entry, similarity))

        # Sort by similarity descending
        similar_entries.sort(key=lambda x: x[1], reverse=True)
        return similar_entries

class TTLManager:
    """Adaptive TTL management"""

    def __init__(self):
        # Base TTL values in seconds
        self.base_ttls = {
            ContentType.FACTUAL: 3600 * 24,      # 24 hours
            ContentType.PROCEDURAL: 3600 * 6,    # 6 hours
            ContentType.TEMPORAL: 3600,          # 1 hour
            ContentType.RESEARCH: 3600 * 12,     # 12 hours
            ContentType.GENERAL: 3600 * 4        # 4 hours
        }

        # TTL adjustment factors
        self.quality_factor = 0.5      # High quality = longer TTL
        self.performance_factor = 0.3  # Good performance = longer TTL
        self.access_factor = 0.2       # Frequent access = longer TTL

    def determine_content_type(self, query: str, query_context: QueryContext) -> ContentType:
        """Determine content type from query"""
        query_lower = query.lower()

        # Temporal indicators
        if any(word in query_lower for word in ['recent', 'latest', 'current', '2024', '2025']):
            return ContentType.TEMPORAL

        # Procedural indicators
        if any(word in query_lower for word in ['how', 'steps', 'process', 'method']):
            return ContentType.PROCEDURAL

        # Research indicators
        if any(word in query_lower for word in ['study', 'research', 'analysis', 'findings']):
            return ContentType.RESEARCH

        # Factual indicators
        if any(word in query_lower for word in ['what', 'define', 'definition', 'is']):
            return ContentType.FACTUAL

        return ContentType.GENERAL

    def calculate_ttl(
        self,
        content_type: ContentType,
        quality_score: float = 0.5,
        performance_score: float = 0.5,
        access_count: int = 0
    ) -> int:
        """Calculate adaptive TTL"""
        base_ttl = self.base_ttls[content_type]

        # Quality adjustment (0.0 to 1.0)
        quality_adj = 1.0 + (quality_score - 0.5) * self.quality_factor

        # Performance adjustment (0.0 to 1.0)
        performance_adj = 1.0 + (performance_score - 0.5) * self.performance_factor

        # Access frequency adjustment (logarithmic)
        access_adj = 1.0 + min(0.5, access_count / 100) * self.access_factor

        # Calculate final TTL
        final_ttl = base_ttl * quality_adj * performance_adj * access_adj

        # Ensure reasonable bounds
        return max(300, min(86400 * 7, int(final_ttl)))  # 5 minutes to 1 week

class IntelligentCache:
    """Main intelligent cache system"""

    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        strategy: CacheStrategy = CacheStrategy.HYBRID,
        semantic_matcher: Optional[SemanticMatcher] = None,
        ttl_manager: Optional[TTLManager] = None,
        max_cache_size: int = 10000
    ):
        self.backend = backend or self._create_default_backend()
        self.strategy = strategy
        self.semantic_matcher = semantic_matcher or SemanticMatcher()
        self.ttl_manager = ttl_manager or TTLManager()
        self.max_cache_size = max_cache_size

        # Performance tracking
        self.performance = CachePerformance()
        self._lock = asyncio.Lock()

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._warmup_task: Optional[asyncio.Task] = None

    def _create_default_backend(self) -> CacheBackend:
        """Create default cache backend"""
        if REDIS_AVAILABLE:
            try:
                return RedisBackend()
            except Exception:
                logger.warning("Failed to create Redis backend, using memory backend")
        return MemoryBackend(max_size=self.max_cache_size)

    async def start(self):
        """Start cache system with background tasks"""
        logger.info("Starting intelligent cache system")

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Start cache warming if configured
        if hasattr(self, 'enable_warmup') and self.enable_warmup:
            self._warmup_task = asyncio.create_task(self._warmup_loop())

    async def stop(self):
        """Stop cache system"""
        logger.info("Stopping intelligent cache system")

        if self._cleanup_task:
            self._cleanup_task.cancel()

        if self._warmup_task:
            self._warmup_task.cancel()

    async def get(
        self,
        query: str,
        query_context: QueryContext,
        strategy_override: Optional[CacheStrategy] = None
    ) -> Optional[RAGResponse]:
        """Get cached response for query"""
        start_time = time.time()

        async with self._lock:
            self.performance.total_requests += 1

        try:
            strategy = strategy_override or self.strategy

            if strategy == CacheStrategy.EXACT_MATCH:
                result = await self._get_exact_match(query, query_context)
            elif strategy == CacheStrategy.SEMANTIC_SIMILARITY:
                result = await self._get_semantic_match(query, query_context)
            else:  # HYBRID
                result = await self._get_hybrid_match(query, query_context)

            # Update performance metrics
            async with self._lock:
                if result:
                    self.performance.cache_hits += 1
                else:
                    self.performance.cache_misses += 1

                response_time = time.time() - start_time
                total_requests = self.performance.total_requests
                self.performance.avg_response_time = (
                    (self.performance.avg_response_time * (total_requests - 1) + response_time) / total_requests
                )
                self.performance.hit_rate = self.performance.cache_hits / total_requests

            return result

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            async with self._lock:
                self.performance.cache_misses += 1
            return None

    async def _get_exact_match(self, query: str, query_context: QueryContext) -> Optional[RAGResponse]:
        """Get exact match from cache"""
        cache_key = self._generate_cache_key(query, query_context)
        entry = await self.backend.get(cache_key)

        if entry and not entry.is_expired():
            entry.update_access()
            await self.backend.set(entry)  # Update access stats
            async with self._lock:
                self.performance.exact_matches += 1
            return entry.response

        return None

    async def _get_semantic_match(self, query: str, query_context: QueryContext) -> Optional[RAGResponse]:
        """Get semantic similarity match from cache"""
        if not self.semantic_matcher.model:
            return None

        # Get query embedding
        query_embedding = await self.semantic_matcher.get_embedding(query)
        if not query_embedding:
            return None

        # Get all cache entries (could be optimized with better indexing)
        all_keys = await self.backend.get_all_keys()
        cache_entries = []

        for key in all_keys[:100]:  # Limit to recent entries for performance
            entry = await self.backend.get(key)
            if entry and not entry.is_expired() and entry.embedding:
                cache_entries.append(entry)

        # Find similar entries
        similar_entries = await self.semantic_matcher.find_similar_queries(
            query_embedding, cache_entries
        )

        if similar_entries:
            best_entry, similarity = similar_entries[0]
            best_entry.update_access()
            await self.backend.set(best_entry)  # Update access stats

            async with self._lock:
                self.performance.semantic_matches += 1

            # Adjust confidence based on similarity
            response = best_entry.response
            response.confidence = response.confidence * similarity
            response.metadata = response.metadata or {}
            response.metadata['cache_similarity'] = similarity

            return response

        return None

    async def _get_hybrid_match(self, query: str, query_context: QueryContext) -> Optional[RAGResponse]:
        """Get hybrid match (exact first, then semantic)"""
        # Try exact match first
        result = await self._get_exact_match(query, query_context)
        if result:
            return result

        # Fall back to semantic match
        return await self._get_semantic_match(query, query_context)

    async def set(
        self,
        query: str,
        query_context: QueryContext,
        response: RAGResponse,
        quality_score: float = 0.5,
        performance_score: float = 0.5
    ) -> bool:
        """Set cached response"""
        try:
            # Determine content type and TTL
            content_type = self.ttl_manager.determine_content_type(query, query_context)
            ttl = self.ttl_manager.calculate_ttl(content_type, quality_score, performance_score)

            # Generate cache key
            cache_key = self._generate_cache_key(query, query_context)

            # Get embedding if using semantic matching
            embedding = None
            if self.strategy in [CacheStrategy.SEMANTIC_SIMILARITY, CacheStrategy.HYBRID]:
                embedding = await self.semantic_matcher.get_embedding(query)

            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                query=query,
                query_context=query_context,
                response=response,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                ttl=ttl,
                content_type=content_type,
                embedding=embedding,
                quality_score=quality_score,
                performance_score=performance_score
            )

            # Store in backend
            success = await self.backend.set(entry)

            if success:
                async with self._lock:
                    self.performance.cache_size += 1

            return success

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def invalidate(self, keys: List[str], reason: InvalidationReason = InvalidationReason.MANUAL):
        """Invalidate cache entries"""
        try:
            invalidated_count = 0
            for key in keys:
                if await self.backend.delete(key):
                    invalidated_count += 1

            async with self._lock:
                self.performance.invalidations += invalidated_count
                self.performance.cache_size -= invalidated_count

            logger.info(f"Invalidated {invalidated_count} cache entries (reason: {reason})")

        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")

    async def clear(self):
        """Clear all cache"""
        success = await self.backend.clear()
        if success:
            async with self._lock:
                self.performance = CachePerformance()
        return success

    async def warm_cache(self, warm_queries: List[Tuple[str, QueryContext]]):
        """Warm cache with common queries"""
        logger.info(f"Warming cache with {len(warm_queries)} queries")

        for query, context in warm_queries:
            # Check if already cached
            cached = await self.get(query, context)
            if not cached:
                logger.info(f"Cache miss for warmup query: {query[:50]}...")
                # Would trigger actual RAG retrieval in real system

    def _generate_cache_key(self, query: str, query_context: QueryContext) -> str:
        """Generate cache key from query and context"""
        # Create deterministic hash from query and relevant context
        key_data = {
            'query': query.lower().strip(),
            'domain': str(query_context.domain),
            'complexity': str(query_context.complexity),
            'intent': query_context.intent
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    async def _cleanup_loop(self):
        """Background cleanup of expired entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _cleanup_expired(self):
        """Clean up expired cache entries"""
        try:
            all_keys = await self.backend.get_all_keys()
            expired_keys = []

            for key in all_keys:
                entry = await self.backend.get(key)
                if entry and entry.is_expired():
                    expired_keys.append(key)

            if expired_keys:
                await self.invalidate(expired_keys, InvalidationReason.EXPIRED)
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    async def _warmup_loop(self):
        """Background cache warming"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                # Would implement intelligent cache warming based on usage patterns
                logger.info("Cache warming cycle completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Warmup loop error: {e}")

    async def get_performance_metrics(self) -> CachePerformance:
        """Get cache performance metrics"""
        async with self._lock:
            # Update calculated metrics
            total_requests = self.performance.total_requests
            if total_requests > 0:
                self.performance.hit_rate = self.performance.cache_hits / total_requests
                self.performance.semantic_hit_rate = self.performance.semantic_matches / total_requests

            # Get current cache size
            all_keys = await self.backend.get_all_keys()
            self.performance.cache_size = len(all_keys)

            return self.performance

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        metrics = await self.get_performance_metrics()
        all_keys = await self.backend.get_all_keys()

        # Analyze cache content
        content_type_counts = {}
        avg_ttl_by_type = {}
        total_access_counts = 0

        for key in all_keys[:100]:  # Sample for performance
            entry = await self.backend.get(key)
            if entry:
                content_type = entry.content_type.value
                content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1

                if content_type not in avg_ttl_by_type:
                    avg_ttl_by_type[content_type] = []
                avg_ttl_by_type[content_type].append(entry.ttl)

                total_access_counts += entry.access_count

        # Calculate averages
        for content_type in avg_ttl_by_type:
            ttls = avg_ttl_by_type[content_type]
            avg_ttl_by_type[content_type] = sum(ttls) / len(ttls) if ttls else 0

        return {
            "performance": asdict(metrics),
            "content_distribution": content_type_counts,
            "avg_ttl_by_type": avg_ttl_by_type,
            "total_accesses": total_access_counts,
            "backend_type": type(self.backend).__name__,
            "strategy": self.strategy.value
        }

def create_intelligent_cache(
    redis_url: Optional[str] = None,
    strategy: CacheStrategy = CacheStrategy.HYBRID,
    max_cache_size: int = 10000
) -> IntelligentCache:
    """Factory function to create intelligent cache"""
    backend = None
    if redis_url and REDIS_AVAILABLE:
        try:
            backend = RedisBackend(redis_url)
        except Exception:
            logger.warning("Failed to create Redis backend, using memory backend")

    if not backend:
        backend = MemoryBackend(max_size=max_cache_size)

    return IntelligentCache(
        backend=backend,
        strategy=strategy,
        max_cache_size=max_cache_size
    )

# Example usage
if __name__ == "__main__":
    async def test_intelligent_cache():
        """Test intelligent cache system"""
        cache = create_intelligent_cache()
        await cache.start()

        try:
            # Create test query context
            from ..rag.unified_rag_orchestrator import QueryContext, QueryComplexity, QueryDomain

            query_context = QueryContext(
                query="What is machine learning?",
                complexity=QueryComplexity.SIMPLE,
                domain=QueryDomain.GENERAL,
                intent="factual",
                confidence=0.9,
                metadata={}
            )

            # Test cache miss
            result = await cache.get("What is machine learning?", query_context)
            print(f"Cache miss result: {result}")

            # Create mock response
            from ..rag.unified_rag_orchestrator import RAGResponse, RAGStrategy

            mock_response = RAGResponse(
                answer="Machine learning is a subset of AI...",
                sources=[],
                confidence=0.9,
                strategy_used=RAGStrategy.SIMPLE_RAG,
                performance_metrics=None
            )

            # Test cache set
            success = await cache.set(
                "What is machine learning?",
                query_context,
                mock_response,
                quality_score=0.8,
                performance_score=0.9
            )
            print(f"Cache set success: {success}")

            # Test cache hit
            result = await cache.get("What is machine learning?", query_context)
            print(f"Cache hit result: {result is not None}")

            # Get performance stats
            stats = await cache.get_cache_stats()
            print(f"Cache stats: {stats}")

        finally:
            await cache.stop()

    # Run test
    asyncio.run(test_intelligent_cache())