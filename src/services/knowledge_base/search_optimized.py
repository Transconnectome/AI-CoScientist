"""Optimized knowledge base search service with caching and performance monitoring."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import time
import re
import asyncio

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.literature import Literature
from src.services.knowledge_base.vector_store_optimized import OptimizedVectorStore
from src.services.knowledge_base.embedding_optimized import OptimizedEmbeddingService
from src.services.knowledge_base.query_cache import QueryCache
from src.services.knowledge_base.search import SearchResult


class QueryPreprocessor:
    """Preprocessor for query optimization and deduplication."""

    def __init__(self):
        """Initialize query preprocessor."""
        self._stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can'
        }

    def normalize(self, query: str) -> str:
        """Normalize query text.

        Args:
            query: Raw query string

        Returns:
            Normalized query
        """
        # Convert to lowercase
        query = query.lower().strip()

        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)

        # Remove special characters but keep hyphens and apostrophes
        query = re.sub(r'[^\w\s\-\']', '', query)

        return query

    def remove_stopwords(self, query: str) -> str:
        """Remove common stopwords from query.

        Args:
            query: Query string

        Returns:
            Query without stopwords
        """
        words = query.split()
        filtered = [w for w in words if w not in self._stopwords]
        return ' '.join(filtered)

    def expand_query(self, query: str) -> List[str]:
        """Generate query variations for better recall.

        Args:
            query: Query string

        Returns:
            List of query variations
        """
        variations = [query]

        # Add version without stopwords
        no_stopwords = self.remove_stopwords(query)
        if no_stopwords != query:
            variations.append(no_stopwords)

        return variations


@dataclass
class SearchMetrics:
    """Search performance metrics."""
    query_time_ms: float
    cache_hit: bool
    embedding_time_ms: float
    vector_search_time_ms: float
    total_results: int


class OptimizedKnowledgeBaseSearch:
    """Optimized search interface with caching and performance monitoring."""

    def __init__(
        self,
        vector_store: OptimizedVectorStore,
        embedding_service: OptimizedEmbeddingService,
        db: AsyncSession,
        cache: Optional[QueryCache] = None
    ):
        """Initialize optimized search service.

        Args:
            vector_store: Optimized vector store
            embedding_service: Optimized embedding service
            db: Database session
            cache: Query cache (created if not provided)
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.db = db
        self.cache = cache or QueryCache()
        self.preprocessor = QueryPreprocessor()

        # Performance metrics
        self._total_queries = 0
        self._cache_hits = 0
        self._total_query_time = 0.0

    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> tuple[List[SearchResult], SearchMetrics]:
        """Perform optimized semantic search.

        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score
            filters: Optional metadata filters
            use_cache: Whether to use query cache

        Returns:
            Tuple of (search results, metrics)
        """
        start_time = time.time()

        # Normalize query
        query_normalized = self.preprocessor.normalize(query)

        # Check cache
        cache_hit = False
        if use_cache:
            cached_results = await self.cache.get(
                query_normalized,
                top_k,
                filters,
                "semantic"
            )
            if cached_results:
                cache_hit = True
                self._cache_hits += 1
                self._total_queries += 1

                metrics = SearchMetrics(
                    query_time_ms=(time.time() - start_time) * 1000,
                    cache_hit=True,
                    embedding_time_ms=0.0,
                    vector_search_time_ms=0.0,
                    total_results=len(cached_results)
                )

                return cached_results, metrics

        # Generate query embedding
        embedding_start = time.time()
        query_embedding = await self.embedding_service.encode_async(
            query_normalized,
            use_cache=use_cache
        )
        embedding_time = (time.time() - embedding_start) * 1000

        # Build metadata filter
        where_filter = self._build_chroma_filter(filters) if filters else None

        # Search ChromaDB
        search_start = time.time()
        results = await self.vector_store.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter
        )
        search_time = (time.time() - search_start) * 1000

        # Parse and filter results
        search_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            score = 1 - results['distances'][0][i]

            if score < min_score:
                continue

            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]

            search_results.append(SearchResult(
                document_id=doc_id,
                title=metadata.get('title', ''),
                abstract=document,
                score=score,
                metadata=metadata,
                highlights=self._extract_highlights(query_normalized, document)
            ))

        # Cache results
        if use_cache:
            await self.cache.set(
                query_normalized,
                top_k,
                search_results,
                filters,
                "semantic"
            )

        # Update metrics
        self._total_queries += 1
        self._total_query_time += (time.time() - start_time)

        metrics = SearchMetrics(
            query_time_ms=(time.time() - start_time) * 1000,
            cache_hit=False,
            embedding_time_ms=embedding_time,
            vector_search_time_ms=search_time,
            total_results=len(search_results)
        )

        return search_results, metrics

    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> tuple[List[SearchResult], SearchMetrics]:
        """Perform optimized keyword search.

        Args:
            query: Search query
            top_k: Number of results
            filters: Optional filters
            use_cache: Whether to use cache

        Returns:
            Tuple of (search results, metrics)
        """
        start_time = time.time()

        # Normalize query
        query_normalized = self.preprocessor.normalize(query)

        # Check cache
        cache_hit = False
        if use_cache:
            cached_results = await self.cache.get(
                query_normalized,
                top_k,
                filters,
                "keyword"
            )
            if cached_results:
                cache_hit = True
                self._cache_hits += 1
                self._total_queries += 1

                metrics = SearchMetrics(
                    query_time_ms=(time.time() - start_time) * 1000,
                    cache_hit=True,
                    embedding_time_ms=0.0,
                    vector_search_time_ms=0.0,
                    total_results=len(cached_results)
                )

                return cached_results, metrics

        # Build WHERE clause
        where_clauses = [
            "to_tsvector('english', title || ' ' || COALESCE(abstract, '')) @@ plainto_tsquery('english', :query)"
        ]
        params: Dict[str, Any] = {"query": query_normalized}

        if filters:
            if "year_min" in filters:
                where_clauses.append("EXTRACT(YEAR FROM publication_date) >= :year_min")
                params["year_min"] = filters["year_min"]

            if "year_max" in filters:
                where_clauses.append("EXTRACT(YEAR FROM publication_date) <= :year_max")
                params["year_max"] = filters["year_max"]

        where_clause = " AND ".join(where_clauses)

        query_sql = f"""
        SELECT
            id,
            title,
            abstract,
            ts_rank(
                to_tsvector('english', title || ' ' || COALESCE(abstract, '')),
                plainto_tsquery('english', :query)
            ) as rank
        FROM literature
        WHERE {where_clause}
        ORDER BY rank DESC
        LIMIT :limit;
        """

        params["limit"] = top_k

        result = await self.db.execute(text(query_sql), params)
        rows = result.fetchall()

        # Convert to SearchResult
        search_results = []
        for row in rows:
            search_results.append(SearchResult(
                document_id=str(row.id),
                title=row.title,
                abstract=row.abstract or "",
                score=float(row.rank),
                metadata={"title": row.title},
                highlights=self._extract_highlights(query_normalized, row.abstract or "")
            ))

        # Cache results
        if use_cache:
            await self.cache.set(
                query_normalized,
                top_k,
                search_results,
                filters,
                "keyword"
            )

        # Update metrics
        self._total_queries += 1
        self._total_query_time += (time.time() - start_time)

        metrics = SearchMetrics(
            query_time_ms=(time.time() - start_time) * 1000,
            cache_hit=False,
            embedding_time_ms=0.0,
            vector_search_time_ms=(time.time() - start_time) * 1000,
            total_results=len(search_results)
        )

        return search_results, metrics

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> tuple[List[SearchResult], SearchMetrics]:
        """Optimized hybrid search combining semantic and keyword search.

        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic results (0-1)
            filters: Optional filters
            use_cache: Whether to use cache

        Returns:
            Tuple of (search results, metrics)
        """
        start_time = time.time()

        # Normalize query
        query_normalized = self.preprocessor.normalize(query)

        # Check cache
        cache_hit = False
        if use_cache:
            cached_results = await self.cache.get(
                query_normalized,
                top_k,
                filters,
                f"hybrid_{semantic_weight}"
            )
            if cached_results:
                cache_hit = True
                self._cache_hits += 1
                self._total_queries += 1

                metrics = SearchMetrics(
                    query_time_ms=(time.time() - start_time) * 1000,
                    cache_hit=True,
                    embedding_time_ms=0.0,
                    vector_search_time_ms=0.0,
                    total_results=len(cached_results)
                )

                return cached_results, metrics

        # Perform both searches in parallel
        semantic_task = self.semantic_search(
            query_normalized,
            top_k,
            filters=filters,
            use_cache=False  # Don't double-cache
        )
        keyword_task = self.keyword_search(
            query_normalized,
            top_k,
            filters=filters,
            use_cache=False  # Don't double-cache
        )

        (semantic_results, sem_metrics), (keyword_results, kw_metrics) = await asyncio.gather(
            semantic_task,
            keyword_task
        )

        # Combine and re-rank
        combined = self._combine_results(
            semantic_results,
            keyword_results,
            semantic_weight
        )

        final_results = combined[:top_k]

        # Cache results
        if use_cache:
            await self.cache.set(
                query_normalized,
                top_k,
                final_results,
                filters,
                f"hybrid_{semantic_weight}"
            )

        # Update metrics
        self._total_queries += 1
        self._total_query_time += (time.time() - start_time)

        metrics = SearchMetrics(
            query_time_ms=(time.time() - start_time) * 1000,
            cache_hit=False,
            embedding_time_ms=sem_metrics.embedding_time_ms,
            vector_search_time_ms=sem_metrics.vector_search_time_ms + kw_metrics.vector_search_time_ms,
            total_results=len(final_results)
        )

        return final_results, metrics

    def _build_chroma_filter(self, filters: Dict[str, Any]) -> Dict:
        """Build ChromaDB metadata filter."""
        where_filter = {}

        if "year_min" in filters:
            where_filter["year"] = {"$gte": filters["year_min"]}

        if "year_max" in filters:
            if "year" in where_filter:
                where_filter["year"]["$lte"] = filters["year_max"]
            else:
                where_filter["year"] = {"$lte": filters["year_max"]}

        if "fields" in filters:
            where_filter["field_of_study"] = {"$in": filters["fields"]}

        if "journal" in filters:
            where_filter["journal"] = filters["journal"]

        if "min_citations" in filters:
            where_filter["citations_count"] = {"$gte": filters["min_citations"]}

        return where_filter

    def _extract_highlights(
        self,
        query: str,
        text: str,
        window: int = 100
    ) -> List[str]:
        """Extract relevant text snippets."""
        query_terms = query.lower().split()
        text_lower = text.lower()

        highlights = []
        for term in query_terms:
            pos = text_lower.find(term)
            if pos != -1:
                start = max(0, pos - window)
                end = min(len(text), pos + len(term) + window)
                snippet = text[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet = snippet + "..."
                highlights.append(snippet)

        return highlights[:3]

    def _combine_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        semantic_weight: float
    ) -> List[SearchResult]:
        """Combine and re-rank results from multiple sources."""
        scores: Dict[str, Dict] = {}

        # Add semantic scores
        for result in semantic_results:
            scores[result.document_id] = {
                "semantic": result.score,
                "keyword": 0.0,
                "result": result
            }

        # Add keyword scores
        for result in keyword_results:
            if result.document_id in scores:
                scores[result.document_id]["keyword"] = result.score
            else:
                scores[result.document_id] = {
                    "semantic": 0.0,
                    "keyword": result.score,
                    "result": result
                }

        # Compute combined scores
        combined = []
        for doc_id, data in scores.items():
            combined_score = (
                semantic_weight * data["semantic"] +
                (1 - semantic_weight) * data["keyword"]
            )

            result = data["result"]
            result.score = combined_score
            combined.append(result)

        # Sort by combined score
        combined.sort(key=lambda x: x.score, reverse=True)

        return combined

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics.

        Returns:
            Dictionary with performance stats
        """
        cache_hit_rate = (
            self._cache_hits / self._total_queries
            if self._total_queries > 0
            else 0.0
        )

        avg_query_time = (
            self._total_query_time / self._total_queries
            if self._total_queries > 0
            else 0.0
        )

        return {
            "search": {
                "total_queries": self._total_queries,
                "cache_hits": self._cache_hits,
                "cache_hit_rate": cache_hit_rate,
                "avg_query_time_ms": avg_query_time * 1000
            },
            "cache": self.cache.get_stats(),
            "embedding": self.embedding_service.get_stats(),
            "vector_store": self.vector_store.get_stats()
        }

    async def clear_all_caches(self) -> Dict[str, int]:
        """Clear all caches.

        Returns:
            Dictionary with counts of cleared entries
        """
        query_cache_cleared = await self.cache.invalidate()
        embedding_cache_cleared = await self.embedding_service.clear_cache()

        return {
            "query_cache_cleared": query_cache_cleared,
            "embedding_cache_cleared": embedding_cache_cleared
        }
