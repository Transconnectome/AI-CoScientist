"""Knowledge base search service."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.literature import Literature, Author, FieldOfStudy, Citation
from src.services.knowledge_base.vector_store import VectorStore
from src.services.knowledge_base.embedding import EmbeddingService


@dataclass
class SearchResult:
    """Search result from knowledge base."""
    document_id: str
    title: str
    abstract: str
    score: float
    metadata: Dict[str, Any]
    highlights: List[str]


class KnowledgeBaseSearch:
    """Search interface for knowledge base."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        db: AsyncSession
    ):
        """Initialize search service."""
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.db = db

    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        # Generate query embedding
        query_embedding = await self.embedding_service.encode_async(query)

        # Build metadata filter for ChromaDB
        where_filter = self._build_chroma_filter(filters) if filters else None

        # Search ChromaDB
        results = self.vector_store.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter
        )

        # Parse and filter results
        search_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            score = 1 - results['distances'][0][i]  # Convert distance to similarity

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
                highlights=self._extract_highlights(query, document)
            ))

        return search_results

    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform keyword search using PostgreSQL full-text search."""
        # Build WHERE clause
        where_clauses = [
            "to_tsvector('english', title || ' ' || COALESCE(abstract, '')) @@ plainto_tsquery('english', :query)"
        ]
        params: Dict[str, Any] = {"query": query}

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
                highlights=self._extract_highlights(query, row.abstract or "")
            ))

        return search_results

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword search."""
        # Perform both searches
        semantic_results = await self.semantic_search(
            query,
            top_k * 2,
            filters=filters
        )

        keyword_results = await self.keyword_search(
            query,
            top_k * 2,
            filters=filters
        )

        # Combine and re-rank
        combined = self._combine_results(
            semantic_results,
            keyword_results,
            semantic_weight
        )

        return combined[:top_k]

    async def find_similar_papers(
        self,
        paper_id: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Find papers similar to a given paper."""
        # Get paper embedding from ChromaDB
        paper_doc = self.vector_store.get_document(paper_id)

        if not paper_doc or not paper_doc['embedding']:
            raise ValueError(f"Paper {paper_id} not found or not embedded")

        # Search for similar
        results = self.vector_store.query(
            query_embeddings=[paper_doc['embedding']],
            n_results=top_k + 1  # +1 to exclude self
        )

        # Parse and filter out the query paper itself
        search_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            if doc_id == paper_id:
                continue

            score = 1 - results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            document = results['documents'][0][i]

            search_results.append(SearchResult(
                document_id=doc_id,
                title=metadata.get('title', ''),
                abstract=document,
                score=score,
                metadata=metadata,
                highlights=[]
            ))

        return search_results[:top_k]

    async def citation_based_search(
        self,
        paper_id: str,
        depth: int = 2,
        direction: str = "both"
    ) -> List[str]:
        """Find papers through citation network."""
        query_text = """
        WITH RECURSIVE citation_tree AS (
            SELECT
                CASE
                    WHEN :direction IN ('citing', 'both')
                    THEN citing_paper_id
                    ELSE cited_paper_id
                END as paper_id,
                1 as depth
            FROM citations
            WHERE
                CASE
                    WHEN :direction = 'citing' THEN cited_paper_id = :paper_id
                    WHEN :direction = 'cited' THEN citing_paper_id = :paper_id
                    ELSE citing_paper_id = :paper_id OR cited_paper_id = :paper_id
                END

            UNION ALL

            SELECT
                CASE
                    WHEN :direction IN ('citing', 'both')
                    THEN c.citing_paper_id
                    ELSE c.cited_paper_id
                END as paper_id,
                ct.depth + 1
            FROM citations c
            INNER JOIN citation_tree ct ON
                CASE
                    WHEN :direction = 'citing' THEN c.cited_paper_id = ct.paper_id
                    WHEN :direction = 'cited' THEN c.citing_paper_id = ct.paper_id
                    ELSE c.citing_paper_id = ct.paper_id OR c.cited_paper_id = ct.paper_id
                END
            WHERE ct.depth < :max_depth
        )
        SELECT DISTINCT paper_id::text
        FROM citation_tree
        WHERE paper_id::text != :paper_id;
        """

        result = await self.db.execute(
            text(query_text),
            {
                "paper_id": paper_id,
                "direction": direction,
                "max_depth": depth
            }
        )

        return [row[0] for row in result.fetchall()]

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
        # Create score dictionary
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
