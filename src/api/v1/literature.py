"""Literature and knowledge base endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from src.core.database import get_db
from src.core.redis import get_redis
from src.schemas.literature import (
    Literature,
    LiteratureSearchRequest,
    SearchResultSchema,
    LiteratureIngestRequest
)
from src.services.knowledge_base import VectorStore, EmbeddingService, KnowledgeBaseSearch
from src.services.knowledge_base.ingestion import LiteratureIngestion
from src.services.external import SemanticScholarClient, CrossRefClient

router = APIRouter()


def get_vector_store() -> VectorStore:
    """Get vector store dependency."""
    return VectorStore()


def get_embedding_service() -> EmbeddingService:
    """Get embedding service dependency."""
    return EmbeddingService()


def get_knowledge_base_search(
    db: AsyncSession = Depends(get_db),
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> KnowledgeBaseSearch:
    """Get knowledge base search dependency."""
    return KnowledgeBaseSearch(vector_store, embedding_service, db)


@router.post("/search", response_model=List[SearchResultSchema])
async def search_literature(
    request: LiteratureSearchRequest,
    kb_search: KnowledgeBaseSearch = Depends(get_knowledge_base_search)
) -> List[SearchResultSchema]:
    """Search scientific literature."""
    if request.search_type == "semantic":
        results = await kb_search.semantic_search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
    elif request.search_type == "keyword":
        results = await kb_search.keyword_search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
    else:  # hybrid
        results = await kb_search.hybrid_search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )

    return [
        SearchResultSchema(
            document_id=r.document_id,
            title=r.title,
            abstract=r.abstract,
            score=r.score,
            metadata=r.metadata,
            highlights=r.highlights
        )
        for r in results
    ]


@router.post("/ingest", status_code=202)
async def ingest_literature(
    request: LiteratureIngestRequest,
    db: AsyncSession = Depends(get_db),
    vector_store: VectorStore = Depends(get_vector_store),
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> dict:
    """Ingest literature from external sources."""
    ingestion = LiteratureIngestion(
        db=db,
        vector_store=vector_store,
        embedding_service=embedding_service,
        semantic_scholar_client=SemanticScholarClient(),
        crossref_client=CrossRefClient()
    )

    try:
        if request.source_type == "doi":
            paper_id = await ingestion.ingest_by_doi(request.source_value)
            return {
                "status": "completed",
                "paper_ids": [str(paper_id)],
                "count": 1
            }
        elif request.source_type == "query":
            paper_ids = await ingestion.ingest_by_query(
                request.source_value,
                max_results=request.max_results or 50
            )
            return {
                "status": "completed",
                "paper_ids": [str(pid) for pid in paper_ids],
                "count": len(paper_ids)
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source_type: {request.source_type}"
            )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/{paper_id}/similar", response_model=List[SearchResultSchema])
async def find_similar_papers(
    paper_id: str,
    top_k: int = 10,
    kb_search: KnowledgeBaseSearch = Depends(get_knowledge_base_search)
) -> List[SearchResultSchema]:
    """Find papers similar to given paper."""
    try:
        results = await kb_search.find_similar_papers(paper_id, top_k=top_k)
        return [
            SearchResultSchema(
                document_id=r.document_id,
                title=r.title,
                abstract=r.abstract,
                score=r.score,
                metadata=r.metadata,
                highlights=r.highlights
            )
            for r in results
        ]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
