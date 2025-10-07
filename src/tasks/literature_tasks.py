"""Literature-related Celery tasks."""

from typing import Dict, Any, Optional

from src.core.celery_app import celery_app
from src.core.database import AsyncSessionLocal
from src.services.knowledge_base import VectorStore, EmbeddingService
from src.services.knowledge_base.ingestion import LiteratureIngestion
from src.services.external import SemanticScholarClient, CrossRefClient


@celery_app.task(name="ingest_literature")
def ingest_literature_task(
    source_type: str,
    source_value: str,
    max_results: Optional[int] = 50
) -> Dict[str, Any]:
    """Background task for literature ingestion.

    Args:
        source_type: Source type (doi or query)
        source_value: DOI or search query
        max_results: Maximum results for query search

    Returns:
        Ingestion results
    """
    import asyncio

    async def _ingest():
        async with AsyncSessionLocal() as db:
            vector_store = VectorStore()
            embedding_service = EmbeddingService()

            ingestion = LiteratureIngestion(
                db=db,
                vector_store=vector_store,
                embedding_service=embedding_service,
                semantic_scholar_client=SemanticScholarClient(),
                crossref_client=CrossRefClient()
            )

            if source_type == "doi":
                paper_id = await ingestion.ingest_by_doi(source_value)
                return {
                    "status": "completed",
                    "paper_ids": [str(paper_id)],
                    "count": 1
                }
            elif source_type == "query":
                paper_ids = await ingestion.ingest_by_query(
                    source_value,
                    max_results=max_results or 50
                )
                return {
                    "status": "completed",
                    "paper_ids": [str(pid) for pid in paper_ids],
                    "count": len(paper_ids)
                }
            else:
                raise ValueError(f"Invalid source_type: {source_type}")

    return asyncio.run(_ingest())
