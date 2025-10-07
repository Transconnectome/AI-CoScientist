"""Hypothesis-related Celery tasks."""

from uuid import UUID
from typing import Any, Dict, List, Optional

from src.core.celery_app import celery_app
from src.core.database import AsyncSessionLocal
from src.core.redis import get_redis_client
from src.services.hypothesis import HypothesisGenerator
from src.services.llm import LLMService
from src.services.knowledge_base import VectorStore, EmbeddingService, KnowledgeBaseSearch


@celery_app.task(name="generate_hypotheses")
def generate_hypotheses_task(
    project_id: str,
    research_question: str,
    num_hypotheses: int = 5,
    creativity_level: float = 0.7,
    literature_context: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Background task for hypothesis generation.

    Args:
        project_id: Project UUID as string
        research_question: Research question
        num_hypotheses: Number of hypotheses to generate
        creativity_level: Temperature for generation
        literature_context: Literature context papers

    Returns:
        Generation results
    """
    import asyncio

    async def _generate():
        async with AsyncSessionLocal() as db:
            redis = await get_redis_client()
            llm_service = LLMService(redis_client=redis)

            vector_store = VectorStore()
            embedding_service = EmbeddingService()
            kb_search = KnowledgeBaseSearch(vector_store, embedding_service, db)

            generator = HypothesisGenerator(
                llm_service=llm_service,
                knowledge_base=kb_search,
                db=db
            )

            hypotheses = await generator.generate_hypotheses(
                project_id=UUID(project_id),
                research_question=research_question,
                num_hypotheses=num_hypotheses,
                creativity_level=creativity_level,
                literature_context=literature_context
            )

            return {
                "project_id": project_id,
                "status": "completed",
                "hypotheses_generated": len(hypotheses),
                "hypothesis_ids": [str(h.id) for h in hypotheses]
            }

    return asyncio.run(_generate())


@celery_app.task(name="validate_hypothesis")
def validate_hypothesis_task(hypothesis_id: str) -> Dict[str, Any]:
    """Background task for hypothesis validation.

    Args:
        hypothesis_id: Hypothesis UUID as string

    Returns:
        Validation results
    """
    import asyncio

    async def _validate():
        async with AsyncSessionLocal() as db:
            redis = await get_redis_client()
            llm_service = LLMService(redis_client=redis)

            vector_store = VectorStore()
            embedding_service = EmbeddingService()
            kb_search = KnowledgeBaseSearch(vector_store, embedding_service, db)

            generator = HypothesisGenerator(
                llm_service=llm_service,
                knowledge_base=kb_search,
                db=db
            )

            results = await generator.validate_hypothesis(UUID(hypothesis_id))

            return {
                "hypothesis_id": hypothesis_id,
                "status": "completed",
                "novelty_score": results.get("novelty_score", 0.0),
                "testability_score": results.get("testability_score", 0.0)
            }

    return asyncio.run(_validate())
