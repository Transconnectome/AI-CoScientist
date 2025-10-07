"""Hypothesis generation endpoints."""

from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from src.core.database import get_db
from src.core.redis import get_redis
from src.schemas.project import HypothesisGenerateRequest, Hypothesis as HypothesisSchema
from src.services.hypothesis import HypothesisGenerator
from src.services.llm import LLMService
from src.services.knowledge_base import VectorStore, EmbeddingService, KnowledgeBaseSearch

router = APIRouter()


async def get_hypothesis_generator(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
) -> HypothesisGenerator:
    """Get hypothesis generator dependency."""
    llm_service = LLMService(redis_client=redis)

    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    kb_search = KnowledgeBaseSearch(vector_store, embedding_service, db)

    return HypothesisGenerator(
        llm_service=llm_service,
        knowledge_base=kb_search,
        db=db
    )


@router.post("/projects/{project_id}/hypotheses/generate", status_code=202)
async def generate_hypotheses(
    project_id: UUID,
    request: HypothesisGenerateRequest,
    background_tasks: BackgroundTasks,
    hypothesis_gen: HypothesisGenerator = Depends(get_hypothesis_generator)
) -> dict:
    """Generate hypotheses for a project (async task)."""
    # For now, execute synchronously (TODO: use Celery for async)
    try:
        hypotheses = await hypothesis_gen.generate_hypotheses(
            project_id=project_id,
            research_question=request.research_question,
            num_hypotheses=request.num_hypotheses,
            creativity_level=request.creativity_level,
            literature_context=request.literature_context
        )

        return {
            "status": "completed",
            "project_id": str(project_id),
            "hypotheses_generated": len(hypotheses),
            "hypothesis_ids": [str(h.id) for h in hypotheses]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Hypothesis generation failed: {str(e)}"
        )


@router.post("/hypotheses/{hypothesis_id}/validate")
async def validate_hypothesis(
    hypothesis_id: UUID,
    hypothesis_gen: HypothesisGenerator = Depends(get_hypothesis_generator)
) -> Dict[str, Any]:
    """Validate hypothesis for novelty and testability."""
    try:
        validation_result = await hypothesis_gen.validate_hypothesis(hypothesis_id)
        return validation_result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )
