"""Multi-agent research endpoints."""

from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from src.core.database import get_db
from src.core.redis import get_redis
from src.services.llm.service import LLMService
from src.router.meta_router import MetaRouter
from src.router.types import ResearchTask
from src.agents.pool import AgentPool
from src.context.manager import ResearchContextManager

router = APIRouter()


async def get_meta_router(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
) -> MetaRouter:
    """Get meta-router dependency."""
    # For testing/fallback: use None for llm_service to trigger mock behavior
    try:
        llm_service = LLMService(redis_client=redis) if redis else None
    except Exception:
        llm_service = None

    agent_pool = AgentPool(llm_service, None)
    context_manager = ResearchContextManager(None, None)

    return MetaRouter(llm_service, agent_pool, context_manager)


@router.post("/research")
async def multi_agent_research(
    task: Dict[str, Any],
    meta_router: MetaRouter = Depends(get_meta_router)
):
    """Execute multi-agent research task."""

    research_task = ResearchTask(
        description=task["description"],
        task_type=task["task_type"],
        quality_target=task.get("quality_target", 0.8)
    )

    try:
        result = await meta_router.route_and_execute(research_task)

        return {
            "status": result.status,
            "quality_score": result.quality_score,
            "execution_time_ms": result.execution_time_ms,
            "agent_results": [
                {
                    "agent_id": r.agent_id,
                    "output": r.output,
                    "confidence": r.confidence
                }
                for r in result.agent_results
            ],
            "metadata": result.metadata
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hypothesis")
async def generate_hypothesis(
    request: Dict[str, Any],
    meta_router: MetaRouter = Depends(get_meta_router)
):
    """Generate research hypothesis using multi-agent system."""

    task = ResearchTask(
        description=request["research_question"],
        task_type="hypothesis_generation",
        prior_work=request.get("context"),
        quality_target=request.get("quality_target", 0.85)
    )

    result = await meta_router.route_and_execute(task)

    return {
        "status": result.status,
        "hypotheses": [r.output for r in result.agent_results],
        "quality_score": result.quality_score,
        "agents_used": result.metadata.get("agents_used", [])
    }
