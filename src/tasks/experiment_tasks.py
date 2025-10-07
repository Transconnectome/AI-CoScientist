"""Experiment-related Celery tasks."""

from uuid import UUID
from typing import Any, Dict, List, Optional

from src.core.celery_app import celery_app
from src.core.database import AsyncSessionLocal
from src.core.redis import get_redis_client
from src.services.experiment import ExperimentDesigner, DataAnalyzer
from src.services.llm import LLMService
from src.services.knowledge_base import VectorStore, EmbeddingService, KnowledgeBaseSearch


@celery_app.task(name="design_experiment")
def design_experiment_task(
    hypothesis_id: str,
    research_question: str,
    hypothesis_content: str,
    desired_power: float = 0.8,
    significance_level: float = 0.05,
    expected_effect_size: Optional[float] = None,
    constraints: Optional[Dict[str, Any]] = None,
    experimental_approach: Optional[str] = None
) -> Dict[str, Any]:
    """Background task for experiment design.

    Args:
        hypothesis_id: Hypothesis UUID as string
        research_question: Research question
        hypothesis_content: Hypothesis text
        desired_power: Desired statistical power
        significance_level: Alpha level
        expected_effect_size: Expected effect size
        constraints: Resource constraints
        experimental_approach: Suggested approach

    Returns:
        Experiment design results
    """
    import asyncio

    async def _design():
        async with AsyncSessionLocal() as db:
            redis = await get_redis_client()
            llm_service = LLMService(redis_client=redis)

            vector_store = VectorStore()
            embedding_service = EmbeddingService()
            kb_search = KnowledgeBaseSearch(vector_store, embedding_service, db)

            designer = ExperimentDesigner(
                llm_service=llm_service,
                knowledge_base=kb_search,
                db=db
            )

            experiment = await designer.design_experiment(
                hypothesis_id=UUID(hypothesis_id),
                research_question=research_question,
                hypothesis_content=hypothesis_content,
                desired_power=desired_power,
                significance_level=significance_level,
                expected_effect_size=expected_effect_size,
                constraints=constraints,
                experimental_approach=experimental_approach
            )

            return {
                "experiment_id": str(experiment.id),
                "title": experiment.title,
                "status": "completed"
            }

    return asyncio.run(_design())


@celery_app.task(name="analyze_experiment")
def analyze_experiment_task(
    experiment_id: str,
    data: Dict[str, Any],
    analysis_types: Optional[List[str]] = None,
    visualization_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Background task for experiment data analysis.

    Args:
        experiment_id: Experiment UUID as string
        data: Experimental data
        analysis_types: Types of analyses
        visualization_types: Types of visualizations

    Returns:
        Analysis results
    """
    import asyncio

    async def _analyze():
        async with AsyncSessionLocal() as db:
            redis = await get_redis_client()
            llm_service = LLMService(redis_client=redis)

            analyzer = DataAnalyzer(llm_service=llm_service, db=db)

            results = await analyzer.analyze_experiment_data(
                experiment_id=UUID(experiment_id),
                data=data,
                analysis_types=analysis_types,
                visualization_types=visualization_types
            )

            return {
                "experiment_id": str(results["experiment_id"]),
                "status": "completed",
                "interpretation": results["interpretation"]
            }

    return asyncio.run(_analyze())
