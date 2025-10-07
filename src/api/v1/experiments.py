"""Experiment design and analysis endpoints."""

from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from src.core.database import get_db
from src.core.redis import get_redis
from src.schemas.experiment import (
    ExperimentDesignRequest,
    ExperimentDesignResponse,
    DataAnalysisRequest,
    DataAnalysisResponse,
    PowerAnalysisRequest,
    PowerAnalysisResponse,
    Experiment as ExperimentSchema
)
from src.services.experiment import ExperimentDesigner, DataAnalyzer
from src.services.llm import LLMService
from src.services.knowledge_base import VectorStore, EmbeddingService, KnowledgeBaseSearch

router = APIRouter()


async def get_experiment_designer(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
) -> ExperimentDesigner:
    """Get experiment designer dependency."""
    llm_service = LLMService(redis_client=redis)

    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    kb_search = KnowledgeBaseSearch(vector_store, embedding_service, db)

    return ExperimentDesigner(
        llm_service=llm_service,
        knowledge_base=kb_search,
        db=db
    )


async def get_data_analyzer(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis)
) -> DataAnalyzer:
    """Get data analyzer dependency."""
    llm_service = LLMService(redis_client=redis)
    return DataAnalyzer(llm_service=llm_service, db=db)


@router.post("/hypotheses/{hypothesis_id}/experiments/design", status_code=201)
async def design_experiment(
    hypothesis_id: UUID,
    request: ExperimentDesignRequest,
    designer: ExperimentDesigner = Depends(get_experiment_designer)
) -> Dict[str, Any]:
    """Design an experiment for a hypothesis.

    Args:
        hypothesis_id: Hypothesis UUID
        request: Experiment design parameters
        designer: Experiment designer service

    Returns:
        Designed experiment details
    """
    try:
        experiment = await designer.design_experiment(
            hypothesis_id=hypothesis_id,
            research_question=request.research_question,
            hypothesis_content=request.hypothesis_content,
            desired_power=request.desired_power,
            significance_level=request.significance_level,
            expected_effect_size=request.expected_effect_size,
            constraints=request.constraints,
            experimental_approach=request.experimental_approach
        )

        # Parse protocol to extract structured info
        import json
        protocol_data = json.loads(experiment.protocol) if experiment.protocol.startswith("{") else {"protocol": experiment.protocol}

        return {
            "experiment_id": experiment.id,
            "title": experiment.title,
            "protocol": protocol_data.get("protocol", experiment.protocol),
            "sample_size": experiment.sample_size,
            "power": experiment.power,
            "effect_size": experiment.effect_size,
            "significance_level": experiment.significance_level,
            "estimated_duration": protocol_data.get("estimated_duration"),
            "resource_requirements": protocol_data.get("resource_requirements", {}),
            "suggested_methods": protocol_data.get("methods", []),
            "potential_confounds": protocol_data.get("potential_confounds", []),
            "mitigation_strategies": protocol_data.get("mitigation_strategies", [])
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Experiment design failed: {str(e)}"
        )


@router.post("/experiments/{experiment_id}/analyze")
async def analyze_experiment(
    experiment_id: UUID,
    request: DataAnalysisRequest,
    analyzer: DataAnalyzer = Depends(get_data_analyzer)
) -> DataAnalysisResponse:
    """Analyze experimental data.

    Args:
        experiment_id: Experiment UUID
        request: Data analysis parameters
        analyzer: Data analyzer service

    Returns:
        Analysis results with statistics and visualizations
    """
    try:
        results = await analyzer.analyze_experiment_data(
            experiment_id=experiment_id,
            data=request.data,
            analysis_types=request.analysis_types,
            visualization_types=request.visualization_types
        )

        return DataAnalysisResponse(
            experiment_id=results["experiment_id"],
            descriptive_statistics=results["descriptive_statistics"],
            statistical_tests=[
                {
                    "test_name": test.get("test_name", "Unknown"),
                    "statistic": test.get("statistic", 0.0),
                    "p_value": test.get("p_value", 1.0),
                    "degrees_of_freedom": test.get("degrees_of_freedom"),
                    "confidence_interval": test.get("confidence_interval"),
                    "effect_size": test.get("effect_size"),
                    "interpretation": test.get("interpretation", "")
                }
                for test in results.get("statistical_tests", [])
            ],
            visualizations=[
                {
                    "visualization_type": viz.get("visualization_type", "unknown"),
                    "url": viz.get("url", ""),
                    "description": viz.get("description", ""),
                    "format": viz.get("format", "png")
                }
                for viz in results.get("visualizations", [])
            ],
            overall_interpretation=results["interpretation"],
            confidence_level=0.95,  # Default confidence level
            recommendations=results["recommendations"],
            limitations=["Sample size limitations", "Potential confounding variables"]
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data analysis failed: {str(e)}"
        )


@router.post("/power-analysis")
async def calculate_power(
    request: PowerAnalysisRequest,
    designer: ExperimentDesigner = Depends(get_experiment_designer)
) -> PowerAnalysisResponse:
    """Calculate statistical power or required sample size.

    Args:
        request: Power analysis parameters
        designer: Experiment designer service

    Returns:
        Power analysis results
    """
    try:
        if request.sample_size and request.effect_size:
            # Calculate power
            power = designer.calculate_power(
                effect_size=request.effect_size,
                sample_size=request.sample_size,
                alpha=request.significance_level
            )

            return PowerAnalysisResponse(
                effect_size=request.effect_size,
                sample_size=request.sample_size,
                power=power,
                significance_level=request.significance_level,
                recommendation=(
                    f"With sample size of {request.sample_size} per group and "
                    f"effect size of {request.effect_size}, you will achieve "
                    f"{power:.2%} power."
                )
            )

        elif request.power and request.effect_size:
            # Calculate sample size
            sample_size = designer._calculate_sample_size(
                effect_size=request.effect_size,
                power=request.power,
                alpha=request.significance_level
            )

            return PowerAnalysisResponse(
                effect_size=request.effect_size,
                sample_size=sample_size,
                power=request.power,
                significance_level=request.significance_level,
                recommendation=(
                    f"To achieve {request.power:.2%} power with effect size "
                    f"{request.effect_size}, you need {sample_size} participants "
                    f"per group."
                )
            )

        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either (sample_size + effect_size) or (power + effect_size)"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Power analysis failed: {str(e)}"
        )


@router.get("/experiments/{experiment_id}")
async def get_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db)
) -> ExperimentSchema:
    """Get experiment details.

    Args:
        experiment_id: Experiment UUID
        db: Database session

    Returns:
        Experiment details
    """
    from sqlalchemy import select
    from src.models.project import Experiment

    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return ExperimentSchema.model_validate(experiment)
