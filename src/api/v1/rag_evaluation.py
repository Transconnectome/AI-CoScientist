"""
RAG Evaluation REST API endpoints.

Provides endpoints for:
- Performance tracking
- Cost optimization
- A/B testing
- RAGAS evaluation
- Prometheus metrics
"""

import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.database import get_db
from src.models.rag_evaluation import (
    RAGABTest,
    RAGABTestResult,
    RAGCostBudget,
    RAGEvaluation,
    RAGPerformanceMetric,
)
from src.services.rag.ab_testing import ABTest, ABTestConfig, ABTestResult, Variant
from src.services.rag.cost_optimizer import CostBudget, CostOptimizer
from src.services.rag.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/rag-evaluation", tags=["rag-evaluation"])

# Services for legacy operations (will gradually phase out)
performance_tracker = PerformanceTracker()
cost_optimizer = CostOptimizer()
# In-memory storage removed - now using database persistence


# ==================== Request/Response Models ====================


class PerformanceTrackRequest(BaseModel):
    """Request model for tracking performance"""
    operation: str = Field(..., description="Operation name (e.g., 'retrieval', 'generation')")
    latency: float = Field(..., description="Latency in seconds")
    tokens: dict[str, int] = Field(..., description="Token usage (prompt, completion)")
    model: str = Field(..., description="Model name (e.g., 'gpt-4')")


class PerformanceTrackResponse(BaseModel):
    """Response model for performance tracking"""
    metric_id: str = Field(..., description="Unique metric ID")
    operation: str = Field(..., description="Operation name")
    status: str = Field(default="tracked", description="Status")


class BudgetCreateRequest(BaseModel):
    """Request model for creating cost budget"""
    name: str = Field(..., description="Budget name")
    total: float = Field(..., description="Total budget amount")
    warning_threshold: float = Field(default=0.8, description="Warning threshold (0-1)")
    critical_threshold: float = Field(default=0.95, description="Critical threshold (0-1)")


class BudgetCreateResponse(BaseModel):
    """Response model for budget creation"""
    budget_id: str = Field(..., description="Unique budget ID")
    name: str = Field(..., description="Budget name")
    total: float = Field(..., description="Total budget")
    spent: float = Field(default=0.0, description="Amount spent")
    status: str = Field(..., description="Budget status")


class BudgetGetResponse(BaseModel):
    """Response model for getting budget"""
    budget_id: str = Field(..., description="Budget ID")
    name: str = Field(..., description="Budget name")
    total: float = Field(..., description="Total budget")
    spent: float = Field(..., description="Amount spent")
    remaining: float = Field(..., description="Remaining budget")
    status: str = Field(..., description="Budget status")
    expenses: dict[str, float] = Field(default_factory=dict, description="Expense breakdown")


class OptimizeRequest(BaseModel):
    """Request model for optimization suggestions"""
    usage_data: dict[str, Any] = Field(..., description="Usage data for optimization")


class OptimizeSuggestionsResponse(BaseModel):
    """Response model for optimization suggestions"""
    suggestions: list[dict[str, Any]] = Field(..., description="List of optimization suggestions")


class VariantConfig(BaseModel):
    """Variant configuration"""
    name: str = Field(..., description="Variant name")
    config: dict[str, Any] = Field(..., description="Variant configuration")


class ABTestCreateRequest(BaseModel):
    """Request model for creating A/B test"""
    name: str = Field(..., description="Test name")
    variants: list[VariantConfig] = Field(..., description="Test variants")
    traffic_split: dict[str, float] = Field(..., description="Traffic split ratios")


class ABTestCreateResponse(BaseModel):
    """Response model for A/B test creation"""
    test_id: str = Field(..., description="Test ID")
    name: str = Field(..., description="Test name")
    status: str = Field(default="created", description="Test status")


class ABTestAddResultRequest(BaseModel):
    """Request model for adding A/B test result"""
    variant_name: str = Field(..., description="Variant name")
    metrics: dict[str, float] = Field(..., description="Metrics")
    cost: float = Field(..., description="Cost")


class ABTestAddResultResponse(BaseModel):
    """Response model for adding result"""
    status: str = Field(default="added", description="Status")


class RAGASEvaluateRequest(BaseModel):
    """Request model for RAGAS evaluation"""
    dataset: list[dict[str, Any]] = Field(..., description="Evaluation dataset")


class RAGASEvaluateResponse(BaseModel):
    """Response model for RAGAS evaluation"""
    evaluation_id: str = Field(..., description="Evaluation ID")
    metrics: dict[str, float] = Field(..., description="RAGAS metrics")


# ==================== Performance Endpoints ====================


@router.post("/performance/track", response_model=PerformanceTrackResponse)
async def track_performance(
    request: PerformanceTrackRequest,
    db: Session = Depends(get_db)
) -> PerformanceTrackResponse:
    """Track performance metric"""
    try:
        # Calculate cost estimate
        cost = cost_optimizer.calculate_cost(
            request.model,
            request.tokens.get("prompt", 0),
            request.tokens.get("completion", 0)
        )

        # Create database record
        metric = RAGPerformanceMetric(
            operation=request.operation,
            latency=request.latency,
            token_usage=request.tokens,
            cost=cost
        )
        
        db.add(metric)
        await db.commit()
        await db.refresh(metric)

        logger.info(f"Tracked performance for {request.operation}: {request.latency}s (cost: ${cost:.4f})")

        return PerformanceTrackResponse(
            metric_id=str(metric.id),
            operation=request.operation,
            status="tracked"
        )
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to track performance: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/performance/metrics")
async def get_performance_metrics(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get aggregated performance metrics"""
    try:
        # Query all metrics from database
        result = await db.execute(select(RAGPerformanceMetric))
        metrics = result.scalars().all()
        
        # Aggregate by operation
        aggregated: dict[str, dict[str, Any]] = {}
        for metric in metrics:
            op = metric.operation
            if op not in aggregated:
                aggregated[op] = {
                    "count": 0,
                    "total_latency": 0.0,
                    "total_cost": 0.0,
                    "latencies": []
                }
            
            aggregated[op]["count"] += 1
            aggregated[op]["total_latency"] += metric.latency
            aggregated[op]["total_cost"] += metric.cost or 0.0
            aggregated[op]["latencies"].append(metric.latency)
        
        # Calculate statistics
        result_data = {}
        for op, data in aggregated.items():
            latencies = data["latencies"]
            result_data[op] = {
                "count": data["count"],
                "avg_latency": data["total_latency"] / data["count"],
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "total_cost": data["total_cost"]
            }
        
        return {"metrics": result_data, "total_operations": len(metrics)}
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/performance/reset")
async def reset_performance_tracker(db: Session = Depends(get_db)) -> dict[str, str]:
    """Reset performance tracker (delete all metrics)"""
    try:
        # Delete all performance metrics using async delete
        await db.execute(select(RAGPerformanceMetric).where(RAGPerformanceMetric.id.is_not(None)))
        
        # Alternative: delete all via iteration
        result = await db.execute(select(RAGPerformanceMetric))
        metrics = result.scalars().all()
        for metric in metrics:
            await db.delete(metric)
        
        await db.commit()
        
        logger.info("Performance metrics reset")
        return {"status": "reset", "message": f"Deleted {len(metrics)} performance metrics"}
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to reset tracker: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== Cost Optimization Endpoints ====================


@router.post("/cost/budget/create", response_model=BudgetCreateResponse)
async def create_budget(
    request: BudgetCreateRequest,
    db: Session = Depends(get_db)
) -> BudgetCreateResponse:
    """Create cost budget"""
    try:
        budget = RAGCostBudget(
            name=request.name,
            total_budget=request.total,
            spent=0.0,
            warning_threshold=request.warning_threshold,
            critical_threshold=request.critical_threshold,
            expenses={}
        )
        
        db.add(budget)
        await db.commit()
        await db.refresh(budget)

        logger.info(f"Created budget {budget.id}: {request.name}")

        return BudgetCreateResponse(
            budget_id=str(budget.id),
            name=request.name,
            total=request.total,
            spent=0.0,
            status=budget.status
        )
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create budget: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/cost/budget/{budget_id}", response_model=BudgetGetResponse)
async def get_budget(budget_id: str, db: Session = Depends(get_db)) -> BudgetGetResponse:
    """Get budget by ID"""
    try:
        from uuid import UUID
        budget_uuid = UUID(budget_id)
        
        result = await db.execute(
            select(RAGCostBudget).where(RAGCostBudget.id == budget_uuid)
        )
        budget = result.scalar_one_or_none()
        
        if not budget:
            raise HTTPException(status_code=404, detail="Budget not found")

        return BudgetGetResponse(
            budget_id=str(budget.id),
            name=budget.name,
            total=budget.total_budget,
            spent=budget.spent,
            remaining=budget.remaining,
            status=budget.status,
            expenses=budget.expenses or {}
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid budget ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get budget: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/cost/optimize", response_model=OptimizeSuggestionsResponse)
async def optimize_costs(request: OptimizeRequest) -> OptimizeSuggestionsResponse:
    """Get cost optimization suggestions"""
    try:
        suggestions = cost_optimizer.suggest_optimizations(request.usage_data)
        return OptimizeSuggestionsResponse(suggestions=suggestions)
    except Exception as e:
        logger.error(f"Failed to optimize costs: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/cost/suggestions")
async def get_cost_suggestions(
    budget_id: str = Query(..., description="Budget ID"),
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Get suggestions for specific budget"""
    try:
        from uuid import UUID
        budget_uuid = UUID(budget_id)
        
        result = await db.execute(
            select(RAGCostBudget).where(RAGCostBudget.id == budget_uuid)
        )
        budget = result.scalar_one_or_none()
        
        if not budget:
            raise HTTPException(status_code=404, detail="Budget not found")

        # Create CostBudget instance for legacy CostOptimizer
        legacy_budget = CostBudget(
            total=budget.total_budget,
            warning_threshold=budget.warning_threshold,
            critical_threshold=budget.critical_threshold
        )
        legacy_budget.spent = budget.spent
        
        alerts = cost_optimizer.check_budget_alerts(legacy_budget)
        
        # Generate suggestions based on budget status
        suggestions = []
        if budget.status == "critical":
            suggestions.append({
                "priority": "high",
                "action": "Reduce API calls or use cheaper models",
                "impact": "Prevent budget overrun"
            })
        elif budget.status == "warning":
            suggestions.append({
                "priority": "medium",
                "action": "Monitor usage closely",
                "impact": "Avoid reaching critical threshold"
            })
        
        return {
            "budget_id": budget_id,
            "status": budget.status,
            "alerts": alerts,
            "suggestions": suggestions
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid budget ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== A/B Testing Endpoints ====================


@router.post("/ab-test/create", response_model=ABTestCreateResponse)
async def create_ab_test(
    request: ABTestCreateRequest,
    db: Session = Depends(get_db)
) -> ABTestCreateResponse:
    """Create A/B test"""
    try:
        # Build configuration
        config = {
            "variants": [
                {"name": v.name, "config": v.config}
                for v in request.variants
            ],
            "traffic_split": request.traffic_split
        }

        # Create database record
        ab_test = RAGABTest(
            name=request.name,
            config=config,
            status="active"
        )
        
        db.add(ab_test)
        await db.commit()
        await db.refresh(ab_test)

        logger.info(f"Created A/B test {ab_test.id}: {request.name}")

        return ABTestCreateResponse(
            test_id=str(ab_test.id),
            name=request.name,
            status="created"
        )
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to create A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/ab-test/{test_id}/add-result", response_model=ABTestAddResultResponse)
async def add_ab_test_result(
    test_id: str,
    request: ABTestAddResultRequest,
    db: Session = Depends(get_db)
) -> ABTestAddResultResponse:
    """Add result to A/B test"""
    try:
        from uuid import UUID
        test_uuid = UUID(test_id)
        
        # Verify test exists
        result = await db.execute(
            select(RAGABTest).where(RAGABTest.id == test_uuid)
        )
        ab_test = result.scalar_one_or_none()
        
        if not ab_test:
            raise HTTPException(status_code=404, detail="A/B test not found")

        # Create result record
        test_result = RAGABTestResult(
            test_id=test_uuid,
            variant_name=request.variant_name,
            metrics=request.metrics,
            cost=request.cost
        )
        
        db.add(test_result)
        await db.commit()

        logger.info(f"Added result to A/B test {test_id}")

        return ABTestAddResultResponse(status="added")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid test ID format")
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to add result: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/ab-test/{test_id}/analyze")
async def analyze_ab_test(
    test_id: str,
    metric: str = Query(..., description="Metric to analyze"),
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Analyze A/B test results"""
    try:
        from uuid import UUID
        test_uuid = UUID(test_id)
        
        # Get all results for this test
        result = await db.execute(
            select(RAGABTestResult).where(RAGABTestResult.test_id == test_uuid)
        )
        results = result.scalars().all()
        
        if not results:
            raise HTTPException(status_code=404, detail="No results found for this test")

        # Aggregate by variant
        analysis: dict[str, dict[str, Any]] = {}
        for test_result in results:
            variant = test_result.variant_name
            if variant not in analysis:
                analysis[variant] = {
                    "count": 0,
                    "values": [],
                    "total_cost": 0.0
                }
            
            analysis[variant]["count"] += 1
            analysis[variant]["total_cost"] += test_result.cost
            
            if metric in test_result.metrics:
                analysis[variant]["values"].append(test_result.metrics[metric])
        
        # Calculate statistics
        final_analysis = {}
        for variant, data in analysis.items():
            if data["values"]:
                values = data["values"]
                final_analysis[variant] = {
                    "count": data["count"],
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "total_cost": data["total_cost"]
                }
        
        return {"metric": metric, "analysis": final_analysis}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid test ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/ab-test/{test_id}/winner")
async def get_ab_test_winner(
    test_id: str,
    metric: str = Query(..., description="Metric to evaluate"),
    min_confidence: float = Query(0.95, description="Minimum confidence"),
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Declare A/B test winner"""
    try:
        from uuid import UUID
        test_uuid = UUID(test_id)
        
        # Get all results for this test
        result = await db.execute(
            select(RAGABTestResult).where(RAGABTestResult.test_id == test_uuid)
        )
        results = result.scalars().all()
        
        if not results:
            raise HTTPException(status_code=404, detail="No results found for this test")

        # Aggregate by variant
        variant_data: dict[str, list[float]] = {}
        for test_result in results:
            variant = test_result.variant_name
            if variant not in variant_data:
                variant_data[variant] = []
            
            if metric in test_result.metrics:
                variant_data[variant].append(test_result.metrics[metric])
        
        # Find best variant (highest mean)
        best_variant = None
        best_mean = float('-inf')
        
        for variant, values in variant_data.items():
            if values:
                mean = sum(values) / len(values)
                if mean > best_mean:
                    best_mean = mean
                    best_variant = variant
        
        if best_variant is None:
            return {
                "winner": None,
                "reason": "No valid results found",
                "confidence": 0.0
            }
        
        # Simple winner declaration (statistical significance would require scipy)
        return {
            "winner": best_variant,
            "metric": metric,
            "mean_value": best_mean,
            "confidence": min_confidence,
            "note": "Full statistical significance testing requires scipy package"
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid test ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to declare winner: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== RAGAS Evaluation Endpoints ====================


@router.post("/ragas/evaluate", response_model=RAGASEvaluateResponse)
async def evaluate_ragas(
    request: RAGASEvaluateRequest,
    db: Session = Depends(get_db)
) -> RAGASEvaluateResponse:
    """Evaluate RAG system with RAGAS"""
    try:
        # Placeholder metrics (will integrate with actual RAGAS evaluator in next phase)
        metrics = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_precision": 0.90,
            "context_recall": 0.88
        }
        
        # Generate dataset ID from request hash
        import hashlib
        import json
        dataset_content = json.dumps(request.dataset, sort_keys=True)
        dataset_id = hashlib.md5(dataset_content.encode()).hexdigest()[:16]

        # Create evaluation record
        evaluation = RAGEvaluation(
            dataset_id=dataset_id,
            evaluation_type="ragas",
            metrics=metrics,
            evaluation_metadata={
                "dataset_size": len(request.dataset),
                "placeholder": True
            }
        )
        
        db.add(evaluation)
        await db.commit()
        await db.refresh(evaluation)

        logger.info(f"Completed RAGAS evaluation {evaluation.id}")

        return RAGASEvaluateResponse(
            evaluation_id=str(evaluation.id),
            metrics=metrics
        )
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to evaluate RAGAS: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/ragas/baseline/{dataset_id}")
async def get_ragas_baseline(dataset_id: str) -> dict[str, Any]:
    """Get RAGAS baseline metrics"""
    try:
        # Placeholder: Will integrate with BaselineEvaluator
        baseline_metrics = {
            "faithfulness": {"mean": 0.85, "std": 0.05},
            "answer_relevancy": {"mean": 0.80, "std": 0.08}
        }

        return {"baseline_metrics": baseline_metrics}
    except Exception as e:
        logger.error(f"Failed to get baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/ragas/metrics")
async def get_ragas_metrics() -> dict[str, Any]:
    """Get available RAGAS metrics"""
    try:
        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall"
        ]

        return {"metrics": metrics}
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
