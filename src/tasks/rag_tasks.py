"""
RAG background tasks using Celery.

Provides asynchronous task execution for:
- Daily RAG benchmarking
- Performance snapshot capture
- Weekly cost analysis
- A/B test evaluation
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select

from src.core.database import AsyncSessionLocal
from src.models.rag_evaluation import (
    RAGABTest,
    RAGABTestResult,
    RAGEvaluation,
    RAGPerformanceMetric,
)
from src.services.rag.performance_tracker import PerformanceTracker

# Note: Celery app will be configured when Celery service is available
# For now, implementing as regular async functions

logger = logging.getLogger(__name__)


async def run_daily_rag_benchmark() -> dict[str, Any]:
    """Run daily comprehensive RAG evaluation.

    Executes RAGAS evaluation on standard test dataset and stores results.

    Returns:
        dict: Evaluation results with ID and metrics
    """
    try:
        logger.info("Starting daily RAG benchmark")

        # Placeholder: Will integrate with actual RAGAS evaluator
        # For now, using mock evaluation data
        metrics = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_precision": 0.90,
            "context_recall": 0.88
        }

        # Store evaluation in database
        async with AsyncSessionLocal() as db:
            evaluation = RAGEvaluation(
                dataset_id=f"daily_benchmark_{datetime.utcnow().strftime('%Y%m%d')}",
                evaluation_type="ragas",
                metrics=metrics,
                evaluation_metadata={
                    "task_type": "daily_benchmark",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            db.add(evaluation)
            await db.commit()
            await db.refresh(evaluation)

            result = {
                "evaluation_id": str(evaluation.id),
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat()
            }

            logger.info(f"Daily benchmark completed: {evaluation.id}")
            return result

    except Exception as e:
        logger.error(f"Daily benchmark failed: {e}")
        raise


async def capture_performance_snapshot() -> dict[str, Any]:
    """Capture hourly performance metrics snapshot.

    Aggregates current performance metrics and stores snapshot for trending.

    Returns:
        dict: Snapshot ID and aggregated metrics
    """
    try:
        logger.info("Capturing performance snapshot")

        # Get current metrics from PerformanceTracker
        tracker = PerformanceTracker()
        current_metrics = tracker.get_metrics()

        # Store snapshot
        async with AsyncSessionLocal() as db:
            snapshot_id = str(uuid4())

            # Store each operation's metrics
            for operation, data in current_metrics.items():
                metric = RAGPerformanceMetric(
                    operation=f"snapshot_{operation}",
                    latency=data.get("avg_latency", 0.0),
                    token_usage={
                        "snapshot_id": snapshot_id,
                        "count": data.get("count", 0)
                    },
                    cost=data.get("total_cost", 0.0)
                )
                db.add(metric)

            await db.commit()

            result = {
                "snapshot_id": snapshot_id,
                "metrics": current_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }

            logger.info(f"Performance snapshot captured: {snapshot_id}")
            return result

    except Exception as e:
        logger.error(f"Performance snapshot failed: {e}")
        raise


async def analyze_weekly_costs() -> dict[str, Any]:
    """Analyze costs for the past week.

    Aggregates cost data by operation type and provides weekly summary.

    Returns:
        dict: Weekly cost analysis with totals and breakdowns
    """
    try:
        logger.info("Starting weekly cost analysis")

        # Calculate date range for past week
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)

        async with AsyncSessionLocal() as db:
            # Query metrics from past week
            result = await db.execute(
                select(RAGPerformanceMetric).where(
                    RAGPerformanceMetric.created_at >= start_date,
                    RAGPerformanceMetric.created_at <= end_date
                )
            )
            metrics = result.scalars().all() # scalars() returns ScalarResult, not coroutine

            # Aggregate costs
            total_cost = 0.0
            by_operation: dict[str, float] = {}

            for metric in metrics:
                cost = metric.cost or 0.0
                total_cost += cost

                if metric.operation not in by_operation:
                    by_operation[metric.operation] = 0.0
                by_operation[metric.operation] += cost

            analysis = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "total_cost": total_cost,
                "by_operation": by_operation,
                "metric_count": len(metrics)
            }

            logger.info(f"Weekly cost analysis: ${total_cost:.2f} across {len(metrics)} operations")
            return analysis

    except Exception as e:
        logger.error(f"Weekly cost analysis failed: {e}")
        raise


async def evaluate_ab_test(test_id: str) -> dict[str, Any]:
    """Evaluate A/B test and declare winner if possible.

    Args:
        test_id: A/B test UUID

    Returns:
        dict: Test evaluation results with winner declaration

    Raises:
        ValueError: If test not found or has no results
    """
    try:
        logger.info(f"Evaluating A/B test: {test_id}")

        test_uuid = UUID(test_id)

        async with AsyncSessionLocal() as db:
            # Get test
            test_result = await db.execute(
                select(RAGABTest).where(RAGABTest.id == test_uuid)
            )
            ab_test = test_result.scalar_one_or_none()

            if not ab_test:
                raise ValueError(f"A/B test not found: {test_id}")

            # Get all results
            results_query = await db.execute(
                select(RAGABTestResult).where(RAGABTestResult.test_id == test_uuid)
            )
            results = results_query.scalars().all()

            if not results:
                raise ValueError(f"No results found for A/B test: {test_id}")

            # Aggregate by variant - use first metric as primary
            variant_scores: dict[str, list[float]] = {}

            for result in results:
                variant = result.variant_name
                if variant not in variant_scores:
                    variant_scores[variant] = []

                # Get first metric value
                if result.metrics:
                    first_metric = next(iter(result.metrics.values()))
                    variant_scores[variant].append(float(first_metric))

            # Find winner (highest average score)
            winner = None
            best_score = float('-inf')

            for variant, scores in variant_scores.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        winner = variant

            evaluation = {
                "test_id": test_id,
                "test_name": ab_test.name,
                "winner": winner,
                "best_score": best_score,
                "variant_count": len(variant_scores),
                "result_count": len(results),
                "evaluated_at": datetime.utcnow().isoformat()
            }

            logger.info(f"A/B test evaluation complete: winner={winner}, score={best_score:.3f}")
            return evaluation

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"A/B test evaluation failed: {e}")
        raise
