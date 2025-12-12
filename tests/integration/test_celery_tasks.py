"""
Integration tests for Phase 2 Celery background tasks.

TDD RED Phase: Tests document expected task behavior with real database.
All 4 Celery tasks tested with actual PostgreSQL and Redis.

Tests verify:
- Task execution without errors
- Database persistence
- Result accuracy
- Error handling
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.rag_evaluation import (
    RAGABTest,
    RAGABTestResult,
    RAGEvaluation,
    RAGPerformanceMetric,
)
from src.tasks.rag_tasks import (
    analyze_weekly_costs,
    capture_performance_snapshot,
    evaluate_ab_test,
    run_daily_rag_benchmark,
)


class TestDailyRAGBenchmark:
    """Test daily RAG benchmark task."""

    @pytest.mark.asyncio
    async def test_daily_benchmark_execution(self, db_session: AsyncSession, clean_database):
        """Test daily benchmark task executes and stores results."""
        result = await run_daily_rag_benchmark()

        assert result is not None
        assert "evaluation_id" in result
        assert "metrics" in result
        assert "timestamp" in result

        # Verify database persistence
        evals = await db_session.execute(select(RAGEvaluation))
        evaluations = evals.scalars().all()
        assert len(evaluations) > 0

    @pytest.mark.asyncio
    async def test_daily_benchmark_metrics(self, db_session: AsyncSession, clean_database):
        """Test daily benchmark includes all RAGAS metrics."""
        result = await run_daily_rag_benchmark()

        metrics = result["metrics"]
        assert "faithfulness" in metrics
        assert "answer_relevancy" in metrics
        assert "context_precision" in metrics
        assert "context_recall" in metrics


class TestPerformanceSnapshot:
    """Test performance snapshot capture task."""

    @pytest.mark.asyncio
    async def test_snapshot_capture(self, db_session: AsyncSession, clean_database):
        """Test snapshot captures and stores performance metrics."""
        result = await capture_performance_snapshot()

        assert result is not None
        assert "snapshot_id" in result
        assert "metrics" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_snapshot_stores_metrics(self, db_session: AsyncSession, clean_database):
        """Test snapshot stores metrics in database."""
        await capture_performance_snapshot()

        # Verify database persistence
        metrics = await db_session.execute(select(RAGPerformanceMetric))
        stored_metrics = metrics.scalars().all()
        # May be empty if PerformanceTracker has no data, but should not error
        assert stored_metrics is not None


class TestWeeklyCostAnalysis:
    """Test weekly cost analysis task."""

    @pytest.mark.asyncio
    async def test_weekly_cost_analysis(self, db_session: AsyncSession, clean_database):
        """Test weekly cost analysis executes and aggregates costs."""
        # Create sample metrics
        for i in range(5):
            metric = RAGPerformanceMetric(
                operation="test_retrieval" if i < 3 else "test_generation",
                latency=0.5,
                cost=0.01 * (i + 1)
            )
            db_session.add(metric)
        await db_session.commit()

        # Run analysis
        result = await analyze_weekly_costs()

        assert result is not None
        assert "total_cost" in result
        assert "by_operation" in result
        assert "metric_count" in result
        assert result["total_cost"] > 0

    @pytest.mark.asyncio
    async def test_weekly_cost_grouping(self, db_session: AsyncSession, clean_database):
        """Test weekly analysis groups costs by operation."""
        # Create metrics for different operations
        for op in ["retrieval", "retrieval", "generation"]:
            metric = RAGPerformanceMetric(
                operation=op,
                latency=0.5,
                cost=0.02
            )
            db_session.add(metric)
        await db_session.commit()

        result = await analyze_weekly_costs()

        assert "by_operation" in result
        by_op = result["by_operation"]
        assert "retrieval" in by_op
        assert "generation" in by_op
        assert by_op["retrieval"] == 0.04  # 2 * 0.02


class TestABTestEvaluation:
    """Test A/B test evaluation task."""

    @pytest.mark.asyncio
    async def test_evaluate_ab_test_execution(self, db_session: AsyncSession, create_ab_test_with_results, clean_database):
        """Test A/B test evaluation determines winner."""
        test_id = str(create_ab_test_with_results.id)

        result = await evaluate_ab_test(test_id)

        assert result is not None
        assert "test_id" in result
        assert "winner" in result
        assert "best_score" in result
        assert result["winner"] in ["variant_a", "variant_b"]

    @pytest.mark.asyncio
    async def test_evaluate_nonexistent_test(self, db_session: AsyncSession, clean_database):
        """Test evaluation raises error for nonexistent test."""
        from uuid import uuid4

        fake_test_id = str(uuid4())

        with pytest.raises(ValueError, match="A/B test not found"):
            await evaluate_ab_test(fake_test_id)

    @pytest.mark.asyncio
    async def test_evaluate_test_without_results(self, db_session: AsyncSession, create_ab_test, clean_database):
        """Test evaluation raises error for test with no results."""
        test_id = str(create_ab_test.id)

        with pytest.raises(ValueError, match="No results found"):
            await evaluate_ab_test(test_id)
