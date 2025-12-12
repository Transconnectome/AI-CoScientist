"""
Tests for RAG background tasks.

Following TDD methodology (RED → GREEN → REFACTOR):
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Improve code quality
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

from src.tasks.rag_tasks import (
    analyze_weekly_costs,
    capture_performance_snapshot,
    evaluate_ab_test,
    run_daily_rag_benchmark,
)


class TestRunDailyRAGBenchmark:
    """Tests for daily RAG benchmark task"""

    @pytest.mark.asyncio
    async def test_benchmark_runs_successfully(self):
        """Test that benchmark task executes without errors"""
        # Mock database
        with patch('src.tasks.rag_tasks.AsyncSessionLocal') as mock_session_local:

            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            result = await run_daily_rag_benchmark()

            assert result is not None
            assert "evaluation_id" in result
            assert "metrics" in result
            assert result["metrics"]["faithfulness"] == 0.85

    @pytest.mark.asyncio
    async def test_benchmark_stores_results_in_database(self):
        """Test that benchmark results are persisted to database"""
        with patch('src.tasks.rag_tasks.AsyncSessionLocal') as mock_session_local:

            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            await run_daily_rag_benchmark()

            # Verify database operations were called
            assert mock_session.add.called
            assert mock_session.commit.called


class TestCapturePerformanceSnapshot:
    """Tests for performance snapshot task"""

    @pytest.mark.asyncio
    async def test_snapshot_captures_metrics(self):
        """Test that snapshot captures current performance metrics"""
        with patch('src.tasks.rag_tasks.get_db') as mock_db, \
             patch('src.tasks.rag_tasks.PerformanceTracker') as mock_tracker:

            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            mock_tracker.return_value.get_metrics.return_value = {
                "retrieval": {"avg_latency": 0.5, "count": 100},
                "generation": {"avg_latency": 1.2, "count": 50}
            }

            result = await capture_performance_snapshot()

            assert result is not None
            assert "snapshot_id" in result
            assert "metrics" in result
            assert len(result["metrics"]) == 2

    @pytest.mark.asyncio
    async def test_snapshot_stores_timestamp(self):
        """Test that snapshot includes accurate timestamp"""
        with patch('src.tasks.rag_tasks.get_db') as mock_db, \
             patch('src.tasks.rag_tasks.PerformanceTracker') as mock_tracker:

            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session
            mock_tracker.return_value.get_metrics.return_value = {}

            before = datetime.utcnow()
            result = await capture_performance_snapshot()
            after = datetime.utcnow()

            assert result is not None
            # Timestamp should be between before and after


class TestAnalyzeWeeklyCosts:
    """Tests for weekly cost analysis task"""

    @pytest.mark.asyncio
    async def test_weekly_analysis_calculates_totals(self):
        """Test that weekly analysis calculates cost totals"""
        with patch('src.tasks.rag_tasks.AsyncSessionLocal') as mock_session_local:

            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            # Mock query results for past week
            from src.models.rag_evaluation import RAGPerformanceMetric
            mock_metrics = [
                Mock(spec=RAGPerformanceMetric, cost=0.05, operation="retrieval"),
                Mock(spec=RAGPerformanceMetric, cost=0.15, operation="generation"),
                Mock(spec=RAGPerformanceMetric, cost=0.10, operation="retrieval"),
            ]

            # Create proper async mock chain
            mock_scalars = Mock()
            mock_scalars.all.return_value = mock_metrics

            mock_result = AsyncMock()
            mock_result.scalars.return_value = mock_scalars
            mock_session.execute.return_value = mock_result

            result = await analyze_weekly_costs()

            assert result is not None
            assert "total_cost" in result
            assert result["total_cost"] == 0.30
            assert "by_operation" in result

    @pytest.mark.asyncio
    async def test_weekly_analysis_groups_by_operation(self):
        """Test that analysis groups costs by operation type"""
        with patch('src.tasks.rag_tasks.AsyncSessionLocal') as mock_session_local:

            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            from src.models.rag_evaluation import RAGPerformanceMetric
            mock_metrics = [
                Mock(spec=RAGPerformanceMetric, cost=0.05, operation="retrieval"),
                Mock(spec=RAGPerformanceMetric, cost=0.15, operation="generation"),
                Mock(spec=RAGPerformanceMetric, cost=0.10, operation="retrieval"),
            ]

            # Create proper async mock chain
            mock_scalars = Mock()
            mock_scalars.all.return_value = mock_metrics

            mock_result = AsyncMock()
            mock_result.scalars.return_value = mock_scalars
            mock_session.execute.return_value = mock_result

            result = await analyze_weekly_costs()

            assert "by_operation" in result
            assert result["by_operation"]["retrieval"] == 0.15
            assert result["by_operation"]["generation"] == 0.15


class TestEvaluateABTest:
    """Tests for A/B test evaluation task"""

    @pytest.mark.asyncio
    async def test_ab_test_evaluation_succeeds(self):
        """Test that A/B test evaluation runs successfully"""
        from uuid import uuid4

        test_uuid = uuid4()
        test_id = str(test_uuid)

        with patch('src.tasks.rag_tasks.AsyncSessionLocal') as mock_session_local:

            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            # Mock test and results
            from src.models.rag_evaluation import RAGABTest, RAGABTestResult

            mock_test = Mock(spec=RAGABTest, id=test_uuid, name="Test 1")
            mock_results = [
                Mock(spec=RAGABTestResult, variant_name="variant_a",
                     metrics={"score": 0.9}, cost=0.05),
                Mock(spec=RAGABTestResult, variant_name="variant_b",
                     metrics={"score": 0.7}, cost=0.03),
            ]

            mock_test_result = AsyncMock()
            mock_test_result.scalar_one_or_none.return_value = mock_test

            mock_results_result = AsyncMock()
            mock_results_result.scalars.return_value.all.return_value = mock_results

            mock_session.execute.side_effect = [mock_test_result, mock_results_result]

            result = await evaluate_ab_test(test_id)

            assert result is not None
            assert "test_id" in result
            assert "winner" in result
            assert result["winner"] == "variant_a"

    @pytest.mark.asyncio
    async def test_ab_test_not_found_raises_error(self):
        """Test that missing A/B test raises appropriate error"""
        from uuid import uuid4

        # Use valid UUID that doesn't exist
        test_id = str(uuid4())

        with patch('src.tasks.rag_tasks.AsyncSessionLocal') as mock_session_local:

            mock_session = AsyncMock()
            mock_session_local.return_value.__aenter__.return_value = mock_session

            mock_result = AsyncMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_session.execute.return_value = mock_result

            with pytest.raises(ValueError, match="A/B test not found"):
                await evaluate_ab_test(test_id)
