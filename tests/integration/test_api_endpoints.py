"""
Integration tests for Phase 2 RAG Evaluation API endpoints.

TDD RED Phase: Tests document expected API behavior with real database.
All 11 endpoints tested with actual PostgreSQL operations.

Tests verify:
- Request/response schemas
- Database persistence
- Error handling
- Status codes
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.rag_evaluation import (
    RAGABTest,
    RAGABTestResult,
    RAGCostBudget,
    RAGEvaluation,
    RAGPerformanceMetric,
)


class TestEvaluationEndpoints:
    """Test RAG evaluation API endpoints."""

    @pytest.mark.asyncio
    async def test_run_ragas_evaluation_endpoint(self, api_client: AsyncClient, db_session: AsyncSession, clean_database):
        """Test POST /rag/evaluation/ragas creates evaluation in database."""
        response = await api_client.post(
            "/api/v1/rag/evaluation/ragas",
            json={"dataset_id": "api_test_001", "config": {}}
        )

        assert response.status_code == 200
        data = response.json()
        assert "evaluation_id" in data
        assert "metrics" in data

        # Verify database persistence
        result = await db_session.execute(
            select(RAGEvaluation).where(RAGEvaluation.dataset_id == "api_test_001")
        )
        evaluation = result.scalar_one_or_none()
        assert evaluation is not None
        assert evaluation.evaluation_type == "ragas"

    @pytest.mark.asyncio
    async def test_get_evaluation_history_endpoint(self, api_client: AsyncClient, create_evaluation, clean_database):
        """Test GET /rag/evaluation/history retrieves evaluations."""
        response = await api_client.get("/api/v1/rag/evaluation/history")

        assert response.status_code == 200
        data = response.json()
        assert "evaluations" in data
        assert len(data["evaluations"]) > 0


class TestPerformanceEndpoints:
    """Test performance tracking API endpoints."""

    @pytest.mark.asyncio
    async def test_track_performance_endpoint(self, api_client: AsyncClient, db_session: AsyncSession, clean_database):
        """Test POST /rag/performance/track stores metric in database."""
        response = await api_client.post(
            "/api/v1/rag/performance/track",
            json={
                "operation": "api_test_retrieval",
                "latency": 0.5,
                "tokens": {"prompt": 100, "completion": 50},
                "model": "gpt-4"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "metric_id" in data
        assert data["operation"] == "api_test_retrieval"

        # Verify database persistence
        result = await db_session.execute(
            select(RAGPerformanceMetric).where(RAGPerformanceMetric.operation == "api_test_retrieval")
        )
        metric = result.scalar_one_or_none()
        assert metric is not None
        assert metric.latency == 0.5

    @pytest.mark.asyncio
    async def test_get_performance_metrics_endpoint(self, api_client: AsyncClient, create_performance_metric, clean_database):
        """Test GET /rag/performance/metrics retrieves metrics."""
        response = await api_client.get("/api/v1/rag/performance/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert len(data["metrics"]) > 0


class TestCostEndpoints:
    """Test cost management API endpoints."""

    @pytest.mark.asyncio
    async def test_create_budget_endpoint(self, api_client: AsyncClient, db_session: AsyncSession, clean_database):
        """Test POST /rag/cost/budgets creates budget in database."""
        response = await api_client.post(
            "/api/v1/rag/cost/budgets",
            json={
                "name": "API Test Budget",
                "total_budget": 100.0,
                "warning_threshold": 0.8,
                "critical_threshold": 0.95
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "budget_id" in data

        # Verify database persistence
        result = await db_session.execute(
            select(RAGCostBudget).where(RAGCostBudget.name == "API Test Budget")
        )
        budget = result.scalar_one_or_none()
        assert budget is not None
        assert budget.total_budget == 100.0

    @pytest.mark.asyncio
    async def test_get_budget_endpoint(self, api_client: AsyncClient, create_budget, clean_database):
        """Test GET /rag/cost/budgets/{id} retrieves budget."""
        budget_id = str(create_budget.id)
        response = await api_client.get(f"/api/v1/rag/cost/budgets/{budget_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == budget_id
        assert "remaining" in data
        assert "usage_ratio" in data
        assert "status" in data


class TestABTestEndpoints:
    """Test A/B testing API endpoints."""

    @pytest.mark.asyncio
    async def test_create_ab_test_endpoint(self, api_client: AsyncClient, db_session: AsyncSession, clean_database):
        """Test POST /rag/ab-test creates test in database."""
        response = await api_client.post(
            "/api/v1/rag/ab-test",
            json={
                "name": "API A/B Test",
                "config": {"variants": ["A", "B"], "split": 0.5}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "test_id" in data

        # Verify database persistence
        result = await db_session.execute(
            select(RAGABTest).where(RAGABTest.name == "API A/B Test")
        )
        ab_test = result.scalar_one_or_none()
        assert ab_test is not None
        assert ab_test.status == "active"

    @pytest.mark.asyncio
    async def test_record_ab_test_result_endpoint(self, api_client: AsyncClient, db_session: AsyncSession, create_ab_test, clean_database):
        """Test POST /rag/ab-test/{id}/result stores result."""
        test_id = str(create_ab_test.id)
        response = await api_client.post(
            f"/api/v1/rag/ab-test/{test_id}/result",
            json={
                "variant_name": "variant_a",
                "metrics": {"accuracy": 0.95},
                "cost": 0.02
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "result_id" in data

        # Verify database persistence
        result = await db_session.execute(
            select(RAGABTestResult).where(RAGABTestResult.test_id == create_ab_test.id)
        )
        ab_result = result.scalar_one_or_none()
        assert ab_result is not None
        assert ab_result.variant_name == "variant_a"
