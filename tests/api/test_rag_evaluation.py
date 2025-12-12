"""
Tests for RAG evaluation REST API endpoints.

Following TDD methodology (RED → GREEN → REFACTOR):
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Improve code quality
"""

import pytest
from httpx import AsyncClient, ASGITransport
from uuid import uuid4

from src.main import app


@pytest.fixture
async def client():
    """Create async test client"""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


class TestPerformanceEndpoints:
    """Performance tracking endpoint tests"""

    @pytest.mark.asyncio
    async def test_track_performance(self, client):
        """Track performance metric"""
        response = await client.post(
            "/api/v1/rag-evaluation/performance/track",
            json={
                "operation": "retrieval",
                "latency": 0.5,
                "tokens": {"prompt": 100, "completion": 50},
                "model": "gpt-4"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "metric_id" in data
        assert data["operation"] == "retrieval"

    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, client):
        """Get aggregated performance metrics"""
        response = await client.get("/api/v1/rag-evaluation/performance/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "latency" in data
        assert "tokens" in data

    @pytest.mark.asyncio
    async def test_reset_performance_tracker(self, client):
        """Reset performance tracker"""
        response = await client.post("/api/v1/rag-evaluation/performance/reset")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reset"


class TestCostEndpoints:
    """Cost optimization endpoint tests"""

    @pytest.mark.asyncio
    async def test_create_budget(self, client):
        """Create cost budget"""
        response = await client.post(
            "/api/v1/rag-evaluation/cost/budget/create",
            json={
                "name": "Monthly Budget",
                "total": 100.0,
                "warning_threshold": 0.8,
                "critical_threshold": 0.95
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "budget_id" in data
        assert data["total"] == 100.0

    @pytest.mark.asyncio
    async def test_get_budget(self, client):
        """Get budget by ID"""
        # Create budget first
        create_response = await client.post(
            "/api/v1/rag-evaluation/cost/budget/create",
            json={"name": "Test Budget", "total": 50.0}
        )
        budget_id = create_response.json()["budget_id"]

        # Get budget
        response = await client.get(f"/api/v1/rag-evaluation/cost/budget/{budget_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["budget_id"] == budget_id
        assert data["total"] == 50.0

    @pytest.mark.asyncio
    async def test_get_optimization_suggestions(self, client):
        """Get cost optimization suggestions"""
        response = await client.post(
            "/api/v1/rag-evaluation/cost/optimize",
            json={
                "usage_data": {
                    "tokens": {
                        "gpt-4": {
                            "prompt_tokens": 10000,
                            "completion_tokens": 5000,
                            "estimated_cost": 0.75
                        }
                    }
                }
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

    @pytest.mark.asyncio
    async def test_get_cost_suggestions_for_budget(self, client):
        """Get suggestions for specific budget"""
        # Create budget
        create_response = await client.post(
            "/api/v1/rag-evaluation/cost/budget/create",
            json={"name": "Test Budget", "total": 50.0}
        )
        budget_id = create_response.json()["budget_id"]

        response = await client.get(
            f"/api/v1/rag-evaluation/cost/suggestions?budget_id={budget_id}"
        )

        assert response.status_code == 200


class TestABTestEndpoints:
    """A/B testing endpoint tests"""

    @pytest.mark.asyncio
    async def test_create_ab_test(self, client):
        """Create A/B test"""
        response = await client.post(
            "/api/v1/rag-evaluation/ab-test/create",
            json={
                "name": "Model Comparison",
                "variants": [
                    {"name": "gpt4", "config": {"model": "gpt-4"}},
                    {"name": "gpt35", "config": {"model": "gpt-3.5-turbo"}}
                ],
                "traffic_split": {"gpt4": 0.5, "gpt35": 0.5}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "test_id" in data
        assert data["name"] == "Model Comparison"

    @pytest.mark.asyncio
    async def test_add_ab_test_result(self, client):
        """Add result to A/B test"""
        # Create test first
        create_response = await client.post(
            "/api/v1/rag-evaluation/ab-test/create",
            json={
                "name": "Test",
                "variants": [
                    {"name": "a", "config": {}},
                    {"name": "b", "config": {}}
                ],
                "traffic_split": {"a": 0.5, "b": 0.5}
            }
        )
        test_id = create_response.json()["test_id"]

        # Add result
        response = await client.post(
            f"/api/v1/rag-evaluation/ab-test/{test_id}/add-result",
            json={
                "variant_name": "a",
                "metrics": {"faithfulness": 0.85},
                "cost": 0.05
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "added"

    @pytest.mark.asyncio
    async def test_analyze_ab_test(self, client):
        """Analyze A/B test results"""
        # Create test
        create_response = await client.post(
            "/api/v1/rag-evaluation/ab-test/create",
            json={
                "name": "Test",
                "variants": [
                    {"name": "a", "config": {}},
                    {"name": "b", "config": {}}
                ],
                "traffic_split": {"a": 0.5, "b": 0.5}
            }
        )
        test_id = create_response.json()["test_id"]

        # Add some results
        for _ in range(10):
            await client.post(
                f"/api/v1/rag-evaluation/ab-test/{test_id}/add-result",
                json={"variant_name": "a", "metrics": {"score": 0.8}, "cost": 0.05}
            )

        # Analyze
        response = await client.get(
            f"/api/v1/rag-evaluation/ab-test/{test_id}/analyze?metric=score"
        )

        assert response.status_code == 200
        data = response.json()
        assert "winner" in data
        assert "confidence" in data

    @pytest.mark.asyncio
    async def test_declare_ab_test_winner(self, client):
        """Declare A/B test winner"""
        # Create test
        create_response = await client.post(
            "/api/v1/rag-evaluation/ab-test/create",
            json={
                "name": "Test",
                "variants": [
                    {"name": "a", "config": {}},
                    {"name": "b", "config": {}}
                ],
                "traffic_split": {"a": 0.5, "b": 0.5}
            }
        )
        test_id = create_response.json()["test_id"]

        # Add many results to ensure statistical significance
        for _ in range(100):
            await client.post(
                f"/api/v1/rag-evaluation/ab-test/{test_id}/add-result",
                json={"variant_name": "a", "metrics": {"score": 0.9}, "cost": 0.05}
            )
            await client.post(
                f"/api/v1/rag-evaluation/ab-test/{test_id}/add-result",
                json={"variant_name": "b", "metrics": {"score": 0.7}, "cost": 0.01}
            )

        # Declare winner
        response = await client.get(
            f"/api/v1/rag-evaluation/ab-test/{test_id}/winner?metric=score"
        )

        assert response.status_code == 200
        data = response.json()
        if data["winner"] is not None:
            assert "variant" in data["winner"]
            assert "confidence" in data["winner"]


class TestRAGASEndpoints:
    """RAGAS evaluation endpoint tests"""

    @pytest.mark.asyncio
    async def test_evaluate_ragas(self, client):
        """Evaluate RAG system with RAGAS"""
        response = await client.post(
            "/api/v1/rag-evaluation/ragas/evaluate",
            json={
                "dataset": [
                    {
                        "question": "What is AI?",
                        "answer": "AI is artificial intelligence",
                        "contexts": ["AI stands for artificial intelligence"],
                        "ground_truth": "Artificial intelligence"
                    }
                ]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "evaluation_id" in data
        assert "metrics" in data
        assert "faithfulness" in data["metrics"]

    @pytest.mark.asyncio
    async def test_get_ragas_baseline(self, client):
        """Get RAGAS baseline metrics"""
        # Evaluate first
        eval_response = await client.post(
            "/api/v1/rag-evaluation/ragas/evaluate",
            json={
                "dataset": [
                    {
                        "question": "Test?",
                        "answer": "Test answer",
                        "contexts": ["Test context"],
                        "ground_truth": "Ground truth"
                    }
                ]
            }
        )
        dataset_id = eval_response.json()["evaluation_id"]

        # Get baseline
        response = await client.get(
            f"/api/v1/rag-evaluation/ragas/baseline/{dataset_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert "baseline_metrics" in data

    @pytest.mark.asyncio
    async def test_get_ragas_metrics(self, client):
        """Get available RAGAS metrics"""
        response = await client.get("/api/v1/rag-evaluation/ragas/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert isinstance(data["metrics"], list)


class TestPrometheusEndpoint:
    """Prometheus metrics endpoint test"""

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = await client.get("/metrics")

        assert response.status_code == 200
        # Prometheus format is plain text
        assert "text/plain" in response.headers.get("content-type", "")
