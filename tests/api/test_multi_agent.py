"""Tests for multi-agent API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock

from src.main import app
from src.api.v1.multi_agent import get_meta_router
from src.router.meta_router import MetaRouter
from src.router.execution import ExecutionResult
from src.agents.types import AgentResult


@pytest.fixture
async def mock_meta_router():
    """Create mock meta router for testing."""
    mock_router = AsyncMock(spec=MetaRouter)

    # Mock successful research result
    mock_result = ExecutionResult(
        status="success",
        agent_results=[
            AgentResult(
                agent_id="test_agent",
                task_id="test_task",
                output="Test research output",
                confidence=0.85
            )
        ],
        quality_score=0.85,
        execution_time_ms=100.0,
        metadata={"agents_used": ["test_agent"]}
    )

    mock_router.route_and_execute = AsyncMock(return_value=mock_result)
    return mock_router


@pytest.fixture
async def client(mock_meta_router):
    """Create async HTTP client for testing with mocked dependencies."""
    # Override the dependency
    app.dependency_overrides[get_meta_router] = lambda: mock_meta_router

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.mark.asyncio
class TestMultiAgentAPI:
    """Integration tests for Multi-Agent API."""

    async def test_multi_agent_research_endpoint(self, client):
        """Test multi-agent research API."""
        response = await client.post(
            "/api/v1/multi-agent/research",
            json={
                "description": "Search for fMRI emotion papers",
                "task_type": "literature_search",
                "quality_target": 0.8
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "quality_score" in data
        assert "agent_results" in data

    async def test_multi_agent_hypothesis_generation(self, client):
        """Test hypothesis generation endpoint."""
        response = await client.post(
            "/api/v1/multi-agent/hypothesis",
            json={
                "research_question": "Novel DL approach for fMRI emotion prediction",
                "context": "Previous work used CNNs",
                "quality_target": 0.85
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "hypotheses" in data
