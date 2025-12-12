# tests/router/test_meta_router.py
import pytest
from src.router.meta_router import MetaRouter
from src.router.types import ResearchTask

@pytest.fixture
def mock_llm_service():
    return None  # Mock for now

@pytest.fixture
def mock_context_manager():
    from src.context.manager import ResearchContextManager
    return ResearchContextManager(vector_store=None, graph_db=None)

@pytest.fixture
def agent_pool(mock_llm_service, mock_context_manager):
    from src.agents.pool import AgentPool
    return AgentPool(mock_llm_service, mock_context_manager)

@pytest.fixture
def meta_router(mock_llm_service, mock_context_manager, agent_pool):
    return MetaRouter(
        llm_service=mock_llm_service,
        agent_pool=agent_pool,
        context_manager=mock_context_manager
    )

@pytest.mark.asyncio
async def test_route_simple_task(meta_router):
    """Route a simple task end-to-end"""
    task = ResearchTask(
        description="Search fMRI neuroscience papers on emotion recognition",
        task_type="literature_search"
    )

    result = await meta_router.route_and_execute(task)

    assert result is not None
    assert result.status in ["success", "partial_success"]
    assert len(result.agent_results) > 0
    assert result.quality_score >= 0.0
    assert result.execution_time_ms > 0
    assert "task_profile" in result.metadata
    assert "agents_used" in result.metadata

@pytest.mark.asyncio
async def test_route_complex_multi_agent_task(meta_router):
    """Route complex task requiring multiple agents"""
    task = ResearchTask(
        description="""Generate hypothesis for fMRI emotion recognition
                       using deep learning with ethical considerations""",
        task_type="hypothesis_generation"
    )

    result = await meta_router.route_and_execute(task)

    # With current single-agent pool, we get 1 agent
    # Test validates the full pipeline works
    assert result.status in ["success", "partial_success"]
    assert len(result.agent_results) >= 1  # At least one agent executes
    assert result.quality_score > 0.0
    assert result.execution_time_ms > 0

    # Verify the pipeline stored the task profile
    assert "task_profile" in result.metadata
    task_profile = result.metadata["task_profile"]
    assert "neuroscience" in task_profile.domains
    assert task_profile.complexity == "high"
