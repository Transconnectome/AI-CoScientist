# tests/context/test_manager.py
import pytest
from src.context.manager import ResearchContextManager
from src.context.types import Insight, ResearchSession

@pytest.fixture
def context_manager():
    # For now, use in-memory storage
    return ResearchContextManager(vector_store=None, graph_db=None)

@pytest.mark.asyncio
async def test_store_insight(context_manager):
    """Store an insight with metadata"""
    insight = Insight(
        content="Deep learning shows promise for fMRI analysis",
        type="finding",
        domains=["neuroscience", "machine_learning"],
        score=0.85
    )

    node_id = await context_manager.store_insight(
        insight=insight,
        source_agent="literature_scout",
        task_id="task1",
        metadata={"source": "paper_123"}
    )

    assert node_id is not None

@pytest.mark.asyncio
async def test_get_relevant_context(context_manager):
    """Retrieve relevant context for agent"""
    # Store some insights first
    insight1 = Insight(
        content="fMRI has 2s lag",
        type="constraint",
        domains=["neuroscience"],
        score=0.9
    )

    await context_manager.store_insight(
        insight1,
        "neuro_expert",
        "task1",
        {}
    )

    # Retrieve for agent
    context = await context_manager.get_relevant(
        agent_id="hypothesis_generator",
        task_type="hypothesis_generation",
        max_tokens=1000
    )

    assert context is not None
    assert "insights" in context

@pytest.mark.asyncio
async def test_context_budget_management(context_manager):
    """Context stays within token budget"""
    # Store multiple insights
    for i in range(10):
        insight = Insight(
            content=f"Finding {i}: " + "x" * 100,
            type="finding",
            domains=["test"],
            score=0.7
        )
        await context_manager.store_insight(insight, "test", f"t{i}", {})

    # Retrieve with budget
    context = await context_manager.get_relevant(
        agent_id="test",
        task_type="test",
        max_tokens=500  # Small budget
    )

    # Should prioritize and fit within budget
    total_length = sum(len(i.content) for i in context["insights"])
    assert total_length < 500 * 4  # Rough token estimate
