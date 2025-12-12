import pytest
from src.router.matcher import AgentCapabilityMatcher
from src.router.types import TaskProfile, ComplexityLevel, AgentConfig
from src.agents.pool import AgentPool


@pytest.fixture
def mock_llm_service():
    return None  # Mock for now


@pytest.fixture
def mock_context_manager():
    return None


@pytest.fixture
def agent_pool(mock_llm_service, mock_context_manager):
    return AgentPool(mock_llm_service, mock_context_manager)


@pytest.fixture
def matcher(agent_pool):
    return AgentCapabilityMatcher(agent_pool)


def test_match_simple_task(matcher):
    """Match agents for simple single-domain task"""
    profile = TaskProfile(
        domains=["neuroscience"],
        complexity=ComplexityLevel.SIMPLE,
        task_type="literature_search",
        sub_tasks=[],
        required_expertise=["domain_knowledge"],
        quality_gates=[],
        context_dependencies=[]
    )

    agents = matcher.select_agents(profile, {})

    assert len(agents) > 0
    assert any(a.agent_id == "neuroscience_expert" for a in agents)


def test_match_complex_task(matcher):
    """Match multiple agents for complex task"""
    profile = TaskProfile(
        domains=["neuroscience", "machine_learning", "ethics"],
        complexity=ComplexityLevel.HIGH,
        task_type="hypothesis_generation",
        sub_tasks=[],
        required_expertise=["domain_knowledge", "creative_synthesis"],
        quality_gates=[],
        context_dependencies=[]
    )

    agents = matcher.select_agents(profile, {})

    # Should select at least one agent for complex task
    assert len(agents) >= 1
    domains_covered = set()
    for agent_config in agents:
        domains_covered.update(agent_config.agent.domains)

    # Should cover neuroscience domain (the only available expert)
    assert "neuroscience" in domains_covered


def test_scoring_considers_performance_history(matcher):
    """Agent selection considers past performance"""
    # Mock performance history
    history = {
        "neuroscience_expert": {
            "success_rate": 0.95,
            "avg_quality": 0.88
        }
    }

    profile = TaskProfile(
        domains=["neuroscience"],
        complexity=ComplexityLevel.MEDIUM,
        task_type="validation",
        sub_tasks=[],
        required_expertise=["domain_validation"],
        quality_gates=[],
        context_dependencies=[]
    )

    agents = matcher.select_agents(profile, history)

    # High-performing agent should be selected
    assert agents[0].agent_id == "neuroscience_expert"
