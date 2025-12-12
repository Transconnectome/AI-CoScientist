# tests/agents/test_agent_pool.py
import pytest
from src.agents.pool import AgentPool
from src.agents.base import ResearchAgent
from src.agents.types import AgentTask, AgentResult

@pytest.fixture
def mock_llm_service():
    return None  # Mock for now

@pytest.fixture
def mock_context_manager():
    return None

@pytest.fixture
def agent_pool(mock_llm_service, mock_context_manager):
    return AgentPool(mock_llm_service, mock_context_manager)

def test_agent_pool_has_agents(agent_pool):
    """Agent pool should have registered agents"""
    assert len(agent_pool.agents) > 0
    assert "neuroscience_expert" in agent_pool.agents

def test_get_agent_by_id(agent_pool):
    """Can retrieve agent by ID"""
    agent = agent_pool.get_agent("neuroscience_expert")
    assert agent is not None
    assert agent.agent_id == "neuroscience_expert"

def test_get_agents_by_capability(agent_pool):
    """Can find agents by capability"""
    agents = agent_pool.get_agents_by_capability("domain_validation")
    assert len(agents) > 0
    assert all("domain_validation" in a.capabilities for a in agents)

def test_get_agents_by_domain(agent_pool):
    """Can find agents by domain"""
    agents = agent_pool.get_agents_by_domain("neuroscience")
    assert len(agents) > 0
    assert all("neuroscience" in a.domains for a in agents)
