import pytest
from src.agents.base import ResearchAgent
from src.agents.types import AgentTask, AgentResult

class TestAgent(ResearchAgent):
    async def process(self, task: AgentTask, context: dict) -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            output="test output",
            confidence=0.9
        )

@pytest.mark.asyncio
async def test_agent_has_required_attributes():
    agent = TestAgent(
        agent_id="test_agent",
        llm_service=None,
        context_manager=None
    )

    assert agent.agent_id == "test_agent"
    assert hasattr(agent, 'capabilities')
    assert hasattr(agent, 'domains')
    assert hasattr(agent, 'performance_history')

@pytest.mark.asyncio
async def test_agent_can_process_task():
    agent = TestAgent("test", None, None)
    from src.agents.types import TaskType
    task = AgentTask(task_id="t1", task_type=TaskType.LITERATURE_SEARCH, description="test")

    result = await agent.process(task, {})

    assert result.agent_id == "test"
    assert result.task_id == "t1"
    assert result.confidence == 0.9
