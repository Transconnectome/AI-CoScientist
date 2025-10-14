# tests/metrics/test_tracker.py
import pytest
from unittest.mock import AsyncMock, MagicMock


class MockExecution:
    """Mock AgentExecution for testing"""
    def __init__(self, agent_id, success, confidence):
        self.agent_id = agent_id
        self.success = success
        self.confidence = confidence
        self.execution_time_ms = 100.0


class MockDBSession:
    """Mock database session for testing"""

    def __init__(self):
        self.data = []
        self.committed = False

    def add(self, obj):
        """Add object to session"""
        # Store a simplified version to avoid SQLAlchemy issues
        self.data.append(MockExecution(
            agent_id=obj.agent_id,
            success=obj.success,
            confidence=obj.confidence
        ))

    async def commit(self):
        """Commit transaction"""
        self.committed = True

    async def execute(self, query):
        """Execute query"""
        # Return stored data
        class MockResult:
            def __init__(self, data):
                self.data = data

            def scalars(self):
                class MockScalars:
                    def __init__(self, data):
                        self.data = data

                    def all(self):
                        return self.data
                return MockScalars(self.data)

        return MockResult(self.data)


@pytest.fixture
def mock_db_session():
    """Provide mock database session"""
    return MockDBSession()


@pytest.mark.asyncio
async def test_record_agent_performance(mock_db_session):
    """Record agent performance metrics"""
    # Import here to avoid early SQLAlchemy configuration
    from src.metrics.tracker import PerformanceTracker
    from src.agents.types import AgentResult

    tracker = PerformanceTracker(mock_db_session)

    result = AgentResult(
        agent_id="neuroscience_expert",
        task_id="t1",
        output="analysis",
        confidence=0.88,
        execution_time_ms=450.0
    )

    await tracker.record_agent_execution(
        result,
        task_type="validation",
        success=True
    )

    # Should be stored
    stats = await tracker.get_agent_stats("neuroscience_expert")
    assert stats["total_executions"] == 1
    assert stats["success_rate"] == 1.0


@pytest.mark.asyncio
async def test_calculate_success_rate(mock_db_session):
    """Calculate agent success rate over time"""
    # Import here to avoid early SQLAlchemy configuration
    from src.metrics.tracker import PerformanceTracker
    from src.agents.types import AgentResult

    tracker = PerformanceTracker(mock_db_session)

    # Record multiple executions
    for i in range(10):
        result = AgentResult(
            agent_id="test_agent",
            task_id=f"t{i}",
            output="output",
            confidence=0.7 + (i * 0.02)
        )
        await tracker.record_agent_execution(
            result,
            task_type="test",
            success=(i % 2 == 0)  # 50% success
        )

    stats = await tracker.get_agent_stats("test_agent")
    assert stats["success_rate"] == 0.5
    assert stats["total_executions"] == 10
