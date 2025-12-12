# src/metrics/tracker.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Dict, Any
from src.metrics.types import AgentExecution, WorkflowMetric
from src.agents.types import AgentResult


class PerformanceTracker:
    """Tracks agent and workflow performance"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def record_agent_execution(
        self,
        result: AgentResult,
        task_type: str,
        success: bool
    ):
        """Record individual agent execution"""

        execution = AgentExecution(
            agent_id=result.agent_id,
            task_type=task_type,
            task_id=result.task_id,
            success=success,
            confidence=result.confidence,
            execution_time_ms=result.execution_time_ms,
            tokens_used=result.tokens_used,
            extra_metadata=result.metadata or {}
        )

        self.db.add(execution)
        await self.db.commit()

    async def get_agent_stats(
        self,
        agent_id: str,
        task_type: str = None
    ) -> Dict[str, Any]:
        """Get performance stats for agent"""

        query = select(AgentExecution).where(
            AgentExecution.agent_id == agent_id
        )

        if task_type:
            query = query.where(AgentExecution.task_type == task_type)

        result = await self.db.execute(query)
        executions = result.scalars().all()

        if not executions:
            return {
                "total_executions": 0,
                "success_rate": 0.5,
                "avg_confidence": 0.5
            }

        successes = sum(1 for e in executions if e.success)

        return {
            "total_executions": len(executions),
            "success_rate": successes / len(executions),
            "avg_confidence": sum(e.confidence or 0 for e in executions) / len(executions),
            "avg_execution_time_ms": sum(e.execution_time_ms or 0 for e in executions) / len(executions)
        }
