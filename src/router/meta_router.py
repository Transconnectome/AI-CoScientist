# src/router/meta_router.py
import asyncio
from typing import List
from src.router.analyzer import TaskAnalyzer
from src.router.matcher import AgentCapabilityMatcher
from src.router.types import ResearchTask
from src.router.execution import ExecutionResult
from src.agents.types import AgentTask, AgentResult
from src.agents.pool import AgentPool
from datetime import datetime

class MetaRouter:
    """Orchestrates task analysis, agent selection, and execution"""

    def __init__(self, llm_service, agent_pool: AgentPool, context_manager):
        self.task_analyzer = TaskAnalyzer(llm_service)
        self.agent_matcher = AgentCapabilityMatcher(agent_pool)
        self.context_manager = context_manager
        self.agent_pool = agent_pool

    async def route_and_execute(
        self,
        task: ResearchTask
    ) -> ExecutionResult:
        """Full pipeline: analyze → select agents → execute → calculate quality"""

        start_time = datetime.utcnow()

        # Step 1: Analyze task
        task_profile = await self.task_analyzer.analyze_task(task)

        # Step 2: Select agents
        performance_history = {}  # TODO: Load from database
        agent_configs = self.agent_matcher.select_agents(
            task_profile,
            performance_history
        )

        # Step 3: Execute with agents
        agent_results = await self._execute_with_agents(
            task,
            agent_configs
        )

        # Step 4: Calculate quality
        quality_score = self._calculate_quality(agent_results)

        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return ExecutionResult(
            status="success" if quality_score > 0.7 else "partial_success",
            agent_results=agent_results,
            quality_score=quality_score,
            execution_time_ms=execution_time,
            metadata={
                "task_profile": task_profile,
                "agents_used": [c.agent_id for c in agent_configs]
            }
        )

    async def _execute_with_agents(
        self,
        task: ResearchTask,
        agent_configs: List
    ) -> List[AgentResult]:
        """Execute task with selected agents"""

        results = []

        for agent_config in agent_configs:
            # Get relevant context for this agent
            context = await self.context_manager.get_relevant(
                agent_id=agent_config.agent_id,
                task_type=task.task_type,
                max_tokens=4000
            )

            # Create agent task
            agent_task = AgentTask(
                task_id=f"{task.task_type}_1",
                task_type=task.task_type,
                description=task.description
            )

            # Execute
            result = await agent_config.agent.process(
                agent_task,
                context
            )

            results.append(result)

            # Store result as insight for next agents
            if result.confidence > 0.7:
                from src.context.types import Insight
                insight = Insight(
                    content=str(result.output),
                    type="agent_result",
                    domains=[],
                    score=result.confidence
                )
                await self.context_manager.store_insight(
                    insight,
                    agent_config.agent_id,
                    agent_task.task_id,
                    {}
                )

        return results

    def _calculate_quality(self, results: List[AgentResult]) -> float:
        """Calculate overall quality from agent results"""
        if not results:
            return 0.0

        return sum(r.confidence for r in results) / len(results)
