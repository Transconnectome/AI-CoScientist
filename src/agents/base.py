from abc import ABC, abstractmethod
from typing import Dict, List, Any
from src.agents.types import AgentTask, AgentResult

class ResearchAgent(ABC):
    """Base class for all specialized research agents"""

    def __init__(
        self,
        agent_id: str,
        llm_service: Any,
        context_manager: Any
    ):
        self.agent_id = agent_id
        self.llm = llm_service
        self.context = context_manager

        # Agent metadata
        self.capabilities: List[str] = []
        self.domains: List[str] = []
        self.specializations: List[str] = []
        self.performance_history: Dict = {}

    @abstractmethod
    async def process(
        self,
        task: AgentTask,
        relevant_context: Dict
    ) -> AgentResult:
        """Core processing logic - each agent implements this"""
        pass

    def update_performance(self, task_id: str, success: bool, score: float):
        """Track performance for learning"""
        self.performance_history[task_id] = {
            "success": success,
            "score": score
        }

    def get_success_rate(self) -> float:
        """Calculate historical success rate"""
        if not self.performance_history:
            return 0.5  # Default for new agents

        successes = sum(
            1 for h in self.performance_history.values()
            if h["success"]
        )
        return successes / len(self.performance_history)
