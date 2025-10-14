# src/agents/pool.py
from typing import Dict, List
from src.agents.base import ResearchAgent
from src.agents.domain_experts import NeuroscienceExpertAgent

class AgentPool:
    """Registry and manager for all agents"""

    def __init__(self, llm_service, context_manager):
        self.llm_service = llm_service
        self.context_manager = context_manager

        # Initialize agent registry
        self.agents: Dict[str, ResearchAgent] = {}
        self._register_agents()

    def _register_agents(self):
        """Register all available agents"""

        # Domain experts
        self.agents["neuroscience_expert"] = NeuroscienceExpertAgent(
            "neuroscience_expert",
            self.llm_service,
            self.context_manager
        )

        # More agents will be added in future tasks

    def get_agent(self, agent_id: str) -> ResearchAgent:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def get_agents_by_capability(
        self,
        capability: str
    ) -> List[ResearchAgent]:
        """Find agents with specific capability"""
        return [
            agent for agent in self.agents.values()
            if capability in agent.capabilities
        ]

    def get_agents_by_domain(
        self,
        domain: str
    ) -> List[ResearchAgent]:
        """Find agents for specific domain"""
        return [
            agent for agent in self.agents.values()
            if domain in agent.domains
        ]

    def list_all_agents(self) -> Dict[str, Dict]:
        """Get metadata for all agents"""
        return {
            agent_id: {
                "capabilities": agent.capabilities,
                "domains": agent.domains,
                "specializations": agent.specializations,
                "success_rate": agent.get_success_rate()
            }
            for agent_id, agent in self.agents.items()
        }
