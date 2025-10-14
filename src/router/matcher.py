from typing import List, Dict
from src.router.types import TaskProfile, AgentConfig
from src.agents.pool import AgentPool


class AgentCapabilityMatcher:
    """Matches task requirements to agent capabilities"""

    def __init__(self, agent_pool: AgentPool):
        self.agent_pool = agent_pool

    def select_agents(
        self,
        task_profile: TaskProfile,
        performance_history: Dict
    ) -> List[AgentConfig]:
        """Select optimal agents for task"""

        # Score all agents
        scores = {}
        for agent_id, agent in self.agent_pool.agents.items():
            score = self._calculate_match_score(
                agent,
                task_profile,
                performance_history.get(agent_id, {})
            )
            scores[agent_id] = score

        # Select top agents based on complexity
        num_agents = self._determine_team_size(task_profile)
        top_agents = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:num_agents]

        # Create configs
        configs = [
            AgentConfig(
                agent_id=agent_id,
                agent=self.agent_pool.get_agent(agent_id),
                match_score=score
            )
            for agent_id, score in top_agents
            if score > 0.2  # Minimum threshold (lowered to allow partial matches)
        ]

        # If no agents meet threshold for complex tasks, select best available
        if not configs and task_profile.complexity == "high" and top_agents:
            configs = [
                AgentConfig(
                    agent_id=top_agents[0][0],
                    agent=self.agent_pool.get_agent(top_agents[0][0]),
                    match_score=top_agents[0][1]
                )
            ]

        return configs

    def _calculate_match_score(
        self,
        agent,
        task_profile: TaskProfile,
        history: Dict
    ) -> float:
        """Calculate how well agent matches task"""

        # Domain overlap (40%)
        domain_match = len(
            set(agent.domains) & set(task_profile.domains)
        ) / max(len(task_profile.domains), 1)

        # Capability overlap (30%)
        capability_match = len(
            set(agent.capabilities) & set(task_profile.required_expertise)
        ) / max(len(task_profile.required_expertise), 1)

        # Past performance (20%)
        past_performance = history.get("success_rate", 0.5)

        # Specialization bonus (10%)
        specialization_bonus = any(
            spec.lower() in task_profile.keywords
            for spec in agent.specializations
        ) * 0.1

        return (
            domain_match * 0.4 +
            capability_match * 0.3 +
            past_performance * 0.2 +
            specialization_bonus
        )

    def _determine_team_size(self, task_profile: TaskProfile) -> int:
        """Determine how many agents needed"""
        if task_profile.complexity == "simple":
            return 1
        elif task_profile.complexity == "medium":
            return min(2, len(task_profile.domains))
        else:  # high
            return min(3, len(task_profile.domains))
