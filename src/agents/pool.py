# src/agents/pool.py
from typing import Dict, List, Optional, Any
from src.agents.base import ResearchAgent
from src.agents.domain_experts import NeuroscienceExpertAgent
from src.agents.specialist_agents import (
    StatisticalAnalysisAgent,
    GrantWriterAgent,
    HypothesisGeneratorAgent,
    ClinicalValidationAgent,
    EnhancedLiteratureAnalystAgent
)

class AgentPool:
    """Enhanced Agent Pool 2.0 - Registry and manager for all specialized agents"""

    def __init__(self, llm_service, context_manager):
        self.llm_service = llm_service
        self.context_manager = context_manager

        # Initialize agent registry
        self.agents: Dict[str, ResearchAgent] = {}
        self._register_agents()

    def _register_agents(self):
        """Register all available agents including new specialists"""

        # Original domain experts
        self.agents["neuroscience_expert"] = NeuroscienceExpertAgent(
            "neuroscience_expert",
            self.llm_service,
            self.context_manager
        )

        # NEW: Enhanced specialist agents
        self.agents["statistical_analyst"] = StatisticalAnalysisAgent(
            "statistical_analyst",
            self.llm_service,
            self.context_manager
        )

        self.agents["grant_writer"] = GrantWriterAgent(
            "grant_writer",
            self.llm_service,
            self.context_manager
        )

        self.agents["hypothesis_generator"] = HypothesisGeneratorAgent(
            "hypothesis_generator",
            self.llm_service,
            self.context_manager
        )

        self.agents["clinical_validator"] = ClinicalValidationAgent(
            "clinical_validator",
            self.llm_service,
            self.context_manager
        )

        self.agents["literature_analyst"] = EnhancedLiteratureAnalystAgent(
            "literature_analyst",
            self.llm_service,
            self.context_manager
        )

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

    async def execute_parallel_tasks(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """Execute multiple agent tasks in parallel"""
        import asyncio
        from .types import AgentTask

        # Create coroutines for all tasks
        coroutines = []
        for task_config in tasks:
            agent_id = task_config["agent_id"]
            task_data = task_config["task"]
            context = task_config.get("context", {})

            # Get agent
            agent = self.get_agent(agent_id)
            if agent:
                # Create task object if needed
                if isinstance(task_data, dict):
                    task = AgentTask(**task_data)
                else:
                    task = task_data

                # Add coroutine to list
                coroutines.append(agent.process(task, context))

        # Execute all tasks in parallel
        if coroutines:
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            return results
        else:
            return []

    async def collaborative_analysis(self,
                                   research_question: str,
                                   agent_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform collaborative analysis using multiple agents"""
        import asyncio
        from .types import AgentTask
        from datetime import datetime

        # Default to all agents if none specified
        if agent_ids is None:
            agent_ids = list(self.agents.keys())

        # Create tasks for each agent
        tasks = []
        task_id_base = f"collab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for i, agent_id in enumerate(agent_ids):
            task = {
                "agent_id": agent_id,
                "task": {
                    "task_id": f"{task_id_base}_{i}",
                    "description": research_question,
                    "task_type": "collaborative_analysis",
                    "priority": 1
                },
                "context": {
                    "research_question": research_question,
                    "collaboration_mode": True
                }
            }
            tasks.append(task)

        # Execute in parallel
        results = await self.execute_parallel_tasks(tasks)

        # Process and combine results
        processed_results = {}
        for i, result in enumerate(results):
            agent_id = agent_ids[i]
            if isinstance(result, Exception):
                processed_results[agent_id] = {
                    "status": "error",
                    "error": str(result),
                    "output": None
                }
            else:
                processed_results[agent_id] = {
                    "status": "success",
                    "output": result.output,
                    "confidence": result.confidence,
                    "agent_id": result.agent_id
                }

        return {
            "research_question": research_question,
            "participating_agents": agent_ids,
            "results": processed_results,
            "timestamp": datetime.now().isoformat(),
            "success_rate": len([r for r in processed_results.values() if r["status"] == "success"]) / len(processed_results)
        }

    def get_optimal_agent_team(self, task_requirements: Dict[str, Any]) -> List[str]:
        """Select optimal team of agents for specific task requirements"""

        required_capabilities = task_requirements.get("capabilities", [])
        required_domains = task_requirements.get("domains", [])
        task_type = task_requirements.get("task_type", "general")

        # Score agents based on requirements
        agent_scores = {}

        for agent_id, agent in self.agents.items():
            score = 0.0

            # Capability match scoring
            if required_capabilities:
                capability_matches = len(set(required_capabilities) & set(agent.capabilities))
                capability_score = capability_matches / len(required_capabilities)
                score += capability_score * 0.4

            # Domain match scoring
            if required_domains:
                domain_matches = len(set(required_domains) & set(agent.domains))
                domain_score = domain_matches / len(required_domains)
                score += domain_score * 0.3

            # Success rate scoring
            score += agent.get_success_rate() * 0.3

            agent_scores[agent_id] = score

        # Sort by score and return top agents
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)

        # Return appropriate number of agents based on task type
        if task_type == "simple":
            return [sorted_agents[0][0]] if sorted_agents else []
        elif task_type == "complex":
            return [agent[0] for agent in sorted_agents[:3]]
        elif task_type == "comprehensive":
            return [agent[0] for agent in sorted_agents[:5]]
        else:
            return [agent[0] for agent in sorted_agents[:2]]

    def get_agent_workload(self) -> Dict[str, Dict[str, Any]]:
        """Get current workload status for all agents"""

        workload_info = {}
        for agent_id, agent in self.agents.items():
            workload_info[agent_id] = {
                "success_rate": agent.get_success_rate(),
                "capabilities": len(agent.capabilities),
                "domains": len(agent.domains),
                "specializations": len(agent.specializations),
                "performance_history_size": len(agent.performance_history),
                "status": "available"  # In production, would track actual workload
            }

        return workload_info

    async def smart_task_routing(self, task: Dict[str, Any]) -> str:
        """Intelligently route task to most appropriate agent"""

        task_description = task.get("description", "")
        task_type = task.get("task_type", "general")

        # Analyze task requirements
        requirements = {
            "capabilities": [],
            "domains": [],
            "task_type": "simple"
        }

        # Extract requirements from task description
        if "statistical" in task_description.lower():
            requirements["capabilities"].append("statistical_analysis")
            requirements["domains"].append("statistics")

        if "grant" in task_description.lower() or "proposal" in task_description.lower():
            requirements["capabilities"].append("grant_writing")
            requirements["domains"].append("grant_writing")

        if "hypothesis" in task_description.lower():
            requirements["capabilities"].append("hypothesis_generation")
            requirements["domains"].append("scientific_method")

        if "clinical" in task_description.lower() or "validation" in task_description.lower():
            requirements["capabilities"].append("clinical_validation")
            requirements["domains"].append("clinical_research")

        if "literature" in task_description.lower() or "review" in task_description.lower():
            requirements["capabilities"].append("literature_synthesis")
            requirements["domains"].append("scientific_literature")

        if "neuroscience" in task_description.lower() or "brain" in task_description.lower():
            requirements["domains"].append("neuroscience")

        # Determine complexity
        if len(requirements["capabilities"]) > 2 or "comprehensive" in task_description.lower():
            requirements["task_type"] = "comprehensive"
        elif len(requirements["capabilities"]) > 1 or "complex" in task_description.lower():
            requirements["task_type"] = "complex"

        # Get optimal agent team
        optimal_agents = self.get_optimal_agent_team(requirements)

        # Return best agent or team leader
        return optimal_agents[0] if optimal_agents else "neuroscience_expert"  # Fallback

    def get_collaboration_matrix(self) -> Dict[str, List[str]]:
        """Get collaboration compatibility matrix between agents"""

        collaboration_matrix = {}

        for agent_id, agent in self.agents.items():
            compatible_agents = []

            for other_agent_id, other_agent in self.agents.items():
                if agent_id != other_agent_id:
                    # Check capability complementarity
                    capability_overlap = len(set(agent.capabilities) & set(other_agent.capabilities))
                    domain_overlap = len(set(agent.domains) & set(other_agent.domains))

                    # Agents are compatible if they share domains but have different capabilities
                    if domain_overlap > 0 and capability_overlap < len(agent.capabilities):
                        compatible_agents.append(other_agent_id)

            collaboration_matrix[agent_id] = compatible_agents

        return collaboration_matrix
