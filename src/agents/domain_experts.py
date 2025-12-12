# src/agents/domain_experts.py
from src.agents.base import ResearchAgent
from src.agents.types import AgentTask, AgentResult

class NeuroscienceExpertAgent(ResearchAgent):
    """Domain expert in neuroscience"""

    def __init__(self, agent_id: str, llm_service, context_manager):
        super().__init__(agent_id, llm_service, context_manager)
        self.capabilities = [
            "domain_validation",
            "methodology_review",
            "literature_interpretation"
        ]
        self.domains = ["neuroscience", "fMRI", "brain_imaging"]
        self.specializations = ["emotion_recognition", "connectivity"]

    async def process(
        self,
        task: AgentTask,
        relevant_context: dict
    ) -> AgentResult:
        """Validate from neuroscience perspective"""

        # For now, minimal implementation
        prompt = f"""
        As a neuroscience expert, analyze:
        {task.description}

        Context: {relevant_context.get('summary', 'None')}

        Provide expert validation covering:
        1. Scientific validity
        2. Methodological appropriateness
        3. Potential limitations
        """

        if self.llm:
            response = await self.llm.complete(prompt)
            output = response.content
        else:
            output = "Mock neuroscience validation"

        return AgentResult(
            agent_id=self.agent_id,
            task_id=task.task_id,
            output=output,
            confidence=0.85
        )
