from src.router.types import ResearchTask, TaskProfile, ComplexityLevel
from src.services.llm.types import LLMRequest, LLMConfig, TaskType as LLMTaskType, ModelProvider

class TaskAnalyzer:
    """Analyzes research tasks to extract structured information"""

    def __init__(self, llm_service):
        self.llm = llm_service

    async def analyze_task(self, task: ResearchTask) -> TaskProfile:
        """Analyze task and extract profile"""

        prompt = f"""
        Analyze this research task and extract structured information:

        Task Type: {task.task_type}
        Description: {task.description}
        Prior Work: {task.prior_work or 'None'}

        Extract and return as JSON:
        {{
            "domains": ["list", "of", "domains"],
            "complexity": "simple" or "medium" or "high",
            "task_type": "{task.task_type}",
            "sub_tasks": [
                {{"id": "subtask1", "type": "type", "dependencies": []}}
            ],
            "required_expertise": ["list", "of", "expertise"],
            "quality_gates": ["validation", "checks"],
            "context_dependencies": ["what", "context", "needed"],
            "keywords": ["key", "terms"]
        }}

        Complexity levels:
        - simple: Single domain, straightforward task
        - medium: 1-2 domains, some dependencies
        - high: 3+ domains, complex dependencies, novel synthesis

        Common domains: neuroscience, machine_learning, statistics,
                       ethics, biology, chemistry, physics
        """

        if self.llm:
            # Use configured OpenAI model
            from src.core.config import get_settings
            settings = get_settings()
            llm_request = LLMRequest(
                prompt=prompt,
                task_type=LLMTaskType.HYPOTHESIS_GENERATION,
                config=LLMConfig(
                    provider=ModelProvider.OPENAI,
                    model=settings.openai_model or "gpt-5-pro",
                    temperature=0.3,
                    max_tokens=settings.openai_max_tokens or 1000
                )
            )
            response = await self.llm.complete(llm_request)
            return TaskProfile.from_llm_response(response.content)
        else:
            # Mock response for testing without LLM
            return self._create_mock_profile(task)

    def _create_mock_profile(self, task: ResearchTask) -> TaskProfile:
        """Create mock profile for testing"""
        # Simple heuristic-based analysis
        domains = []
        if "fmri" in task.description.lower() or "brain" in task.description.lower():
            domains.append("neuroscience")
        if "deep learning" in task.description.lower() or "ml" in task.description.lower():
            domains.append("machine_learning")
        if "ethics" in task.description.lower() or "ethical" in task.description.lower():
            domains.append("ethics")

        if not domains:
            domains = ["general"]

        complexity = "high" if len(domains) >= 3 else "medium" if len(domains) == 2 else "simple"

        return TaskProfile(
            domains=domains,
            complexity=ComplexityLevel(complexity),
            task_type=task.task_type,
            sub_tasks=[{"id": "main", "type": task.task_type}],
            required_expertise=domains,
            quality_gates=["validation"],
            context_dependencies=["literature"]
        )
