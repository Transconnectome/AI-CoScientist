"""Experiment design service."""

import json
import math
from typing import Any, Dict, List, Optional
from uuid import UUID

from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.models.project import Experiment, Hypothesis, ExperimentStatus
from src.services.llm import LLMService
from src.services.knowledge_base import KnowledgeBaseSearch


class ExperimentDesigner:
    """Service for designing scientific experiments."""

    def __init__(
        self,
        llm_service: LLMService,
        knowledge_base: KnowledgeBaseSearch,
        db: AsyncSession
    ):
        """Initialize experiment designer.

        Args:
            llm_service: LLM service for protocol generation
            knowledge_base: Knowledge base for methodology search
            db: Database session
        """
        self.llm = llm_service
        self.kb = knowledge_base
        self.db = db

    async def design_experiment(
        self,
        hypothesis_id: UUID,
        research_question: str,
        hypothesis_content: str,
        desired_power: float = 0.8,
        significance_level: float = 0.05,
        expected_effect_size: Optional[float] = None,
        constraints: Optional[Dict[str, Any]] = None,
        experimental_approach: Optional[str] = None
    ) -> Experiment:
        """Design an experiment for a hypothesis.

        Args:
            hypothesis_id: Hypothesis UUID
            research_question: Research question
            hypothesis_content: Hypothesis text
            desired_power: Desired statistical power (default 0.8)
            significance_level: Alpha level (default 0.05)
            expected_effect_size: Expected effect size (Cohen's d)
            constraints: Resource/time constraints
            experimental_approach: Suggested approach (optional)

        Returns:
            Designed experiment
        """
        # Verify hypothesis exists
        result = await self.db.execute(
            select(Hypothesis).where(Hypothesis.id == hypothesis_id)
        )
        hypothesis = result.scalar_one_or_none()
        if not hypothesis:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")

        # Search for similar experimental methods
        methodology_context = await self._search_methodologies(
            research_question, hypothesis_content
        )

        # Calculate sample size
        if expected_effect_size is None:
            expected_effect_size = 0.5  # Medium effect size

        sample_size = self._calculate_sample_size(
            effect_size=expected_effect_size,
            power=desired_power,
            alpha=significance_level
        )

        # Generate protocol using LLM
        protocol_prompt = self._build_protocol_prompt(
            research_question=research_question,
            hypothesis=hypothesis_content,
            methodology_context=methodology_context,
            sample_size=sample_size,
            constraints=constraints,
            approach=experimental_approach
        )

        protocol_response = await self.llm.generate(
            prompt=protocol_prompt,
            temperature=0.7,
            max_tokens=2000
        )

        protocol_data = self._parse_protocol_response(protocol_response)

        # Create experiment
        experiment = Experiment(
            hypothesis_id=hypothesis_id,
            title=protocol_data["title"],
            protocol=protocol_data["protocol"],
            sample_size=sample_size,
            power=desired_power,
            effect_size=expected_effect_size,
            significance_level=significance_level,
            status=ExperimentStatus.DESIGNED.value
        )

        self.db.add(experiment)
        await self.db.commit()
        await self.db.refresh(experiment)

        return experiment

    async def _search_methodologies(
        self,
        research_question: str,
        hypothesis: str
    ) -> str:
        """Search for relevant experimental methodologies.

        Args:
            research_question: Research question
            hypothesis: Hypothesis content

        Returns:
            Methodology context string
        """
        query = f"{research_question} {hypothesis} experimental methods"

        results = await self.kb.semantic_search(
            query=query,
            top_k=5,
            filters={"source_type": "methodology"}
        )

        if not results:
            return "No specific methodology found. Use standard experimental design."

        context_parts = []
        for result in results[:3]:
            context_parts.append(
                f"Paper: {result.title}\n"
                f"Methods: {result.abstract[:300]}\n"
            )

        return "\n\n".join(context_parts)

    def _calculate_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05
    ) -> int:
        """Calculate required sample size for two-sample t-test.

        Args:
            effect_size: Expected effect size (Cohen's d)
            power: Desired statistical power
            alpha: Significance level

        Returns:
            Required sample size per group
        """
        # Using approximation for two-sample t-test
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        # Round up and add 10% buffer
        return math.ceil(n * 1.1)

    def calculate_power(
        self,
        effect_size: float,
        sample_size: int,
        alpha: float = 0.05
    ) -> float:
        """Calculate statistical power given parameters.

        Args:
            effect_size: Effect size (Cohen's d)
            sample_size: Sample size per group
            alpha: Significance level

        Returns:
            Statistical power
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        ncp = effect_size * math.sqrt(sample_size / 2)

        power = 1 - stats.norm.cdf(z_alpha - ncp)
        return min(power, 0.99)

    def _build_protocol_prompt(
        self,
        research_question: str,
        hypothesis: str,
        methodology_context: str,
        sample_size: int,
        constraints: Optional[Dict[str, Any]] = None,
        approach: Optional[str] = None
    ) -> str:
        """Build prompt for protocol generation.

        Args:
            research_question: Research question
            hypothesis: Hypothesis content
            methodology_context: Literature methodology context
            sample_size: Calculated sample size
            constraints: Resource constraints
            approach: Suggested approach

        Returns:
            Protocol generation prompt
        """
        prompt = f"""Design a detailed experimental protocol for the following research:

Research Question: {research_question}

Hypothesis: {hypothesis}

Sample Size: {sample_size} per group (calculated for adequate statistical power)

"""
        if approach:
            prompt += f"Suggested Approach: {approach}\n\n"

        if constraints:
            prompt += f"Constraints: {json.dumps(constraints, indent=2)}\n\n"

        prompt += f"""Relevant Methodologies from Literature:
{methodology_context}

Generate a comprehensive experimental protocol in JSON format with the following structure:
{{
    "title": "Experiment title",
    "protocol": "Detailed step-by-step protocol",
    "methods": ["Method 1", "Method 2"],
    "materials": ["Material 1", "Material 2"],
    "variables": {{
        "independent": ["IV1", "IV2"],
        "dependent": ["DV1", "DV2"],
        "controlled": ["CV1", "CV2"]
    }},
    "data_collection": "Data collection procedure",
    "statistical_analysis": "Statistical analysis plan",
    "potential_confounds": ["Confound 1", "Confound 2"],
    "mitigation_strategies": ["Strategy 1", "Strategy 2"],
    "estimated_duration": "Time estimate",
    "resource_requirements": {{"type": "amount"}}
}}

Be specific, rigorous, and follow best practices in experimental design.
"""
        return prompt

    def _parse_protocol_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM protocol response.

        Args:
            response: LLM response

        Returns:
            Parsed protocol data
        """
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return data
            else:
                # Fallback if no JSON found
                return {
                    "title": "Untitled Experiment",
                    "protocol": response,
                    "methods": [],
                    "materials": [],
                    "variables": {},
                    "data_collection": "",
                    "statistical_analysis": "",
                    "potential_confounds": [],
                    "mitigation_strategies": [],
                    "estimated_duration": "Unknown",
                    "resource_requirements": {}
                }
        except json.JSONDecodeError:
            # Fallback for malformed JSON
            return {
                "title": "Generated Experiment",
                "protocol": response,
                "methods": [],
                "materials": [],
                "variables": {},
                "data_collection": "",
                "statistical_analysis": "",
                "potential_confounds": [],
                "mitigation_strategies": [],
                "estimated_duration": "Unknown",
                "resource_requirements": {}
            }

    async def optimize_variables(
        self,
        experiment_id: UUID,
        objective: str,
        variable_ranges: Dict[str, tuple]
    ) -> Dict[str, Any]:
        """Optimize experimental variables.

        Args:
            experiment_id: Experiment UUID
            objective: Optimization objective
            variable_ranges: Variable ranges for optimization

        Returns:
            Optimized variable settings
        """
        # Placeholder for variable optimization
        # In production, implement DOE or Bayesian optimization
        result = await self.db.execute(
            select(Experiment).where(Experiment.id == experiment_id)
        )
        experiment = result.scalar_one_or_none()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        optimization_prompt = f"""Analyze the following experimental design and suggest optimal variable settings:

Protocol: {experiment.protocol}

Objective: {objective}

Variable Ranges: {json.dumps(variable_ranges, indent=2)}

Suggest optimal settings for each variable to maximize {objective}.
Provide reasoning for each recommendation.
"""

        response = await self.llm.generate(
            prompt=optimization_prompt,
            temperature=0.5,
            max_tokens=1000
        )

        return {
            "recommendations": response,
            "objective": objective,
            "variable_ranges": variable_ranges
        }
