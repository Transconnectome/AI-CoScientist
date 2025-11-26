"""Defense agent for responding to criticisms and generating improvements."""

from typing import Dict, List, Optional
from src.core.config import settings


class DefenseAgent:
    """Generates responses to criticisms and proposes improvements."""
    
    def __init__(self, llm_service=None):
        """Initialize defense agent.
        
        Args:
            llm_service: Optional OpenAI AsyncClient for advanced analysis. If None, uses the global config client.
        """
        # Use provided client or fallback to config's OpenAI client
        self.llm_service = llm_service or getattr(settings, "openai_client", None)
    
    async def analyze_and_defend(self, criticism: Dict) -> Dict:
        """Analyze criticism and categorize by validity.
        
        If an LLM client is available, delegate to LLM for richer analysis.
        Otherwise fall back to heuristic logic.
        """
        # Use LLM if available
        if self.llm_service:
            try:
                return await self._analyze_and_defend_with_llm(criticism)
            except Exception as e:
                print(f"LLM analyze_and_defend failed: {e}, falling back to heuristic")
        
        # Heuristic fallback (original logic)
        weaknesses = criticism.get("weaknesses", [])
        severity_scores = criticism.get("severity_scores", {})
        
        valid_criticisms = []
        questionable_criticisms = []
        defense_strategies = []
        
        for weakness in weaknesses:
            severity = self._get_weakness_severity(weakness, severity_scores)
            if severity >= 7.0:
                valid_criticisms.append(weakness)
                strategy = self._generate_defense_strategy(weakness, "address")
                defense_strategies.append(strategy)
            elif severity >= 4.0:
                valid_criticisms.append(weakness)
                strategy = self._generate_defense_strategy(weakness, "clarify")
                defense_strategies.append(strategy)
            else:
                questionable_criticisms.append(weakness)
                strategy = self._generate_defense_strategy(weakness, "counter")
                defense_strategies.append(strategy)
        
        return {
            "valid_criticisms": valid_criticisms,
            "questionable_criticisms": questionable_criticisms,
            "defense_strategies": defense_strategies,
        }
    
    async def _analyze_and_defend_with_llm(self, criticism: Dict) -> Dict:
        """Use OpenAI LLM to analyze criticism and produce structured output.
        Expected return format matches the heuristic version.
        """
        prompt = (
            "You are an expert reviewer responding to a Reviewer #2 critique. "
            "Given the following criticism JSON, categorize each weakness as valid or questionable "
            "based on its severity, and propose a defense strategy (address, clarify, counter). "
            "Return a JSON object with keys 'valid_criticisms', 'questionable_criticisms', and 'defense_strategies'. "
            f"Criticism: {criticism}"
        )
        response = await self.llm_service.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            response_format={"type": "json_object"},
        )
        import json
        return json.loads(response.choices[0].message.content)
    
    async def generate_improvements(
        self,
        paper_content: str,
        criticism: Dict
    ) -> Dict:
        """Generate concrete improvements based on criticism.
        
        If an LLM client is available, delegate to LLM for richer suggestions.
        Otherwise fall back to heuristic generation.
        """
        if self.llm_service:
            try:
                return await self._generate_improvements_with_llm(paper_content, criticism)
            except Exception as e:
                print(f"LLM generate_improvements failed: {e}, falling back to heuristic")
        
        # Heuristic fallback (original logic)
        weaknesses = criticism.get("weaknesses", [])
        severity_scores = criticism.get("severity_scores", {})
        
        improvements = []
        
        for weakness in weaknesses:
            severity = self._get_weakness_severity(weakness, severity_scores)
            improvement = self._create_improvement(weakness, severity)
            improvements.append(improvement)
        
        improvements.sort(key=lambda x: x["expected_impact"], reverse=True)
        
        return {
            "improvements": improvements,
            "total_expected_improvement": sum(imp["expected_impact"] for imp in improvements[:3])
        }
    
    async def _generate_improvements_with_llm(self, paper_content: str, criticism: Dict) -> Dict:
        """Use OpenAI LLM to generate improvement suggestions.
        Returns same structure as heuristic version.
        """
        prompt = (
            "You are an expert author responding to reviewer criticisms. "
            "Given the paper content and the criticism JSON, propose concrete improvement actions. "
            "Return a JSON object with a list 'improvements', each containing 'description', 'expected_impact' (0-1), and 'priority' (High/Medium/Low). "
            f"Paper: {paper_content}\nCriticism: {criticism}"
        )
        response = await self.llm_service.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            response_format={"type": "json_object"},
        )
        import json
        return json.loads(response.choices[0].message.content)
    
    async def estimate_improvement_impact(
        self,
        current_score: float,
        target_score: float
    ) -> Dict:
        """Estimate iterations needed to reach target score.
        
        Args:
            current_score: Current quality score (0-10)
            target_score: Target quality score (0-10)
        
        Returns:
            Dict with:
                - estimated_iterations: Number of iterations needed
                - feasible: Whether target is achievable
        """
        score_gap = target_score - current_score
        
        if score_gap <= 0:
            return {
                "estimated_iterations": 0,
                "feasible": True,
                "note": "Already at or above target score"
            }
        
        # Conservative estimate: 0.3-0.5 points per iteration
        avg_gain_per_iteration = 0.4
        estimated_iterations = int(score_gap / avg_gain_per_iteration) + 1
        
        # Cap at 10 iterations - beyond that may not be feasible
        feasible = estimated_iterations <= 10
        
        return {
            "estimated_iterations": min(estimated_iterations, 10),
            "feasible": feasible,
            "score_gap": score_gap,
            "note": "Based on average 0.4 point gain per iteration"
        }
    
    def _get_weakness_severity(
        self,
        weakness: str,
        severity_scores: Dict[str, float]
    ) -> float:
        """Extract severity score for a weakness.
        
        Args:
            weakness: Weakness description
            severity_scores: Dict of severity scores
        
        Returns:
            Severity score (0-10)
        """
        # Try to match weakness to severity score
        for key, score in severity_scores.items():
            if key.lower() in weakness.lower():
                return score
        
        # Default severity based on keywords
        if any(word in weakness.lower() for word in ["critical", "major", "severe"]):
            return 8.0
        elif any(word in weakness.lower() for word in ["minor", "small"]):
            return 3.0
        else:
            return 5.0
    
    def _generate_defense_strategy(self, weakness: str, strategy_type: str) -> Dict:
        """Generate a defense strategy for a weakness.
        
        Args:
            weakness: The weakness to address
            strategy_type: Type of strategy ("address", "clarify", "counter")
        
        Returns:
            Defense strategy dict
        """
        strategies = {
            "address": {
                "approach": "Directly address and fix",
                "actions": ["Add missing content", "Strengthen methodology", "Provide evidence"]
            },
            "clarify": {
                "approach": "Clarify existing content",
                "actions": ["Reword for clarity", "Add examples", "Provide more detail"]
            },
            "counter": {
                "approach": "Provide counterargument",
                "actions": ["Explain rationale", "Cite supporting evidence", "Justify approach"]
            }
        }
        
        strategy = strategies.get(strategy_type, strategies["address"])
        
        return {
            "weakness": weakness,
            "strategy_type": strategy_type,
            "approach": strategy["approach"],
            "suggested_actions": strategy["actions"]
        }
    
    def _create_improvement(self, weakness: str, severity: float) -> Dict:
        """Create an improvement recommendation.
        
        Args:
            weakness: The weakness to improve
            severity: Severity score of the weakness
        
        Returns:
            Improvement dict with description, impact, and priority
        """
        # Map severity to expected impact
        if severity >= 8.0:
            expected_impact = min(1.0, severity / 10)  # Up to 1.0 for critical issues
            priority = "High"
        elif severity >= 6.0:
            expected_impact = min(0.6, severity / 15)
            priority = "Medium"
        else:
            expected_impact = min(0.3, severity / 20)
            priority = "Low"
        
        # Generate specific improvement description
        description = self._generate_improvement_description(weakness)
        
        return {
            "description": description,
            "expected_impact": round(expected_impact, 2),
            "priority": priority,
            "addresses_weakness": weakness
        }
    
    def _generate_improvement_description(self, weakness: str) -> str:
        """Generate specific improvement description.
        
        Args:
            weakness: The weakness to address
        
        Returns:
            Detailed improvement description
        """
        weakness_lower = weakness.lower()
        
        # Statistical/methodology improvements
        if "sample size" in weakness_lower:
            return "Add power analysis to justify sample size and discuss limitations if underpowered"
        elif "power analysis" in weakness_lower:
            return "Conduct and report statistical power analysis (α=0.05, β=0.80)"
        elif "effect size" in weakness_lower:
            return "Calculate and report effect sizes (Cohen's d or equivalent)"
        elif "statistical" in weakness_lower:
            return "Add comprehensive statistical analysis with appropriate tests and corrections"
        
        # Clarity improvements
        elif "vague" in weakness_lower:
            return "Replace vague language with precise, technical terminology"
        elif "missing sections" in weakness_lower or "section" in weakness_lower:
            return "Add or complete missing IMRaD sections (Introduction, Methods, Results, Discussion)"
        
        # Novelty improvements
        elif "novelty" in weakness_lower:
            return "Clearly articulate novel contributions and distinguish from prior work"
        
        # Significance improvements
        elif "impact" in weakness_lower or "significance" in weakness_lower:
            return "Add discussion of practical implications and broader impact"
        
        # Default improvement
        else:
            return f"Address identified weakness: {weakness}"
