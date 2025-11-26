"""Adversarial reviewer for critically evaluating papers."""

from typing import Dict, List, Optional
from src.core.config import settings
import re


class AdversarialReviewer:
    """Simulates a critical 'Reviewer #2' that identifies paper weaknesses."""
    
    def __init__(self, llm_service=None):
        """Initialize adversarial reviewer.

        Args:
            llm_service: Optional OpenAI AsyncClient for advanced review. If None, uses the global config client.
        """
        # Use provided client or fallback to config's OpenAI client
        self.llm_service = llm_service or getattr(settings, "openai_client", None)
    
    async def review(
        self,
        paper_content: str,
        dimensions: Optional[List[str]] = None
    ) -> Dict:
        """Conduct adversarial review of paper.

        If an LLM client is available, delegate to LLM for a richer review.
        Otherwise fall back to heuristic analysis.
        """
        # If we have an LLM service, use it first
        if self.llm_service:
            try:
                return await self._review_with_llm(paper_content, dimensions)
            except Exception as e:
                # Log the error (placeholder) and fall back to heuristic
                print(f"LLM review failed: {e}, falling back to heuristic")

        # Heuristic fallback (original logic)
        if dimensions is None:
            dimensions = ["methodology", "clarity", "novelty", "significance"]

        weaknesses = []
        severity_scores = {}
        dimension_reviews = {}

        # Analyze each dimension
        for dimension in dimensions:
            dimension_analysis = self._analyze_dimension(paper_content, dimension)
            dimension_reviews[dimension] = dimension_analysis
            
            # Extract weaknesses from dimension analysis
            if dimension_analysis["issues"]:
                weaknesses.extend(dimension_analysis["issues"])
                for issue_type, severity in dimension_analysis["severities"].items():
                    severity_scores[issue_type] = severity
        
        # Generate overall feedback
        feedback = self._generate_feedback(weaknesses, severity_scores)
        
        return {
            "weaknesses": weaknesses,
            "severity_scores": severity_scores,
            "feedback": feedback,
            "dimension_reviews": dimension_reviews,
        }

    async def _review_with_llm(self, paper_content: str, dimensions: Optional[List[str]]) -> Dict:
        """Use OpenAI LLM to perform a structured review.
        Returns a dict with the same keys as the heuristic version.
        """
        if dimensions is None:
            dimensions = ["methodology", "clarity", "novelty", "significance"]
        prompt = (
            "You are an expert reviewer (Reviewer #2). Analyze the following scientific paper and provide a JSON response with the keys: "
            "'weaknesses' (list of strings), 'severity_scores' (dict of issue->score 0-10), "
            "'feedback' (string), and 'dimension_reviews' (dict of dimension->analysis dict). "
            "Focus on the dimensions: " + ", ".join(dimensions) + ".\n\nPaper:\n" + paper_content
        )
        # Call the OpenAI chat completion API
        response = await self.llm_service.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            response_format={"type": "json_object"},
        )
        # Parse JSON content
        import json
        result = json.loads(response.choices[0].message.content)
        return result
    
    def _analyze_dimension(self, content: str, dimension: str) -> Dict:
        """Analyze a specific quality dimension.
        
        Args:
            content: Paper content
            dimension: Dimension to analyze
        
        Returns:
            Dict with issues and severities for this dimension
        """
        issues = []
        severities = {}
        
        content_lower = content.lower()
        
        if dimension == "methodology":
            # Check for statistical rigor
            if "sample size" in content_lower or "n=" in content_lower:
                # Extract sample size
                match = re.search(r'n\s*=\s*(\d+)', content_lower)
                if match:
                    n = int(match.group(1))
                    if n < 20:
                        issues.append(f"Sample size too small (n={n})")
                        severities["sample_size"] = 8.0
            
            # Check for power analysis
            if "power analysis" not in content_lower and "power" not in content_lower:
                issues.append("No statistical power analysis")
                severities["power_analysis"] = 7.0
            
            # Check for effect size
            if "effect size" not in content_lower and "cohen" not in content_lower:
                issues.append("Missing effect size calculations")
                severities["effect_size"] = 6.0
            
            # Check for unrealistic claims
            if "100% success" in content_lower or "cure cancer" in content_lower:
                issues.append("Unrealistic or unsubstantiated claims")
                severities["unrealistic_claims"] = 10.0
            
            # Check for lack of statistical analysis
            if "statistical analysis" not in content_lower and "p-value" not in content_lower and "p <" not in content_lower:
                issues.append("No statistical analysis performed")
                severities["no_statistics"] = 9.0
        
        elif dimension == "clarity":
            # Check for vague language
            vague_terms = ["stuff", "things", "good", "cool", "nice"]
            for term in vague_terms:
                if term in content_lower:
                    issues.append(f"Vague language: '{term}'")
                    severities[f"vague_{term}"] = 5.0
            
            # Check for section completeness
            sections = ["introduction", "methods", "results", "discussion"]
            missing_sections = [s for s in sections if s not in content_lower]
            if len(missing_sections) > 2:
                issues.append(f"Missing key sections: {', '.join(missing_sections)}")
                severities["missing_sections"] = 7.0
        
        elif dimension == "novelty":
            # Check for novelty claims
            if "novel" not in content_lower and "first" not in content_lower and "new" not in content_lower:
                issues.append("No clear novelty claim")
                severities["no_novelty"] = 6.0
        
        elif dimension == "significance":
            # Check for impact discussion
            if "impact" not in content_lower and "significance" not in content_lower:
                issues.append("No discussion of significance or impact")
                severities["no_impact"] = 6.0
        
        return {
            "issues": issues,
            "severities": severities,
            "score": max(0, 10 - len(issues) * 1.5)  # Deduct 1.5 points per issue
        }
    
    def _generate_feedback(self, weaknesses: List[str], severity_scores: Dict[str, float]) -> str:
        """Generate detailed feedback text.
        
        Args:
            weaknesses: List of identified weaknesses
            severity_scores: Severity scores for each issue
        
        Returns:
            Detailed feedback string
        """
        if not weaknesses:
            return "No major weaknesses identified. Paper meets basic quality standards."
        
        # Sort weaknesses by severity (highest first)
        severity_map = {w: severity_scores.get(w.split()[0].lower(), 5.0) for w in weaknesses}
        sorted_weaknesses = sorted(weaknesses, key=lambda w: severity_map.get(w, 0), reverse=True)
        
        feedback_parts = [
            "CRITICAL REVIEW:",
            "",
            f"This paper has {len(weaknesses)} identified weaknesses that need to be addressed:",
            ""
        ]
        
        for i, weakness in enumerate(sorted_weaknesses[:5], 1):  # Top 5 issues
            severity = severity_map.get(weakness, 5.0)
            severity_label = "CRITICAL" if severity >= 8 else "MAJOR" if severity >= 6 else "MINOR"
            feedback_parts.append(f"{i}. [{severity_label}] {weakness}")
        
        feedback_parts.extend([
            "",
            "RECOMMENDATION:",
            "The authors must address these issues before publication can be considered.",
            "Particular attention should be paid to the highest-severity items."
        ])
        
        return "\n".join(feedback_parts)
