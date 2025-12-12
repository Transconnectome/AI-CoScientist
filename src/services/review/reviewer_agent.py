"""Adversarial reviewer agent that finds flaws in research papers."""

import json
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ReviewResult:
    """Result of adversarial review."""
    recommendation: str  # "reject", "major_revision", "accept"
    fatal_flaws: List[str]
    major_issues: List[str]
    minor_issues: List[str]
    overall_assessment: str


class ReviewerAgent:
    """Adversarial reviewer (#2) that finds reasons to reject papers."""
    
    REVIEWER_PROMPT_TEMPLATE = """You are the dreaded "Reviewer #2" - the most critical and skeptical peer reviewer.
Your goal is to find VALID scientific reasons to REJECT or request MAJOR REVISION.

You are grumpy, detail-oriented, and hate hype. You look for:
1. FATAL FLAWS: Fundamental errors in experimental design
   - Data leakage (testing on training data)
   - Confounding variables
   - Wrong statistical tests
   - P-hacking or HARKing

2. OVERCLAIMS: Unjustified "State of the Art" or "Novelty"
   - Minor variations presented as breakthroughs
   - Statistically insignificant improvements
   - Missing related work

3. MISSING BASELINES: Inadequate comparisons
   - Only compared against weak baselines
   - Missing standard methods

4. REPRODUCIBILITY FAILURES:
   - No code provided
   - Hyperparameters not listed
   - Data not available

Paper to review:
{paper_text}

Provide your review in JSON format:
{{
    "recommendation": "reject" | "major_revision" | "accept",
    "fatal_flaws": [<list of fatal flaws, if any>],
    "major_issues": [<list of major issues requiring revision>],
    "minor_issues": [<list of minor nitpicks>],
    "overall_assessment": "<1-2 sentence summary>"
}}

Be harsh but fair. Most papers should get "major_revision". Only truly broken papers get "reject".
Only exceptional papers with no flaws get "accept"."""

    async def review_paper(
        self,
        paper_text: str,
        llm_service
    ) -> ReviewResult:
        """Review paper and find flaws.
        
        Args:
            paper_text: Full paper text or abstract+intro
            llm_service: LLM service instance
            
        Returns:
            ReviewResult with recommendation and issues found
        """
        # Load Reviewer #2 persona from file if available
        try:
            with open("prompts/reviewer_2_persona.md", "r") as f:
                persona = f.read()
        except FileNotFoundError:
            persona = "You are Reviewer #2, a critical peer reviewer."
        
        # Format prompt
        prompt = self.REVIEWER_PROMPT_TEMPLATE.format(paper_text=paper_text)
        
        # Get LLM response
        response = await llm_service.get_response(
            system_prompt=persona,
            user_message=prompt
        )
        
        # Parse response
        try:
            result = self._parse_review(response)
            return result
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback to heuristic analysis
            return self._fallback_review(paper_text)
    
    def _parse_review(self, response: str) -> ReviewResult:
        """Parse LLM review response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            ReviewResult
        """
        # Extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
        else:
            data = json.loads(response)
        
        # Validate recommendation
        valid_recommendations = ["reject", "major_revision", "accept"]
        if data["recommendation"] not in valid_recommendations:
            data["recommendation"] = "major_revision"  # Default
        
        return ReviewResult(
            recommendation=data["recommendation"],
            fatal_flaws=data.get("fatal_flaws", []),
            major_issues=data.get("major_issues", []),
            minor_issues=data.get("minor_issues", []),
            overall_assessment=data.get("overall_assessment", "No assessment provided")
        )
    
    def _fallback_review(self, paper_text: str) -> ReviewResult:
        """Fallback heuristic review when LLM parsing fails.
        
        Args:
            paper_text: Paper text
            
        Returns:
            ReviewResult based on heuristics
        """
        text_lower = paper_text.lower()
        
        fatal_flaws = []
        major_issues = []
        minor_issues = []
        
        # Check for common fatal flaws
        if "training data" in text_lower and "test" in text_lower:
            if "same" in text_lower or "overfit" in text_lower:
                fatal_flaws.append("Possible data leakage: testing on training data")
        
        if "99%" in paper_text or "100%" in paper_text:
            fatal_flaws.append("Suspiciously high accuracy - likely overfitting")
        
        # Check for missing reproducibility elements
        if "github" not in text_lower and "code" not in text_lower:
            major_issues.append("No code repository provided")
        
        if "power analysis" not in text_lower:
            major_issues.append("No statistical power analysis")
        
        # Determine recommendation
        if len(fatal_flaws) > 0:
            recommendation = "reject"
        elif len(major_issues) >= 2:
            recommendation = "major_revision"
        else:
            recommendation = "accept"
        
        return ReviewResult(
            recommendation=recommendation,
            fatal_flaws=fatal_flaws,
            major_issues=major_issues,
            minor_issues=minor_issues,
            overall_assessment=f"Heuristic review: {len(fatal_flaws)} fatal flaws, {len(major_issues)} major issues"
        )
