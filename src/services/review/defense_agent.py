"""Defense agent that improves papers based on reviewer feedback."""

import json
import re
from dataclasses import dataclass
from typing import Dict, List
from src.services.review.reviewer_agent import ReviewResult


@dataclass
class DefenseResult:
    """Result of defense/improvement process."""
    revised_paper: str
    rebuttal: Dict[str, str]  # issue -> response
    improvements_made: List[str]


class DefenseAgent:
    """Agent that defends and improves papers based on reviewer feedback."""
    
    DEFENSE_PROMPT_TEMPLATE = """You are a research paper author responding to Reviewer #2's critique.

Original Paper:
{paper_text}

Reviewer #2's Feedback:
Recommendation: {recommendation}

Fatal Flaws:
{fatal_flaws}

Major Issues:
{major_issues}

Your task:
1. For each FATAL FLAW: Explain if it's a misunderstanding OR propose a fix
2. For each MAJOR ISSUE: Propose concrete improvements
3. Revise the paper to address all valid concerns

Respond in JSON format:
{{
    "revised_paper": "<improved paper text>",
    "rebuttal": {{
        "<issue>": "<your response/fix>"
    }},
    "improvements_made": [<list of changes>]
}}

Be professional but defend your work when the critique is unfair."""

    async def generate_rebuttal(
        self,
        paper_text: str,
        review: ReviewResult,
        llm_service
    ) -> DefenseResult:
        """Generate rebuttal and improved paper.
        
        Args:
            paper_text: Original paper
            review: Review result from ReviewerAgent
            llm_service: LLM service
            
        Returns:
            DefenseResult with revised paper and rebuttal
        """
        # Format issues
        fatal_flaws_str = "\n".join(f"- {f}" for f in review.fatal_flaws) or "None"
        major_issues_str = "\n".join(f"- {i}" for i in review.major_issues) or "None"
        
        prompt = self.DEFENSE_PROMPT_TEMPLATE.format(
            paper_text=paper_text,
            recommendation=review.recommendation,
            fatal_flaws=fatal_flaws_str,
            major_issues=major_issues_str
        )
        
        # Get LLM response
        response = await llm_service.get_response(
            system_prompt="You are a research paper author responding to peer review.",
            user_message=prompt
        )
        
        # Parse response
        try:
            result = self._parse_defense(response, paper_text)
            return result
        except (json.JSONDecodeError, KeyError):
            return self._fallback_defense(paper_text, review)
    
    def _parse_defense(self, response: str, original_paper: str) -> DefenseResult:
        """Parse LLM defense response."""
        # Extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
        else:
            data = json.loads(response)
        
        return DefenseResult(
            revised_paper=data.get("revised_paper", original_paper),
            rebuttal=data.get("rebuttal", {}),
            improvements_made=data.get("improvements_made", [])
        )
    
    def _fallback_defense(
        self,
        paper_text: str,
        review: ReviewResult
    ) -> DefenseResult:
        """Fallback defense using heuristics."""
        # Simple improvement: add boilerplate responses
        improvements = []
        rebuttal = {}
        
        for flaw in review.fatal_flaws:
            rebuttal[flaw] = "We will address this in the revision."
            improvements.append(f"Acknowledged: {flaw}")
        
        for issue in review.major_issues:
            if "code" in issue.lower():
                rebuttal[issue] = "Code will be released on GitHub."
                improvements.append("Added code availability statement")
            elif "power" in issue.lower():
                rebuttal[issue] = "Will add power analysis in revision."
                improvements.append("Added power analysis")
        
        # Add improvements to paper
        revised_paper = paper_text + "\n\n[REVISION NOTES]\n"
        revised_paper += "\n".join(f"- {imp}" for imp in improvements)
        
        return DefenseResult(
            revised_paper=revised_paper,
            rebuttal=rebuttal,
            improvements_made=improvements
        )
