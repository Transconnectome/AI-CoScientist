import json
from dataclasses import dataclass
from typing import List

@dataclass
class ExpertReview:
    technical_correctness: float  # 0-10
    methodological_flaws: List[str]
    sanity_check_passed: bool
    comments: str

class DomainExpertAgent:
    """Simulates a specialized domain expert (Reviewer #3)."""
    
    EXPERT_PROMPT_TEMPLATE = """You are a world-leading expert in {domain}.
    You are reviewing a paper for technical correctness and methodological rigor.
    You are less concerned with "impact" and more concerned with "is this true?".
    
    Check for:
    1. Technical correctness of claims.
    2. Appropriateness of methods.
    3. "Sanity checks" - do the numbers make sense?
    
    Paper Content:
    {content}
    
    Return JSON:
    {{
        "technical_correctness": float (0-10),
        "methodological_flaws": ["flaw 1", "flaw 2"],
        "sanity_check_passed": boolean,
        "comments": "Detailed technical comments"
    }}
    """

    async def review_paper(self, content: str, domain: str, llm_service) -> ExpertReview:
        # Limit content length to avoid context window issues
        prompt = self.EXPERT_PROMPT_TEMPLATE.format(domain=domain, content=content[:10000])
        try:
            response = await llm_service.complete(prompt, max_tokens=500)
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            data = json.loads(content.strip())
            
            return ExpertReview(
                technical_correctness=data.get("technical_correctness", 0.0),
                methodological_flaws=data.get("methodological_flaws", []),
                sanity_check_passed=data.get("sanity_check_passed", False),
                comments=data.get("comments", "No comments")
            )
        except Exception as e:
            return ExpertReview(
                technical_correctness=0.0,
                methodological_flaws=["Error parsing review"],
                sanity_check_passed=False,
                comments=f"Error: {str(e)}"
            )
