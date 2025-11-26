import json
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class EditorDecision:
    decision: str  # "proceed_to_review" or "desk_reject"
    significance_score: float  # 0-10
    novelty_score: float  # 0-10
    feedback: str

class EditorAgent:
    """Simulates a top-tier journal editor (e.g., Nature/Science)."""

    EDITOR_PROMPT_TEMPLATE = """You are a Senior Editor at Nature.
    Your job is to screen papers for broad interest, conceptual novelty, and potential impact.
    You reject 90% of submissions without review.
    
    You look for:
    1. Conceptual Novelty: Does this change how we think about the world?
    2. Broad Interest: Is this relevant to scientists outside the immediate field?
    3. Timeliness: Is this a hot topic?
    
    Paper Abstract:
    {abstract}
    
    Introduction:
    {introduction}
    
    Make a decision:
    - "proceed_to_review": Only for exceptional papers (top 10%).
    - "desk_reject": For everything else.
    
    Return JSON:
    {{
        "decision": "proceed_to_review" | "desk_reject",
        "significance_score": float (0-10),
        "novelty_score": float (0-10),
        "feedback": "Brief explanation of decision"
    }}
    """

    async def evaluate_paper(self, abstract: str, introduction: str, llm_service) -> EditorDecision:
        prompt = self.EDITOR_PROMPT_TEMPLATE.format(abstract=abstract, introduction=introduction)
        try:
            response = await llm_service.complete(prompt, max_tokens=300)
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            data = json.loads(content.strip())
            
            return EditorDecision(
                decision=data.get("decision", "desk_reject"),
                significance_score=data.get("significance_score", 0.0),
                novelty_score=data.get("novelty_score", 0.0),
                feedback=data.get("feedback", "No feedback provided")
            )
        except Exception as e:
            return EditorDecision(
                decision="desk_reject",
                significance_score=0.0,
                novelty_score=0.0,
                feedback=f"Error parsing decision: {str(e)}"
            )
