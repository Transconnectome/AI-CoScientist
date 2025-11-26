from typing import Tuple
from src.services.review.loop import ReviewResult, RevisionProposal
import json

class ReviewerAgent:
    def __init__(self, llm):
        self.llm = llm

    async def review(self, text: str) -> ReviewResult:
        """Review the text and provide a score and comments."""
        prompt = f"""
        Act as "Reviewer #2" for a top-tier journal (Nature/Science).
        Critique the following text rigorously.
        
        Text:
        {text[:5000]}
        
        Return a JSON object with:
        - "score": float (0-10)
        - "comments": string (detailed critique)
        - "passed": boolean (true if score >= 8.0)
        """
        
        response, _ = await self.llm.generate(prompt)
        try:
            # Clean up response
            json_str = response.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:-3]
            elif json_str.startswith('```'):
                json_str = json_str[3:-3]
                
            data = json.loads(json_str)
            return ReviewResult(
                score=float(data.get("score", 0)),
                comments=data.get("comments", "No comments"),
                passed=bool(data.get("passed", False))
            )
        except Exception as e:
            return ReviewResult(score=0.0, comments=f"Error parsing review: {e}", passed=False)

class DefenseAgent:
    def __init__(self, llm):
        self.llm = llm

    async def propose_revision(self, original_text: str, comments: str) -> RevisionProposal:
        """Propose a revision based on reviewer comments."""
        prompt = f"""
        Act as a "Defense Agent" improving a paper draft.
        
        Original Text:
        {original_text[:5000]}
        
        Reviewer Comments:
        {comments}
        
        Task: Rewrite the text to address the comments while maintaining the original intent and scientific accuracy.
        
        Return a JSON object with:
        - "revised_text": string
        - "explanation": string (what you changed and why)
        """
        
        response, _ = await self.llm.generate(prompt)
        try:
            json_str = response.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:-3]
            elif json_str.startswith('```'):
                json_str = json_str[3:-3]
                
            data = json.loads(json_str)
            return RevisionProposal(
                revised_text=data.get("revised_text", original_text),
                explanation=data.get("explanation", "No explanation")
            )
        except Exception as e:
            return RevisionProposal(revised_text=original_text, explanation=f"Error parsing revision: {e}")
