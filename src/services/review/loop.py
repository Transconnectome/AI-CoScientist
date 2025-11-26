from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass
class ReviewResult:
    score: float
    comments: str
    passed: bool

@dataclass
class RevisionProposal:
    revised_text: str
    explanation: str

class AdversarialReviewLoop:
    def __init__(self, reviewer, defense):
        self.reviewer = reviewer
        self.defense = defense

    async def run_loop(self, draft: str, max_iterations: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        """Run the adversarial review loop."""
        current_draft = draft
        history = []
        
        for i in range(max_iterations):
            # 1. Reviewer critiques
            review = await self.reviewer.review(current_draft)
            
            history.append({
                "iteration": i + 1,
                "draft": current_draft,
                "score": review.score,
                "comments": review.comments,
                "passed": review.passed
            })
            
            if review.passed:
                break
                
            if i < max_iterations - 1:
                # 2. Defense proposes revision
                revision = await self.defense.propose_revision(current_draft, review.comments)
                current_draft = revision.revised_text
                
        return current_draft, history
