"""Adversarial loop orchestrator: Red Team vs Blue Team iterations."""

from dataclasses import dataclass
from typing import Optional, List
from src.services.review.reviewer_agent import ReviewerAgent, ReviewResult
from src.services.review.defense_agent import DefenseAgent, DefenseResult


@dataclass
class LoopIteration:
    """Single iteration of the adversarial loop."""
    iteration: int
    review: ReviewResult
    defense: Optional[DefenseResult]
    paper_version: str


@dataclass
class LoopResult:
    """Result of complete adversarial loop."""
    final_paper: str
    iterations: List[LoopIteration]
    converged: bool
    final_recommendation: str
    total_improvements: List[str]


class AdversarialLoop:
    """Orchestrates Red Team (Reviewer #2) vs Blue Team (Defense) iterations."""
    
    def __init__(self):
        self.reviewer = ReviewerAgent()
        self.defender = DefenseAgent()
    
    async def run_loop(
        self,
        paper_text: str,
        llm_service,
        max_iterations: int = 3,
        convergence_threshold: float = 0.8
    ) -> LoopResult:
        """Run iterative attack-defend loop until convergence.
        
        Args:
            paper_text: Initial paper text
            llm_service: LLM service instance
            max_iterations: Maximum number of iterations (default: 3)
            convergence_threshold: Not used yet (placeholder for quality score)
            
        Returns:
            LoopResult with final paper and iteration history
            
        Convergence criteria:
        - Reviewer gives "accept" recommendation
        - No fatal flaws for 2 consecutive iterations
        - Max iterations reached
        """
        iterations = []
        current_paper = paper_text
        total_improvements = []
        
        consecutive_no_fatal_flaws = 0
        
        for i in range(max_iterations):
            print(f"🔄 Adversarial Loop - Iteration {i+1}/{max_iterations}")
            
            # Red Team: Review
            print("   👿 Reviewer #2 attacking...")
            review = await self.reviewer.review_paper(current_paper, llm_service)
            
            print(f"   📋 Recommendation: {review.recommendation.upper()}")
            print(f"   ⚠️  Fatal flaws: {len(review.fatal_flaws)}")
            print(f"   ⚡ Major issues: {len(review.major_issues)}")
            
            # Check convergence
            if review.recommendation == "accept":
                print("   ✅ CONVERGED: Reviewer accepted!")
                iterations.append(LoopIteration(
                    iteration=i+1,
                    review=review,
                    defense=None,
                    paper_version=current_paper
                ))
                break
            
            # Track fatal flaw streak
            if len(review.fatal_flaws) == 0:
                consecutive_no_fatal_flaws += 1
            else:
                consecutive_no_fatal_flaws = 0
            
            if consecutive_no_fatal_flaws >= 2:
                print("   ✅ CONVERGED: No fatal flaws for 2 iterations!")
                iterations.append(LoopIteration(
                    iteration=i+1,
                    review=review,
                    defense=None,
                    paper_version=current_paper
                ))
                break
            
            # Blue Team: Defend and improve
            print("   🛡️  Defense agent responding...")
            defense = await self.defender.generate_rebuttal\
(current_paper, review, llm_service)
            
            print(f"   ✏️  Improvements made: {len(defense.improvements_made)}")
            
            # Update paper
            current_paper = defense.revised_paper
            total_improvements.extend(defense.improvements_made)
            
            # Record iteration
            iterations.append(LoopIteration(
                iteration=i+1,
                review=review,
                defense=defense,
                paper_version=current_paper
            ))
        
        # Determine if converged
        converged = (
            len(iterations) > 0 and
            (iterations[-1].review.recommendation == "accept" or
             consecutive_no_fatal_flaws >= 2)
        )
        
        final_recommendation = iterations[-1].review.recommendation if iterations else "unknown"
        
        print(f"\n{'✅' if converged else '⏱️'} Loop {'converged' if converged else 'completed'} after {len(iterations)} iterations")
        
        return LoopResult(
            final_paper=current_paper,
            iterations=iterations,
            converged=converged,
            final_recommendation=final_recommendation,
            total_improvements=total_improvements
        )
