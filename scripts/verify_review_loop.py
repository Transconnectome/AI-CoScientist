import asyncio
import json
from src.services.review.loop import AdversarialReviewLoop
from src.services.review.agents import ReviewerAgent, DefenseAgent

# Mock LLM
class MockLLM:
    async def generate(self, prompt, **kwargs):
        if "Reviewer" in prompt:
            # Simulate reviewer
            return json.dumps({
                "score": 7.5,
                "comments": "Good start, but lacks detail in methods.",
                "passed": False
            }), "mock"
        elif "Defense" in prompt:
            # Simulate defense
            return json.dumps({
                "revised_text": "Revised text with detailed methods...",
                "explanation": "Added method details as requested."
            }), "mock"
        return "{}", "mock"

async def main():
    print("Starting Adversarial Review Loop Verification...")
    
    llm = MockLLM()
    reviewer = ReviewerAgent(llm)
    defense = DefenseAgent(llm)
    loop = AdversarialReviewLoop(reviewer, defense)
    
    initial_draft = "Initial draft text..."
    print(f"\nInitial Draft: {initial_draft}")
    
    final_draft, history = await loop.run_loop(initial_draft, max_iterations=2)
    
    print(f"\nFinal Draft: {final_draft}")
    print("\nHistory:")
    for step in history:
        print(f"  Iteration {step['iteration']}: Score {step['score']}, Passed: {step['passed']}")
        print(f"  Comments: {step['comments']}")
        
    if len(history) > 0:
        print("\n✅ Verification Successful!")
    else:
        print("\n❌ Verification Failed: No history generated.")

if __name__ == "__main__":
    asyncio.run(main())
