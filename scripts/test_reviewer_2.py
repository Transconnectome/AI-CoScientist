import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mocking the interaction since we can't import LLMService due to missing dependencies
async def test_reviewer_2():
    """Test the Reviewer #2 persona."""
    print("👿 Summoning Reviewer #2...")
    
    # Load prompt
    prompt_path = Path("prompts/reviewer_2_persona.md")
    if not prompt_path.exists():
        print("Error: Prompt file not found!")
        return

    system_prompt = prompt_path.read_text()
    print(f"✅ Loaded System Prompt ({len(system_prompt)} chars)")
    
    # Dummy weak paper
    weak_paper = """
    Title: A New Method for Predicting Stock Prices
    Abstract: We used a LSTM model to predict Bitcoin prices. We got 99% accuracy.
    Method: We trained on data from 2020-2023 and tested on the same data.
    Results: The model is perfect.
    """
    
    print(f"\n📄 Submitting Weak Paper:\n{weak_paper}\n")
    
    print("... Sending to LLM (Simulated) ...")
    
    print("\n💬 Reviewer #2 Response (Simulation):")
    print("-" * 40)
    print("Recommendation: REJECT")
    print("Summary: This paper is fundamentally flawed and represents a regression in the field.")
    print("Major Points:")
    print("1. FATAL FLAW: The authors tested on the training data ('tested on the same data'). This is Data Science 101. The 99% accuracy is meaningless overfitting.")
    print("2. Overclaims: 'The model is perfect' is an unscientific statement.")
    print("3. Missing Baselines: No comparison to Buy & Hold or ARIMA.")
    print("-" * 40)

if __name__ == "__main__":
    asyncio.run(test_reviewer_2())
