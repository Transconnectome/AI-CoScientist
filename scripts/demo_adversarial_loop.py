#!/usr/bin/env python3
"""Demo: Adversarial Review Loop (Reviewer #2 vs Defense Agent)

Shows how a weak paper improves through iterative red team/blue team cycles.
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load API keys
load_dotenv()


async def demo_adversarial_loop():
    """Demonstrate the adversarial review loop."""
    
    print("=" * 80)
    print("⚔️  ADVERSARIAL REVIEW LOOP DEMO")
    print("=" * 80)
    print()
    
    # Weak paper that will be improved
    weak_paper = """
    Title: Machine Learning for Stock Price Prediction
    
    Abstract:
    We used LSTM neural networks to predict Bitcoin prices. Our model achieved
    99% accuracy. This is a breakthrough in financial prediction.
    
    Methods:
    We collected Bitcoin price data from 2020-2023. We trained an LSTM model
    on this data and tested it on the same dataset. The model learned the patterns.
    
    Results:
    The model achieved 99% accuracy on our test set. This proves that our approach works.
    
    Conclusion:
    Our model is perfect for predicting Bitcoin prices.
    """
    
    print("📄 Original (Weak) Paper:")
    print("-" * 80)
    print(weak_paper)
    print("-" * 80)
    print()
    
    # Try to import LLM service
    try:
        from src.services.llm.service import LLMService
        from src.services.review.adversarial_loop import AdversarialLoop
        
        llm_service = LLMService()
        loop = AdversarialLoop()
        
        print("🚀 Starting adversarial loop (max 3 iterations)...")
        print()
        
        result = await loop.run_loop(
            paper_text=weak_paper,
            llm_service=llm_service,
            max_iterations=3
        )
        
        print("\n" + "=" * 80)
        print("📊 RESULTS")
        print("=" * 80)
        print(f"Converged: {'✅ Yes' if result.converged else '❌ No'}")
        print(f"Total iterations: {len(result.iterations)}")
        print(f"Final recommendation: {result.final_recommendation.upper()}")
        print(f"Total improvements: {len(result.total_improvements)}")
        print()
        
        print("📝 Improvement History:")
        for imp in result.total_improvements[:10]:  # Show first 10
            print(f"  • {imp}")
        if len(result.total_improvements) > 10:
            print(f"  ... and {len(result.total_improvements) - 10} more")
        print()
        
        print("📄 Final Improved Paper:")
        print("-" * 80)
        print(result.final_paper[:500] + "..." if len(result.final_paper) > 500 else result.final_paper)
        print("-" * 80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📊 SIMULATED RESULTS (Demo Mode):")
        print("=" * 80)
        print("Iteration 1:")
        print("  👿 Reviewer: REJECT - Fatal flaw: data leakage")
        print("  🛡️  Defense: Added train/test split, power analysis")
        print()
        print("Iteration 2:")
        print("  👿 Reviewer: MAJOR REVISION - Missing code, baselines")
        print("  🛡️  Defense: Added GitHub repo, compared to ARIMA")
        print()
        print("Iteration 3:")
        print("  👿 Reviewer: ACCEPT - No fatal flaws")
        print("  ✅ CONVERGED!")


if __name__ == "__main__":
    asyncio.run(demo_adversarial_loop())
