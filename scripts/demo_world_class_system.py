#!/usr/bin/env python3
"""Demo: World-Class Paper Evaluation System

Demonstrates the complete evaluation pipeline with:
- Traditional metrics (Novelty, Methodology, Clarity, Significance)
- Advanced metrics (Reproducibility, Narrative)
- Ensemble scoring (Hybrid + Multitask models)
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def demo_world_class_evaluation():
    """Demonstrate the world-class evaluation system."""
    
    print("=" * 80)
    print("🏆 WORLD-CLASS PAPER EVALUATION SYSTEM DEMO")
    print("=" * 80)
    print()
    
    # Sample papers for comparison
    excellent_paper = """
    Novel Deep Learning Architecture for fMRI Analysis
    
    Abstract: This paper introduces a breakthrough transformer-based architecture achieving 
    state-of-the-art performance in brain decoding tasks. Our method demonstrates 30% 
    improvement over previous approaches.
    
    Introduction:
    fMRI analysis has advanced significantly. However, reproducibility remains a critical 
    challenge in the field. We address this by...
    
    Methods:
    Statistical power analysis was conducted using G*Power 3.1 (alpha=0.05, power=0.8).
    Effect sizes were computed (Cohen's d = 0.85, 95% CI [0.72, 0.98]).
    All code is available at https://github.com/example/fmri-transformer.
    Data repository: https://zenodo.org/record/123456.
    
    Results:
    Our model achieved 89% accuracy (SD=2.3%, N=120 subjects).
    """
    
    weak_paper = """
    Machine Learning for Stock Prediction
    
    Abstract: We used LSTM to predict stocks. Results are good.
    
    Methods: We trained a model.
    
    Results: 99% accuracy on training data.
    """
    
    # Import ensemble scorer
    try:
        from src.services.paper.ensemble_scorer import EnsemblePaperScorer
        
        # Initialize scorer with advanced metrics enabled
        print("🔧 Initializing Ensemble Scorer...")
        scorer = EnsemblePaperScorer(
            gpt4_weight=0.0,  # Disable GPT-4 for demo (no API key needed)
            hybrid_weight=0.5,
            multitask_weight=0.5,
            use_gpt4=False,
            include_advanced_metrics=True,
            advanced_metrics_weight=0.15  # 15% weight for advanced metrics
        )
        print("✅ Scorer initialized\n")
        
        # Evaluate excellent paper
        print("=" * 80)
        print("📄 EVALUATING: Excellent Paper (High Reproducibility)")
        print("=" * 80)
        print(f"Preview: {excellent_paper[:150]}...\n")
        
        result_excellent = await scorer.score_paper(excellent_paper, return_individual=True)
        
        print("📊 RESULTS:")
        print(f"   Overall Score: {result_excellent['overall']:.2f} / 10")
        print(f"   Confidence:    {result_excellent['confidence']:.2f}")
        print()
        
        if "dimensions" in result_excellent:
            print("   Traditional Dimensions:")
            for dim, score in result_excellent["dimensions"].items():
                print(f"     • {dim.capitalize():15s}: {score:.2f}")
            print()
        
        if "advanced_metrics" in result_excellent:
            print("   Advanced Metrics:")
            print(f"     • Reproducibility:   {result_excellent['advanced_metrics']['reproducibility']:.2f}")
            print(f"     • Narrative Hook:    {result_excellent['advanced_metrics']['narrative']['hook_score']:.2f}")
            print(f"     • Feedback: {result_excellent['advanced_metrics']['narrative']['feedback']}")
            print()
        
        # Evaluate weak paper
        print("=" * 80)
        print("📄 EVALUATING: Weak Paper (Low Reproducibility)")
        print("=" * 80)
        print(f"Preview: {weak_paper[:150]}...\n")
        
        result_weak = await scorer.score_paper(weak_paper, return_individual=True)
        
        print("📊 RESULTS:")
        print(f"   Overall Score: {result_weak['overall']:.2f} / 10")
        print(f"   Confidence:    {result_weak['confidence']:.2f}")
        print()
        
        if "advanced_metrics" in result_weak:
            print("   Advanced Metrics:")
            print(f"     • Reproducibility:   {result_weak['advanced_metrics']['reproducibility']:.2f} ⚠️  LOW")
            print(f"     • Narrative Hook:    {result_weak['advanced_metrics']['narrative']['hook_score']:.2f}")
            print()
        
        # Comparison
        print("=" * 80)
        print("📈 COMPARISON SUMMARY")
        print("=" * 80)
        score_diff = result_excellent['overall'] - result_weak['overall']
        repro_diff = result_excellent['advanced_metrics']['reproducibility'] - result_weak['advanced_metrics']['reproducibility']
        
        print(f"   Overall Score Difference:        {score_diff:+.2f}")
        print(f"   Reproducibility Difference:      {repro_diff:+.2f}")
        print()
        print("   💡 Insight: The excellent paper scores higher due to:")
        print("      - GitHub repository link (+3.0)")
        print("      - Statistical power analysis (+2.0)")
        print("      - Effect size reporting (+2.0)")
        print("      - Data repository link (+bonus)")
        print()
        
        print("=" * 80)
        print("✅ DEMO COMPLETE!")
        print("=" * 80)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Note: This demo requires the models to be present.")
        print("Continuing with simulated output...")
        
        print("\n📊 SIMULATED RESULTS:")
        print("   Excellent Paper: 8.5/10 (High reproducibility: 9.0)")
        print("   Weak Paper:      5.2/10 (Low reproducibility: 2.0)")


if __name__ == "__main__":
    asyncio.run(demo_world_class_evaluation())
