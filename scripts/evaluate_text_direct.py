#!/usr/bin/env python3
"""Evaluate text directly using the ensemble scorer."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.paper.ensemble_scorer import EnsemblePaperScorer


async def evaluate_text(text_path: str):
    """Evaluate text file."""
    print("=" * 80)
    print("PAPER QUALITY EVALUATION")
    print("=" * 80)
    print()

    # Read text
    with open(text_path) as f:
        text = f.read()

    print(f"📄 Text length: {len(text)} characters")
    print()

    # Initialize scorer
    print("🔧 Initializing ensemble scorer...")
    scorer = EnsemblePaperScorer()
    print()

    # Evaluate
    print("🔍 Evaluating paper quality...")
    result = await scorer.score_paper(text, return_individual=True)
    print()

    # Display results
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()

    print(f"📊 Overall Quality: {result['overall']:.2f}/10")
    print(f"🎯 Confidence: {result['confidence']:.2f}")
    print()

    print("📈 Dimensional Scores:")
    dimensions = result.get('dimensions', {})

    if dimensions:
        for dim_name in ['novelty', 'methodology', 'clarity', 'significance']:
            score = dimensions.get(dim_name, 0.0)

            # Rating
            if score >= 9.5:
                rating = "🌟 Outstanding"
            elif score >= 9.0:
                rating = "⭐ Excellent"
            elif score >= 8.0:
                rating = "✅ Very Good"
            elif score >= 7.0:
                rating = "✅ Good"
            elif score >= 6.0:
                rating = "⚠️  Fair"
            else:
                rating = "🔴 Needs Work"

            print(f"  {dim_name.capitalize():15s}: {score:.2f}/10  {rating}")
    else:
        print("  ⚠️  Multi-dimensional scores not available")
    print()

    # Model agreement
    individual = result.get('individual_scores', {})
    if individual:
        print("🤝 Model Scores:")
        if individual.get('gpt4') is not None:
            print(f"  GPT-4 Model:      {individual['gpt4']:.2f}/10")
        if individual.get('hybrid') is not None:
            print(f"  Hybrid Model:     {individual['hybrid']:.2f}/10")
        if individual.get('multitask') is not None:
            print(f"  Multi-task Model: {individual['multitask']:.2f}/10")
        print()

    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_text_direct.py <path_to_txt>")
        sys.exit(1)

    text_path = sys.argv[1]
    asyncio.run(evaluate_text(text_path))
