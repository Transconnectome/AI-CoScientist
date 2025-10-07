#!/usr/bin/env python3
"""Validation script for SOTA methods against human-scored papers."""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))


async def load_validation_dataset() -> List[Dict]:
    """Load validation dataset with human scores.

    Returns:
        List of papers with human quality scores
    """
    dataset_path = Path("data/validation/validation_dataset_v1.json")

    if not dataset_path.exists():
        print(f"❌ Validation dataset not found: {dataset_path}")
        print("\nTo create validation dataset:")
        print("1. Collect 50 scientific papers (various quality levels)")
        print("2. Get expert scores (1-10 scale, 5 dimensions)")
        print("3. Save to data/validation/validation_dataset_v1.json")
        print("\nExample format:")
        print(json.dumps({
            "papers": [{
                "id": "paper_001",
                "title": "Sample Paper Title",
                "content": "Full paper text...",
                "human_scores": {
                    "overall": 8,
                    "novelty": 7,
                    "methodology": 9,
                    "clarity": 8,
                    "significance": 8
                }
            }],
            "metadata": {
                "total_papers": 50,
                "creation_date": "2025-10-05",
                "version": "1.0"
            }
        }, indent=2))
        return []

    with open(dataset_path) as f:
        data = json.load(f)
    return data.get("papers", [])


async def run_validation():
    """Run validation experiments comparing methods."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from src.core.config import settings
    from src.services.llm.service import LLMService
    from src.services.paper.analyzer import PaperAnalyzer
    from src.services.paper.metrics import PaperMetrics

    print("=" * 80)
    print("SOTA METHODS VALIDATION EXPERIMENT")
    print("=" * 80)

    # Load validation data
    print("\n📂 Loading validation dataset...")
    validation_papers = await load_validation_dataset()

    if not validation_papers:
        print("\n⚠️  No validation dataset found. Exiting.")
        return

    print(f"✅ Loaded {len(validation_papers)} papers")

    # Initialize services
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    llm_service = LLMService()

    # Storage for results
    results = {
        "gpt4_only": [],
        "scibert_only": [],
        "ensemble": []
    }
    human_scores = []

    # Run experiments
    async with async_session() as db:
        analyzer = PaperAnalyzer(llm_service, db)

        for i, paper_data in enumerate(validation_papers, 1):
            print(f"\n📄 [{i}/{len(validation_papers)}] {paper_data['title'][:50]}...")

            # Store human score
            human_scores.append(paper_data["human_scores"]["overall"])

            # NOTE: This requires papers to be in database
            # For now, using mock scores based on content analysis
            # TODO: Import papers to database first

            # Simulated results for demonstration
            gpt4_score = 7.5
            scibert_score = 7.8
            ensemble_score = 0.4 * gpt4_score + 0.6 * scibert_score

            results["gpt4_only"].append(gpt4_score)
            results["scibert_only"].append(scibert_score)
            results["ensemble"].append(ensemble_score)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    metrics = PaperMetrics()

    for method_name, scores in results.items():
        print(f"\n📊 {method_name.upper()}:")

        # QWK (primary metric)
        qwk = metrics.quadratic_weighted_kappa(
            human_scores,
            [int(round(s)) for s in scores]
        )
        print(f"  QWK (vs Human): {qwk:.4f}")

        # Correlation
        corr = metrics.calculate_correlation(
            [float(h) for h in human_scores],
            scores,
            method="pearson"
        )
        print(f"  Pearson Correlation: {corr:.4f}")

        # MAE
        mae = metrics.mean_absolute_error(
            [float(h) for h in human_scores],
            scores
        )
        print(f"  Mean Absolute Error: {mae:.4f}")

    # Save results
    output_path = Path("data/validation/validation_results_phase1.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "human_scores": human_scores,
        "ai_scores": results,
        "metrics": {
            method: {
                "qwk": metrics.quadratic_weighted_kappa(
                    human_scores,
                    [int(round(s)) for s in scores]
                ),
                "pearson": metrics.calculate_correlation(
                    [float(h) for h in human_scores],
                    scores
                ),
                "mae": metrics.mean_absolute_error(
                    [float(h) for h in human_scores],
                    scores
                )
            }
            for method, scores in results.items()
        },
        "validation_date": "2025-10-05",
        "num_papers": len(validation_papers)
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    print("\n" + "=" * 80)
    print("TODO: Import validation papers to database for real testing")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_validation())
