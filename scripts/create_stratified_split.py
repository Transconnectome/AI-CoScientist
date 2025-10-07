#!/usr/bin/env python3
"""Create stratified train/validation split for balanced quality score distribution.

The current 80/20 random split results in validation sets with only 2 quality levels (7, 8),
which causes QWK calculation to fail (zero variance).

This script creates a stratified split ensuring:
1. Each quality level (2-8) represented in validation set
2. Minimum 2 samples per quality level in validation
3. Proportional distribution maintained
"""

import json
from pathlib import Path
from collections import defaultdict
import random


def create_stratified_split(
    dataset_path: Path,
    output_path: Path,
    val_ratio: float = 0.2,
    min_val_per_score: int = 2,
    random_seed: int = 42
):
    """Create stratified split of dataset.

    Args:
        dataset_path: Path to input dataset JSON
        output_path: Path to output stratified dataset JSON
        val_ratio: Validation set ratio (default 0.2 for 20%)
        min_val_per_score: Minimum samples per score in validation
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)

    print("=" * 80)
    print("CREATING STRATIFIED TRAIN/VALIDATION SPLIT")
    print("=" * 80)
    print()

    # Load dataset
    with open(dataset_path) as f:
        data = json.load(f)

    papers = data.get("papers", [])
    print(f"📊 Total papers: {len(papers)}")
    print()

    # Group papers by overall quality score
    papers_by_score = defaultdict(list)

    for paper in papers:
        score = paper.get("human_scores", {}).get("overall", 0)
        if score > 0:
            papers_by_score[score].append(paper)

    # Show current distribution
    print("📈 Score Distribution:")
    print("-" * 80)
    for score in sorted(papers_by_score.keys()):
        count = len(papers_by_score[score])
        bar = "█" * count
        print(f"  Score {int(score)}: {bar} ({count} papers)")
    print()

    # Perform stratified split
    train_papers = []
    val_papers = []

    print("🔀 Performing Stratified Split:")
    print("-" * 80)
    print(f"Target validation ratio: {val_ratio*100:.0f}%")
    print(f"Minimum validation samples per score: {min_val_per_score}")
    print()

    for score in sorted(papers_by_score.keys()):
        score_papers = papers_by_score[score].copy()
        random.shuffle(score_papers)

        total_for_score = len(score_papers)

        # Calculate validation size for this score
        # Ensure at least min_val_per_score if available
        target_val_size = max(
            min_val_per_score,
            int(total_for_score * val_ratio)
        )

        # Don't exceed available papers
        val_size = min(target_val_size, total_for_score)

        # Split
        val_for_score = score_papers[:val_size]
        train_for_score = score_papers[val_size:]

        val_papers.extend(val_for_score)
        train_papers.extend(train_for_score)

        print(f"  Score {int(score)}: {len(train_for_score)} train, {len(val_for_score)} val")

    print()
    print(f"✅ Total: {len(train_papers)} train, {len(val_papers)} validation")
    print()

    # Verify balance
    print("📊 Validation Set Score Distribution:")
    print("-" * 80)
    val_scores = defaultdict(int)
    for paper in val_papers:
        score = paper.get("human_scores", {}).get("overall", 0)
        val_scores[score] += 1

    for score in sorted(val_scores.keys()):
        count = val_scores[score]
        bar = "█" * count
        print(f"  Score {int(score)}: {bar} ({count} papers)")

    print()

    # Create output dataset with split information
    output_data = {
        "papers": papers,  # Keep all papers
        "metadata": {
            **data.get("metadata", {}),
            "split_method": "stratified",
            "split_ratio": val_ratio,
            "random_seed": random_seed,
            "min_val_per_score": min_val_per_score
        },
        "train_indices": [papers.index(p) for p in train_papers],
        "val_indices": [papers.index(p) for p in val_papers]
    }

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved stratified dataset to: {output_path}")
    print()

    # Calculate expected QWK improvement
    unique_val_scores = len(val_scores)
    print("📈 Expected Impact:")
    print("-" * 80)
    print(f"  Validation set diversity: {unique_val_scores} quality levels")
    print(f"  Previous: 2 quality levels (7, 8) → QWK = 0.000")
    print(f"  Current:  {unique_val_scores} quality levels → QWK calculable")
    print()
    print("💡 With diverse validation set, QWK should improve significantly!")
    print("   Expected: 0.15-0.30 initially, can reach 0.50+ with ordinal loss")
    print()

    return train_papers, val_papers


if __name__ == "__main__":
    dataset_path = Path("data/validation/validation_dataset_v2.json")
    output_path = Path("data/validation/validation_dataset_v2_stratified.json")

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("\nPlease run:")
        print("  python scripts/add_open_access_papers.py")
        exit(1)

    train_papers, val_papers = create_stratified_split(
        dataset_path=dataset_path,
        output_path=output_path,
        val_ratio=0.2,
        min_val_per_score=2,
        random_seed=42
    )

    print("=" * 80)
    print("STRATIFIED SPLIT COMPLETE")
    print("=" * 80)
    print()
    print("📋 Next steps:")
    print("1. Update training scripts to use stratified dataset")
    print("2. Retrain models with ordinal regression loss")
    print("3. Verify QWK > 0 with balanced validation set")
    print()
