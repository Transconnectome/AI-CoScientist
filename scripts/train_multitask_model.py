#!/usr/bin/env python3
"""Training script for multi-task paper quality scorer.

Requirements:
- Validation dataset with 5-dimensional scores: data/validation/validation_dataset_v1.json
- GPU recommended (2-4 hours training time on GPU)
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))


async def load_multitask_data() -> Tuple[
    List[Tuple[str, Dict[str, float]]],
    List[Tuple[str, Dict[str, float]]]
]:
    """Load and prepare data for multi-task training.

    Returns:
        (train_data, val_data) with 5-dimensional quality scores
    """
    # Try v2 dataset first (85 papers), fallback to v1 (63 papers)
    dataset_path = Path("data/validation/validation_dataset_v2.json")

    if not dataset_path.exists():
        dataset_path = Path("data/validation/validation_dataset_v1.json")

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("\n📋 Required format (5 dimensions per paper):")
        print(json.dumps({
            "papers": [{
                "id": "paper_001",
                "title": "Sample Title",
                "abstract": "Abstract...",
                "content": "Full text...",
                "human_scores": {
                    "overall": 8,
                    "novelty": 7,
                    "methodology": 9,
                    "clarity": 8,
                    "significance": 8
                }
            }]
        }, indent=2))
        return [], []

    with open(dataset_path) as f:
        data = json.load(f)

    papers = data.get("papers", [])

    if len(papers) < 10:
        print(f"⚠️  Insufficient data: {len(papers)} papers (minimum 10 recommended)")
        return [], []

    # Prepare examples
    examples = []
    for paper in papers:
        # Combine text
        text_parts = [
            paper.get("title", ""),
            paper.get("abstract", ""),
            paper.get("content", "")[:5000]
        ]
        text = "\n\n".join(part for part in text_parts if part)

        # Get all dimension scores
        human_scores = paper.get("human_scores", {})

        # Ensure all 5 dimensions present
        required_dims = ["overall", "novelty", "methodology", "clarity", "significance"]
        if all(dim in human_scores for dim in required_dims):
            scores_dict = {
                dim: float(human_scores[dim])
                for dim in required_dims
            }
            examples.append((text, scores_dict))

    # Split 80/20
    split_idx = int(len(examples) * 0.8)
    train_data = examples[:split_idx]
    val_data = examples[split_idx:]

    print(f"✅ Loaded {len(examples)} papers with 5-dimensional scores")
    print(f"   Training: {len(train_data)} papers")
    print(f"   Validation: {len(val_data)} papers")

    return train_data, val_data


async def train_model():
    """Train multi-task model on dataset."""
    print("=" * 80)
    print("MULTI-TASK MODEL TRAINING PIPELINE")
    print("=" * 80)
    print()

    # Load data
    print("📂 Loading multi-dimensional training data...")
    train_data, val_data = await load_multitask_data()

    if not train_data or not val_data:
        print("\n⚠️  Cannot proceed without training data.")
        print("\n📋 Next steps:")
        print("1. Ensure validation dataset has 5-dimensional scores per paper")
        print("   - overall, novelty, methodology, clarity, significance")
        print("2. Save to data/validation/validation_dataset_v1.json")
        return

    print()

    # Initialize model
    print("🔧 Initializing multi-task model...")
    from src.services.paper.multitask_scorer import MultiTaskPaperScorer, MultiTaskTrainer
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    model = MultiTaskPaperScorer(device=device)

    # Task weights (can adjust based on importance)
    task_weights = {
        "overall": 2.0,       # Higher weight for overall quality
        "novelty": 1.0,
        "methodology": 1.5,   # Important for scientific papers
        "clarity": 1.0,
        "significance": 1.5   # Important for impact assessment
    }

    trainer = MultiTaskTrainer(
        model,
        learning_rate=1e-4,
        task_weights=task_weights,
        device=device
    )

    print()

    # Train
    print("🚀 Starting multi-task training...")
    print()

    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=25,
        batch_size=4,
        checkpoint_dir="models/multitask"
    )

    # Save final model
    final_path = Path("models/multitask/final_model.pt")
    model.save_weights(str(final_path))
    print(f"\n✅ Final model saved to: {final_path}")

    # Save training history
    history_path = Path("models/multitask/training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"✅ Training history saved to: {history_path}")

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL MULTI-DIMENSIONAL EVALUATION")
    print("=" * 80)

    val_metrics = trainer.validate(val_data)
    print(f"\nOverall Validation Loss: {val_metrics['loss']:.4f}\n")

    dimension_names = ["overall", "novelty", "methodology", "clarity", "significance"]

    print("Per-Dimension Performance:")
    for dim_name in dimension_names:
        mae = val_metrics[f"{dim_name}_mae"]
        corr = val_metrics[f"{dim_name}_correlation"]
        print(f"  {dim_name.capitalize():15s}: MAE={mae:.4f}, Correlation={corr:.4f}")

    # Calculate QWK for overall quality
    try:
        from src.services.paper.metrics import PaperMetrics

        predictions = []
        targets = []

        model.eval()
        with torch.no_grad():
            for text, target_scores in val_data:
                result = await model.score_paper(text)
                predictions.append(int(round(result["overall_quality"])))
                targets.append(int(target_scores["overall"]))

        metrics = PaperMetrics()
        qwk = metrics.quadratic_weighted_kappa(targets, predictions)
        accuracy = metrics.calculate_accuracy(targets, predictions, tolerance=1)

        print(f"\nOverall Quality Assessment:")
        print(f"  QWK: {qwk:.4f}")
        print(f"  Accuracy (±1): {accuracy:.4f}")

        # Phase 3 success criteria
        print("\n📊 Phase 3 Success Criteria:")
        if qwk >= 0.90:
            print(f"   ✅ QWK ≥ 0.90: {qwk:.4f}")
        else:
            print(f"   ⚠️  QWK < 0.90: {qwk:.4f} (target: 0.90)")

    except Exception as e:
        print(f"⚠️  Could not calculate QWK: {e}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


async def test_inference():
    """Test trained multi-task model inference."""
    print("\n" + "=" * 80)
    print("TESTING MULTI-TASK INFERENCE")
    print("=" * 80)

    from src.services.paper.multitask_scorer import MultiTaskPaperScorer
    import torch

    # Load model
    model_path = Path("models/multitask/best_model.pt")

    if not model_path.exists():
        print(f"❌ No trained model found at: {model_path}")
        print("Run training first!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskPaperScorer(device=device)
    model.load_weights(str(model_path))

    print(f"✅ Loaded model from: {model_path}")
    print()

    # Sample text
    sample_text = """
    Transformer Networks for Medical Image Segmentation

    Abstract:
    We propose a novel transformer-based architecture for medical image segmentation.
    Our method achieves state-of-the-art performance on multiple benchmark datasets,
    demonstrating the effectiveness of self-attention mechanisms for capturing long-range
    dependencies in medical images. Extensive experiments validate our approach.

    Introduction:
    Medical image segmentation is critical for clinical diagnosis and treatment planning.
    Traditional CNN-based methods struggle with long-range spatial relationships.
    We introduce a pure transformer architecture that addresses these limitations
    while maintaining computational efficiency.
    """

    print("📝 Sample paper text:")
    print(sample_text[:200] + "...\n")

    # Score
    result = await model.score_paper(sample_text)

    print("📊 Multi-Dimensional Quality Assessment:")
    print(f"   Overall Quality:  {result['overall_quality']:.2f} / 10")
    print(f"   Novelty:          {result['novelty_quality']:.2f} / 10")
    print(f"   Methodology:      {result['methodology_quality']:.2f} / 10")
    print(f"   Clarity:          {result['clarity_quality']:.2f} / 10")
    print(f"   Significance:     {result['significance_quality']:.2f} / 10")
    print(f"\n   Model Type: {result['model_type']}")
    print(f"   Trained: {result['trained']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train multi-task paper quality scorer")
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Test inference with trained model (skip training)"
    )
    args = parser.parse_args()

    if args.test_only:
        asyncio.run(test_inference())
    else:
        asyncio.run(train_model())
        asyncio.run(test_inference())
