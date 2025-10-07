#!/usr/bin/env python3
"""Training script for multi-task model with ordinal regression.

Uses:
- MultiTaskPaperScorerOrdinal (ordinal heads for all 5 dimensions)
- HybridOrdinalLoss per dimension
- Stratified dataset split
- Task-specific weights

Expected improvements:
- QWK: 0.000 → 0.20-0.40+ (multi-dimensional ordinal learning)
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))


async def load_stratified_multitask_data() -> Tuple[
    List[Tuple[str, Dict[str, float]]],
    List[Tuple[str, Dict[str, float]]]
]:
    """Load stratified data for multi-task training.

    Returns:
        (train_data, val_data) with 5-dimensional scores
    """
    dataset_path = Path("data/validation/validation_dataset_v2_stratified.json")

    if not dataset_path.exists():
        print(f"❌ Stratified dataset not found: {dataset_path}")
        print("\nPlease run:")
        print("  python scripts/create_stratified_split.py")
        return [], []

    with open(dataset_path) as f:
        data = json.load(f)

    papers = data.get("papers", [])
    train_indices = data.get("train_indices", [])
    val_indices = data.get("val_indices", [])

    if not train_indices or not val_indices:
        print("❌ No split indices found")
        return [], []

    # Prepare examples with all dimensions
    all_examples = []
    for paper in papers:
        text_parts = [
            paper.get("title", ""),
            paper.get("abstract", ""),
            paper.get("content", "")[:5000]
        ]
        text = "\n\n".join(part for part in text_parts if part)

        human_scores = paper.get("human_scores", {})

        # Ensure all 5 dimensions present
        required_dims = ["overall", "novelty", "methodology", "clarity", "significance"]
        if all(dim in human_scores for dim in required_dims):
            scores_dict = {
                dim: float(human_scores[dim])
                for dim in required_dims
            }
            all_examples.append((text, scores_dict))

    # Split using indices
    train_data = [all_examples[i] for i in train_indices if i < len(all_examples)]
    val_data = [all_examples[i] for i in val_indices if i < len(all_examples)]

    print(f"✅ Loaded stratified multi-task dataset")
    print(f"   Training: {len(train_data)} papers")
    print(f"   Validation: {len(val_data)} papers")

    # Show validation distribution
    val_overall_scores = [scores["overall"] for _, scores in val_data]
    print(f"\n📊 Validation Overall Score Distribution:")
    for score in sorted(set(val_overall_scores)):
        count = val_overall_scores.count(score)
        bar = "█" * count
        print(f"   Score {int(score)}: {bar} ({count})")

    return train_data, val_data


async def train_model():
    """Train ordinal multi-task model."""
    print("=" * 80)
    print("MULTI-TASK ORDINAL MODEL TRAINING PIPELINE")
    print("=" * 80)
    print()

    # Load data
    print("📂 Loading stratified multi-dimensional training data...")
    train_data, val_data = await load_stratified_multitask_data()

    if not train_data or not val_data:
        print("\n⚠️  Cannot proceed without training data.")
        return

    print()

    # Initialize ordinal multi-task model
    print("🔧 Initializing multi-task ordinal model...")
    from src.services.paper.multitask_scorer_ordinal import MultiTaskPaperScorerOrdinal, MultiTaskTrainerOrdinal
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    model = MultiTaskPaperScorerOrdinal(
        dropout=0.4,  # Stronger regularization
        device=device
    )

    # Task weights (emphasize overall, methodology, significance)
    task_weights = {
        "overall": 2.0,
        "novelty": 1.0,
        "methodology": 1.5,
        "clarity": 1.0,
        "significance": 1.5
    }

    trainer = MultiTaskTrainerOrdinal(
        model,
        learning_rate=5e-5,
        task_weights=task_weights,
        mse_weight=0.3,
        ordinal_weight=0.7,
        device=device
    )

    print(f"   Dropout: 0.4 (increased regularization)")
    print(f"   Learning rate: 5e-5")
    print(f"   Task weights: {task_weights}")
    print(f"   Loss: HybridOrdinalLoss per dimension")
    print()

    # Train
    print("🚀 Starting multi-task ordinal training...")
    print()

    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=30,
        batch_size=4,
        checkpoint_dir="models/multitask_ordinal"
    )

    # Save final model
    final_path = Path("models/multitask_ordinal/final_model.pt")
    model.save_weights(str(final_path))
    print(f"\n✅ Final model saved to: {final_path}")

    # Save training history
    history_path = Path("models/multitask_ordinal/training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"✅ Training history saved to: {history_path}")

    # Evaluate
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
                outputs = model.forward(text)
                score_output, ordinal_logits = outputs["overall"]

                pred_score = model.predict_from_ordinal(ordinal_logits)[0].item()

                predictions.append(int(round(pred_score)))
                targets.append(int(target_scores["overall"]))

        metrics = PaperMetrics()
        qwk = metrics.quadratic_weighted_kappa(targets, predictions)
        accuracy = metrics.calculate_accuracy(targets, predictions, tolerance=1)

        print(f"\nOverall Quality Assessment:")
        print(f"  QWK: {qwk:.4f}")
        print(f"  Accuracy (±1): {accuracy:.4f}")

        # Compare with original
        print(f"\n📊 Improvement vs Original:")
        print(f"   Original QWK: 0.000 (homogeneous validation)")
        print(f"   Ordinal QWK:  {qwk:.4f}")

        if qwk > 0.20:
            print(f"   ✅ EXCELLENT: QWK > 0.20!")
        elif qwk > 0.10:
            print(f"   ✅ GOOD: QWK > 0.10")
        elif qwk > 0.0:
            print(f"   ⚠️  PARTIAL: QWK > 0 but below 0.10")
        else:
            print(f"   ❌ No improvement")

    except Exception as e:
        print(f"⚠️  Could not calculate QWK: {e}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(train_model())
