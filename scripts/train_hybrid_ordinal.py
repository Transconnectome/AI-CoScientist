#!/usr/bin/env python3
"""Training script for hybrid model with ordinal regression.

Uses:
- HybridPaperScorerOrdinal (dual heads: score + ordinal)
- HybridOrdinalLoss (MSE 30% + Ordinal 70%)
- Stratified dataset split for balanced validation
- Stronger regularization (dropout 0.4)

Expected improvements:
- QWK: 0.000 → 0.15-0.30+ (immediate)
- With more training: 0.30-0.50+
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


async def load_stratified_data() -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Load stratified train/validation split.

    Returns:
        (train_data, val_data) tuples
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
        print("❌ No split indices found in dataset")
        return [], []

    # Prepare examples
    all_examples = []
    for paper in papers:
        text_parts = [
            paper.get("title", ""),
            paper.get("abstract", ""),
            paper.get("content", "")[:5000]
        ]
        text = "\n\n".join(part for part in text_parts if part)

        human_scores = paper.get("human_scores", {})
        overall_score = human_scores.get("overall", 0)

        if text and overall_score > 0:
            all_examples.append((text, float(overall_score)))

    # Split using indices
    train_data = [all_examples[i] for i in train_indices if i < len(all_examples)]
    val_data = [all_examples[i] for i in val_indices if i < len(all_examples)]

    print(f"✅ Loaded stratified dataset")
    print(f"   Training: {len(train_data)} papers")
    print(f"   Validation: {len(val_data)} papers")

    # Show validation score distribution
    val_scores = [score for _, score in val_data]
    print(f"\n📊 Validation Score Distribution:")
    for score in sorted(set(val_scores)):
        count = val_scores.count(score)
        bar = "█" * count
        print(f"   Score {int(score)}: {bar} ({count})")

    return train_data, val_data


async def train_model():
    """Train ordinal hybrid model."""
    print("=" * 80)
    print("HYBRID ORDINAL MODEL TRAINING PIPELINE")
    print("=" * 80)
    print()

    # Load data
    print("📂 Loading stratified training data...")
    train_data, val_data = await load_stratified_data()

    if not train_data or not val_data:
        print("\n⚠️  Cannot proceed without training data.")
        return

    print()

    # Initialize ordinal model
    print("🔧 Initializing hybrid ordinal model...")
    from src.services.paper.hybrid_scorer_ordinal import HybridPaperScorerOrdinal, HybridTrainerOrdinal
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    model = HybridPaperScorerOrdinal(
        dropout=0.4,  # Stronger regularization
        device=device
    )

    trainer = HybridTrainerOrdinal(
        model,
        learning_rate=5e-5,  # Lower LR for stability
        mse_weight=0.3,
        ordinal_weight=0.7,
        device=device
    )

    print(f"   Dropout: 0.4 (increased regularization)")
    print(f"   Learning rate: 5e-5 (reduced for stability)")
    print(f"   Loss: HybridOrdinalLoss (MSE 30% + Ordinal 70%)")
    print()

    # Train
    print("🚀 Starting training...")
    print()

    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=30,  # More epochs for ordinal learning
        batch_size=4,
        checkpoint_dir="models/hybrid_ordinal"
    )

    # Save final model
    final_path = Path("models/hybrid_ordinal/final_model.pt")
    model.save_weights(str(final_path))
    print(f"\n✅ Final model saved to: {final_path}")

    # Save training history
    history_path = Path("models/hybrid_ordinal/training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"✅ Training history saved to: {history_path}")

    # Evaluate
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    val_metrics = trainer.validate(val_data)
    print(f"Validation Loss: {val_metrics['loss']:.4f}")
    print(f"Validation MAE: {val_metrics['mae']:.4f}")
    print(f"Validation Correlation: {val_metrics['correlation']:.4f}")

    # Calculate QWK
    try:
        from src.services.paper.metrics import PaperMetrics

        predictions = []
        targets = []

        model.eval()
        with torch.no_grad():
            for text, target_score in val_data:
                score_output, ordinal_logits = model.forward(text)
                pred_score = model.predict_from_ordinal(ordinal_logits)[0].item()

                predictions.append(int(round(pred_score)))
                targets.append(int(target_score))

        metrics = PaperMetrics()
        qwk = metrics.quadratic_weighted_kappa(targets, predictions)
        accuracy = metrics.calculate_accuracy(targets, predictions, tolerance=1)

        print(f"\nQuadratic Weighted Kappa (QWK): {qwk:.4f}")
        print(f"Accuracy (±1 tolerance): {accuracy:.4f}")

        # Compare with original
        print(f"\n📊 Improvement vs Original:")
        print(f"   Original QWK: 0.000 (homogeneous validation set)")
        print(f"   Ordinal QWK:  {qwk:.4f}")

        if qwk > 0.15:
            print(f"   ✅ SUCCESS: QWK > 0.15 achieved!")
        elif qwk > 0.0:
            print(f"   ⚠️  PARTIAL: QWK > 0 but below 0.15")
        else:
            print(f"   ❌ No improvement: Check model and data")

    except Exception as e:
        print(f"⚠️  Could not calculate QWK: {e}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(train_model())
