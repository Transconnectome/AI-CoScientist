#!/usr/bin/env python3
"""Analyze model predictions to understand QWK collapse.

This script loads trained models, evaluates on validation set,
and generates detailed analysis including:
- Confusion matrix
- Score distribution
- Prediction range analysis
- QWK calculation verification
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


async def load_validation_data() -> List[Tuple[str, float]]:
    """Load validation dataset.

    Returns:
        List of (text, score) tuples
    """
    dataset_path = Path("data/validation/validation_dataset_v2.json")

    with open(dataset_path) as f:
        data = json.load(f)

    papers = data.get("papers", [])

    # Prepare examples
    examples = []
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
            examples.append((text, float(overall_score)))

    # Split: 80% train, 20% validation
    split_idx = int(len(examples) * 0.8)
    val_data = examples[split_idx:]

    return val_data


async def analyze_hybrid_model():
    """Analyze hybrid model predictions."""
    print("=" * 80)
    print("HYBRID MODEL PREDICTION ANALYSIS")
    print("=" * 80)
    print()

    from src.services.paper.hybrid_scorer import HybridPaperScorer
    import torch

    # Load model
    model_path = Path("models/hybrid/best_model.pt")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridPaperScorer(device=device)
    model.load_weights(str(model_path))

    print(f"✅ Loaded model from: {model_path}")
    print()

    # Load validation data
    val_data = await load_validation_data()
    print(f"📊 Validation samples: {len(val_data)}")
    print()

    # Get predictions
    predictions = []
    targets = []

    model.eval()
    with torch.no_grad():
        for text, target_score in val_data:
            result = await model.score_paper(text)
            pred_score = result["overall_quality"]

            predictions.append(pred_score)
            targets.append(target_score)

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Analysis 1: Score Distribution
    print("📊 SCORE DISTRIBUTION")
    print("-" * 80)
    print("\nTarget (Ground Truth):")
    for score in range(1, 11):
        count = np.sum(targets == score)
        if count > 0:
            bar = "█" * int(count)
            print(f"  {score:2d}: {bar} ({count})")

    print("\nPredicted (Model Output):")
    pred_rounded = np.round(predictions).astype(int)
    for score in range(1, 11):
        count = np.sum(pred_rounded == score)
        if count > 0:
            bar = "█" * int(count)
            print(f"  {score:2d}: {bar} ({count})")

    # Analysis 2: Prediction Statistics
    print("\n📈 PREDICTION STATISTICS")
    print("-" * 80)
    print(f"Target Mean:      {np.mean(targets):.2f}")
    print(f"Target Std:       {np.std(targets):.2f}")
    print(f"Target Range:     {np.min(targets):.0f} - {np.max(targets):.0f}")
    print()
    print(f"Prediction Mean:  {np.mean(predictions):.2f}")
    print(f"Prediction Std:   {np.std(predictions):.2f}")
    print(f"Prediction Range: {np.min(predictions):.2f} - {np.max(predictions):.2f}")
    print()

    # Analysis 3: Confusion Matrix
    print("🔍 CONFUSION MATRIX (Rounded Predictions)")
    print("-" * 80)

    # Create confusion matrix
    unique_targets = sorted(set(targets.astype(int)))
    unique_preds = sorted(set(pred_rounded))

    print("\n      Predicted →")
    print("Actual  ", end="")
    for pred in range(1, 11):
        if pred in unique_preds or pred in unique_targets:
            print(f"{pred:3d}", end=" ")
    print()
    print("↓")

    for target in unique_targets:
        print(f"  {target:2d}   ", end="")
        for pred in range(1, 11):
            if pred in unique_preds or pred in unique_targets:
                count = np.sum((targets == target) & (pred_rounded == pred))
                if count > 0:
                    print(f"{count:3d}", end=" ")
                else:
                    print("  .", end=" ")
        print()

    # Analysis 4: Error Analysis
    print("\n⚠️  ERROR ANALYSIS")
    print("-" * 80)

    errors = predictions - targets
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print()

    # Count errors by magnitude
    within_0_5 = np.sum(np.abs(errors) <= 0.5)
    within_1_0 = np.sum(np.abs(errors) <= 1.0)
    within_1_5 = np.sum(np.abs(errors) <= 1.5)
    within_2_0 = np.sum(np.abs(errors) <= 2.0)

    total = len(errors)
    print(f"Within ±0.5: {within_0_5:2d} / {total} ({within_0_5/total*100:.1f}%)")
    print(f"Within ±1.0: {within_1_0:2d} / {total} ({within_1_0/total*100:.1f}%)")
    print(f"Within ±1.5: {within_1_5:2d} / {total} ({within_1_5/total*100:.1f}%)")
    print(f"Within ±2.0: {within_2_0:2d} / {total} ({within_2_0/total*100:.1f}%)")

    # Analysis 5: QWK Calculation
    print("\n🎯 QWK CALCULATION")
    print("-" * 80)

    try:
        from src.services.paper.metrics import PaperMetrics

        metrics = PaperMetrics()
        targets_int = targets.astype(int)
        preds_int = pred_rounded.astype(int)

        qwk = metrics.quadratic_weighted_kappa(targets_int.tolist(), preds_int.tolist())
        accuracy = metrics.calculate_accuracy(targets_int.tolist(), preds_int.tolist(), tolerance=1)

        print(f"QWK:          {qwk:.4f}")
        print(f"Accuracy ±1:  {accuracy:.4f}")

        # Manual QWK verification
        print("\nManual QWK Verification:")
        print(f"  Unique targets: {unique_targets}")
        print(f"  Unique predictions: {list(unique_preds)}")
        print(f"  Prediction variance: {np.var(pred_rounded):.4f}")

        if np.var(pred_rounded) == 0:
            print("  ⚠️  WARNING: Zero prediction variance - all predictions identical!")
            print(f"     All predictions = {pred_rounded[0]}")

    except Exception as e:
        print(f"❌ QWK calculation failed: {e}")

    # Analysis 6: Sample Predictions
    print("\n📋 SAMPLE PREDICTIONS")
    print("-" * 80)
    print("\nTarget | Prediction | Error  | Within ±1?")
    print("-------|------------|--------|------------")
    for i in range(min(10, len(targets))):
        target = targets[i]
        pred = predictions[i]
        error = pred - target
        within = "✅" if abs(error) <= 1.0 else "❌"
        print(f"  {target:.1f}  |    {pred:.2f}    | {error:+.2f}  | {within}")

    print("\n" + "=" * 80)
    print()


async def analyze_multitask_model():
    """Analyze multi-task model predictions."""
    print("=" * 80)
    print("MULTI-TASK MODEL PREDICTION ANALYSIS")
    print("=" * 80)
    print()

    from src.services.paper.multitask_scorer import MultiTaskPaperScorer
    import torch

    # Load model
    model_path = Path("models/multitask/best_model.pt")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskPaperScorer(device=device)
    model.load_weights(str(model_path))

    print(f"✅ Loaded model from: {model_path}")
    print()

    # Load validation data
    val_data = await load_validation_data()
    print(f"📊 Validation samples: {len(val_data)}")
    print()

    # Get predictions
    predictions = []
    targets = []

    model.eval()
    with torch.no_grad():
        for text, target_score in val_data:
            result = await model.score_paper(text)
            pred_score = result["overall_quality"]

            predictions.append(pred_score)
            targets.append(target_score)

    predictions = np.array(predictions)
    targets = np.array(targets)
    pred_rounded = np.round(predictions).astype(int)

    # Similar analysis as hybrid
    print("📊 PREDICTION STATISTICS")
    print("-" * 80)
    print(f"Prediction Mean:  {np.mean(predictions):.2f}")
    print(f"Prediction Std:   {np.std(predictions):.2f}")
    print(f"Prediction Range: {np.min(predictions):.2f} - {np.max(predictions):.2f}")
    print()

    # Error Analysis
    errors = predictions - targets
    mae = np.mean(np.abs(errors))

    within_1_0 = np.sum(np.abs(errors) <= 1.0)
    total = len(errors)

    print(f"MAE:  {mae:.4f}")
    print(f"Within ±1.0: {within_1_0:2d} / {total} ({within_1_0/total*100:.1f}%)")
    print()

    # QWK
    try:
        from src.services.paper.metrics import PaperMetrics

        metrics = PaperMetrics()
        targets_int = targets.astype(int)
        preds_int = pred_rounded.astype(int)

        qwk = metrics.quadratic_weighted_kappa(targets_int.tolist(), preds_int.tolist())

        print(f"QWK: {qwk:.4f}")

        if np.var(pred_rounded) == 0:
            print("⚠️  WARNING: All predictions identical!")

    except Exception as e:
        print(f"❌ QWK calculation failed: {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MODEL PREDICTION ANALYSIS")
    print("=" * 80)
    print("\nAnalyzing why QWK collapsed despite perfect ±1 accuracy...")
    print()

    asyncio.run(analyze_hybrid_model())
    asyncio.run(analyze_multitask_model())

    print("\n✅ Analysis complete!")
    print("\n📋 Next steps:")
    print("1. Review confusion matrix for prediction patterns")
    print("2. Check if predictions are concentrated in narrow range")
    print("3. Implement ordinal regression loss if needed")
    print("4. Consider stratified sampling for validation set")
