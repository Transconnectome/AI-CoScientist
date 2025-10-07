#!/usr/bin/env python3
"""Training script for hybrid paper quality scorer.

Requirements:
- Validation dataset: data/validation/validation_dataset_v1.json
- GPU recommended for faster training (falls back to CPU)
- Approximately 2-3 hours training time on GPU
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


async def load_training_data() -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Load and split validation dataset into train/validation sets.

    Returns:
        (train_data, val_data) where each is list of (text, score) tuples
    """
    # Try v2 dataset first (85 papers), fallback to v1 (63 papers)
    dataset_path = Path("data/validation/validation_dataset_v2.json")

    if not dataset_path.exists():
        dataset_path = Path("data/validation/validation_dataset_v1.json")

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("\n📋 Required dataset format:")
        print(json.dumps({
            "papers": [
                {
                    "id": "paper_001",
                    "title": "Sample Paper Title",
                    "abstract": "Paper abstract...",
                    "content": "Full paper text...",
                    "human_scores": {
                        "overall": 8,
                        "novelty": 7,
                        "methodology": 9,
                        "clarity": 8,
                        "significance": 8
                    }
                }
            ],
            "metadata": {
                "total_papers": 50,
                "creation_date": "2025-10-05",
                "version": "1.0"
            }
        }, indent=2))
        return [], []

    with open(dataset_path) as f:
        data = json.load(f)

    papers = data.get("papers", [])

    if len(papers) < 10:
        print(f"⚠️  Insufficient data: {len(papers)} papers (minimum 10 recommended)")
        return [], []

    # Prepare training examples
    examples = []
    for paper in papers:
        # Combine title, abstract, and content
        text_parts = [
            paper.get("title", ""),
            paper.get("abstract", ""),
            paper.get("content", "")[:5000]  # Limit content to 5000 chars for training
        ]
        text = "\n\n".join(part for part in text_parts if part)

        # Get overall quality score
        human_scores = paper.get("human_scores", {})
        overall_score = human_scores.get("overall", 0)

        if text and overall_score > 0:
            examples.append((text, float(overall_score)))

    # Split: 80% train, 20% validation
    split_idx = int(len(examples) * 0.8)
    train_data = examples[:split_idx]
    val_data = examples[split_idx:]

    print(f"✅ Loaded {len(examples)} papers")
    print(f"   Training: {len(train_data)} papers")
    print(f"   Validation: {len(val_data)} papers")

    return train_data, val_data


async def train_model():
    """Train hybrid model on validation dataset."""
    print("=" * 80)
    print("HYBRID MODEL TRAINING PIPELINE")
    print("=" * 80)
    print()

    # Load data
    print("📂 Loading training data...")
    train_data, val_data = await load_training_data()

    if not train_data or not val_data:
        print("\n⚠️  Cannot proceed without training data.")
        print("\n📋 Next steps:")
        print("1. Collect 50 scientific papers (various quality levels)")
        print("2. Get expert quality scores (1-10 scale)")
        print("3. Save to data/validation/validation_dataset_v1.json")
        return

    print()

    # Initialize model
    print("🔧 Initializing hybrid model...")
    from src.services.paper.hybrid_scorer import HybridPaperScorer, HybridTrainer
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    model = HybridPaperScorer(device=device)
    trainer = HybridTrainer(model, learning_rate=1e-4, device=device)

    print()

    # Train
    print("🚀 Starting training...")
    print()

    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=20,
        batch_size=4,
        checkpoint_dir="models/hybrid"
    )

    # Save final model
    final_path = Path("models/hybrid/final_model.pt")
    model.save_weights(str(final_path))
    print(f"\n✅ Final model saved to: {final_path}")

    # Save training history
    history_path = Path("models/hybrid/training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"✅ Training history saved to: {history_path}")

    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    val_metrics = trainer.validate(val_data)
    print(f"Validation Loss: {val_metrics['loss']:.4f}")
    print(f"Validation MAE: {val_metrics['mae']:.4f}")
    print(f"Validation Correlation: {val_metrics['correlation']:.4f}")

    # Calculate QWK if sklearn available
    try:
        from src.services.paper.metrics import PaperMetrics

        predictions = []
        targets = []

        model.eval()
        with torch.no_grad():
            for text, target_score in val_data:
                result = await model.score_paper(text)
                predictions.append(int(round(result["overall_quality"])))
                targets.append(int(target_score))

        metrics = PaperMetrics()
        qwk = metrics.quadratic_weighted_kappa(targets, predictions)
        accuracy = metrics.calculate_accuracy(targets, predictions, tolerance=1)

        print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
        print(f"Accuracy (±1 tolerance): {accuracy:.4f}")

        # Check Phase 2 success criteria
        print("\n📊 Phase 2 Success Criteria:")
        if qwk >= 0.85:
            print(f"   ✅ QWK ≥ 0.85: {qwk:.4f}")
        else:
            print(f"   ⚠️  QWK < 0.85: {qwk:.4f} (target: 0.85)")

    except Exception as e:
        print(f"⚠️  Could not calculate QWK: {e}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


async def test_inference():
    """Test trained model inference on sample text."""
    print("\n" + "=" * 80)
    print("TESTING INFERENCE")
    print("=" * 80)

    from src.services.paper.hybrid_scorer import HybridPaperScorer
    import torch

    # Load trained model
    model_path = Path("models/hybrid/best_model.pt")

    if not model_path.exists():
        print(f"❌ No trained model found at: {model_path}")
        print("Run training first!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridPaperScorer(device=device)
    model.load_weights(str(model_path))

    print(f"✅ Loaded model from: {model_path}")
    print()

    # Sample paper text
    sample_text = """
    Deep Learning for Natural Language Processing: A Survey

    Abstract:
    This paper provides a comprehensive survey of deep learning methods for natural language processing.
    We review recent advances in neural architectures, including transformers, attention mechanisms,
    and pre-trained language models. Our analysis covers both theoretical foundations and practical
    applications across various NLP tasks.

    Introduction:
    Natural language processing has undergone a paradigm shift with the advent of deep learning.
    Traditional feature-based methods have been largely superseded by end-to-end neural approaches
    that learn representations directly from data. This survey examines the key developments that
    have driven this transformation and their implications for future research.
    """

    print("📝 Sample paper text:")
    print(sample_text[:200] + "...\n")

    # Score
    result = await model.score_paper(sample_text)

    print("📊 Quality Assessment:")
    print(f"   Overall Quality: {result['overall_quality']:.2f} / 10")
    print(f"   Model Type: {result['model_type']}")
    print(f"   Trained: {result['trained']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train hybrid paper quality scorer")
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
