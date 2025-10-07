#!/usr/bin/env python3
"""Optimize trained models for production deployment.

Usage:
    python scripts/optimize_models.py --model hybrid
    python scripts/optimize_models.py --model multitask --techniques quantization onnx
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def optimize_models(
    model_type: str = "all",
    techniques: list = None
):
    """Optimize models for deployment.

    Args:
        model_type: Model to optimize ("hybrid", "multitask", "all")
        techniques: List of optimization techniques to apply
    """
    from src.services.paper.model_optimization import ModelOptimizer

    if techniques is None:
        techniques = ["quantization", "pruning", "onnx"]

    print("=" * 80)
    print("MODEL OPTIMIZATION FOR PRODUCTION DEPLOYMENT")
    print("=" * 80)
    print(f"Model type: {model_type}")
    print(f"Techniques: {', '.join(techniques)}")
    print()

    models_to_optimize = []

    # Hybrid model
    if model_type in ["hybrid", "all"]:
        hybrid_path = Path("models/hybrid/best_model.pt")
        if hybrid_path.exists():
            models_to_optimize.append({
                "name": "Hybrid Model",
                "path": str(hybrid_path),
                "class_import": "src.services.paper.hybrid_scorer",
                "class_name": "HybridPaperScorer",
                "output_dir": "models/optimized/hybrid"
            })
        else:
            print(f"⚠️  Hybrid model not found: {hybrid_path}")

    # Multi-task model
    if model_type in ["multitask", "all"]:
        multitask_path = Path("models/multitask/best_model.pt")
        if multitask_path.exists():
            models_to_optimize.append({
                "name": "Multi-Task Model",
                "path": str(multitask_path),
                "class_import": "src.services.paper.multitask_scorer",
                "class_name": "MultiTaskPaperScorer",
                "output_dir": "models/optimized/multitask"
            })
        else:
            print(f"⚠️  Multi-task model not found: {multitask_path}")

    if not models_to_optimize:
        print("❌ No trained models found to optimize.")
        print("\nTrain models first:")
        print("  python scripts/train_hybrid_model.py")
        print("  python scripts/train_multitask_model.py")
        return

    # Optimize each model
    for model_info in models_to_optimize:
        print("\n" + "=" * 80)
        print(f"Optimizing {model_info['name']}")
        print("=" * 80)
        print()

        # Dynamically import model class
        import importlib
        module = importlib.import_module(model_info["class_import"])
        model_class = getattr(module, model_info["class_name"])

        # Optimize
        output_paths = ModelOptimizer.optimize_for_deployment(
            model_path=model_info["path"],
            model_class=model_class,
            output_dir=model_info["output_dir"],
            techniques=techniques
        )

        print(f"\n✅ {model_info['name']} optimization complete")
        print(f"   Output directory: {model_info['output_dir']}")

    # Summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    print()
    print("Optimized models ready for deployment:")
    for model_info in models_to_optimize:
        output_dir = Path(model_info["output_dir"])
        if output_dir.exists():
            print(f"\n{model_info['name']}:")
            for file in output_dir.glob("*"):
                print(f"  - {file.name}")

    print("\n" + "=" * 80)
    print("DEPLOYMENT GUIDE")
    print("=" * 80)
    print()
    print("1. Quantized Models (*.pt):")
    print("   - Use on CPU for best performance")
    print("   - 4x smaller, 2-4x faster")
    print("   - Load with OptimizedModelLoader.load_quantized_model()")
    print()
    print("2. Pruned Models (pruned_model.pt):")
    print("   - May need fine-tuning")
    print("   - 10-30% faster inference")
    print()
    print("3. ONNX Models (*.onnx):")
    print("   - Cross-platform deployment")
    print("   - Use with ONNX Runtime for 2-3x speedup")
    print("   - Install: pip install onnxruntime")
    print("   - Load with OptimizedModelLoader.load_onnx_model()")
    print()
    print("Example deployment code:")
    print("""
from src.services.paper.model_optimization import OptimizedModelLoader

# Load quantized model (fastest on CPU)
model = OptimizedModelLoader.load_quantized_model(
    "models/optimized/hybrid/quantized_model.pt",
    HybridPaperScorer,
    device="cpu"
)

# Or load ONNX model (cross-platform)
session = OptimizedModelLoader.load_onnx_model(
    "models/optimized/hybrid/model.onnx"
)
    """)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize models for production deployment"
    )
    parser.add_argument(
        "--model",
        choices=["hybrid", "multitask", "all"],
        default="all",
        help="Model to optimize (default: all)"
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        choices=["quantization", "pruning", "onnx"],
        help="Optimization techniques to apply (default: all)"
    )

    args = parser.parse_args()

    optimize_models(
        model_type=args.model,
        techniques=args.techniques
    )
