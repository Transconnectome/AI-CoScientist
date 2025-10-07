"""Ensemble paper quality scorer combining multiple models.

Combines GPT-4, Hybrid, and Multi-task models for robust quality assessment:
- GPT-4 (40% weight): Qualitative analysis with reasoning
- Hybrid (30% weight): Fast RoBERTa + linguistic features
- Multi-task (30% weight): Multi-dimensional quality scores

Ensemble benefits:
- Robustness: Multiple models reduce single-point failures
- Confidence scoring: Agreement indicates reliability
- Multi-dimensional feedback: Rich quality assessment
- Cost-effective: Reduces GPT-4 API costs with local models
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch


class EnsemblePaperScorer:
    """Ensemble scorer combining GPT-4, Hybrid, and Multi-task models."""

    def __init__(
        self,
        gpt4_weight: float = 0.4,
        hybrid_weight: float = 0.3,
        multitask_weight: float = 0.3,
        use_gpt4: bool = True,
        device: Optional[str] = None
    ):
        """Initialize ensemble scorer.

        Args:
            gpt4_weight: Weight for GPT-4 scores (default 0.4)
            hybrid_weight: Weight for Hybrid model (default 0.3)
            multitask_weight: Weight for Multi-task model (default 0.3)
            use_gpt4: Whether to include GPT-4 (requires API key)
            device: Device for PyTorch models (cuda/cpu)
        """
        # Normalize weights
        total_weight = gpt4_weight + hybrid_weight + multitask_weight
        self.gpt4_weight = gpt4_weight / total_weight
        self.hybrid_weight = hybrid_weight / total_weight
        self.multitask_weight = multitask_weight / total_weight

        self.use_gpt4 = use_gpt4

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Initialize models (lazy loading)
        self._hybrid_model = None
        self._multitask_model = None
        self._gpt4_client = None

        print(f"🎯 Ensemble Scorer initialized")
        print(f"   Weights: GPT-4={self.gpt4_weight:.2f}, "
              f"Hybrid={self.hybrid_weight:.2f}, "
              f"Multi-task={self.multitask_weight:.2f}")
        print(f"   Device: {self.device}")

    def _load_hybrid_model(self):
        """Lazy load hybrid model."""
        if self._hybrid_model is not None:
            return

        try:
            from src.services.paper.hybrid_scorer import HybridPaperScorer
        except ModuleNotFoundError:
            # Try relative import
            from hybrid_scorer import HybridPaperScorer

        model_path = Path("models/hybrid/best_model.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Hybrid model not found: {model_path}")

        self._hybrid_model = HybridPaperScorer(device=self.device)
        self._hybrid_model.load_weights(str(model_path))
        print(f"✅ Loaded Hybrid model from: {model_path}")

    def _load_multitask_model(self):
        """Lazy load multi-task model."""
        if self._multitask_model is not None:
            return

        try:
            from src.services.paper.multitask_scorer import MultiTaskPaperScorer
        except ModuleNotFoundError:
            # Try relative import
            from multitask_scorer import MultiTaskPaperScorer

        model_path = Path("models/multitask/best_model.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Multi-task model not found: {model_path}")

        self._multitask_model = MultiTaskPaperScorer(device=self.device)
        self._multitask_model.load_weights(str(model_path))
        print(f"✅ Loaded Multi-task model from: {model_path}")

    def _load_gpt4_client(self):
        """Lazy load GPT-4 client."""
        if self._gpt4_client is not None:
            return

        if not self.use_gpt4:
            return

        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OPENAI_API_KEY not found, disabling GPT-4")
            self.use_gpt4 = False
            return

        self._gpt4_client = AsyncOpenAI(api_key=api_key)
        print("✅ Initialized GPT-4 client")

    async def _score_with_gpt4(self, paper_text: str) -> Dict:
        """Score paper with GPT-4.

        Args:
            paper_text: Paper title + abstract + content

        Returns:
            Dict with overall score and analysis
        """
        self._load_gpt4_client()

        if not self.use_gpt4 or self._gpt4_client is None:
            return None

        # Extract title and abstract for prompt
        lines = paper_text.split('\n')
        title = lines[0] if lines else "Unknown Title"
        abstract_start = next((i for i, line in enumerate(lines) if len(line) > 100), 1)
        abstract = lines[abstract_start] if abstract_start < len(lines) else ""

        prompt = f"""Rate this scientific paper's overall quality on a 1-10 scale.

Title: {title}

Abstract: {abstract[:500]}

Content Preview:
{paper_text[:2000]}

Provide:
1. Overall quality score (1-10)
2. Brief 2-3 sentence analysis

Return JSON:
{{
    "overall": <score>,
    "analysis": "<brief analysis>"
}}
"""

        try:
            response = await self._gpt4_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert peer reviewer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )

            content = response.choices[0].message.content.strip()

            # Strip markdown fences
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            import json
            result = json.loads(content)

            return {
                "overall": float(result.get("overall", 5.0)),
                "analysis": result.get("analysis", "")
            }

        except Exception as e:
            print(f"⚠️  GPT-4 scoring failed: {e}")
            return None

    async def score_paper(
        self,
        paper_text: str,
        return_individual: bool = False
    ) -> Dict:
        """Score paper using ensemble of models.

        Args:
            paper_text: Paper title + abstract + content
            return_individual: Whether to return individual model scores

        Returns:
            Dict with ensemble score and metadata
        """
        # Load models
        self._load_hybrid_model()
        self._load_multitask_model()

        # Run all models in parallel
        tasks = []

        # GPT-4
        if self.use_gpt4:
            tasks.append(self._score_with_gpt4(paper_text))

        # Hybrid model
        tasks.append(self._hybrid_model.score_paper(paper_text))

        # Multi-task model
        tasks.append(self._multitask_model.score_paper(paper_text))

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Extract scores
        gpt4_result = results[0] if self.use_gpt4 else None
        hybrid_idx = 1 if self.use_gpt4 else 0
        multitask_idx = 2 if self.use_gpt4 else 1

        hybrid_result = results[hybrid_idx] if not isinstance(results[hybrid_idx], Exception) else None
        multitask_result = results[multitask_idx] if not isinstance(results[multitask_idx], Exception) else None

        # Collect scores
        scores = []
        weights = []

        if gpt4_result and not isinstance(gpt4_result, Exception):
            scores.append(gpt4_result["overall"])
            weights.append(self.gpt4_weight)
        else:
            gpt4_result = None

        if hybrid_result:
            scores.append(hybrid_result["overall_quality"])
            weights.append(self.hybrid_weight)

        if multitask_result:
            scores.append(multitask_result["overall_quality"])
            weights.append(self.multitask_weight)

        # Normalize weights if some models failed
        if sum(weights) > 0:
            weights = [w / sum(weights) for w in weights]

        # Calculate weighted average
        if scores:
            ensemble_score = sum(s * w for s, w in zip(scores, weights))
        else:
            ensemble_score = 5.0  # Default if all models failed

        # Calculate confidence (lower std dev = higher confidence)
        if len(scores) >= 2:
            std_dev = np.std(scores)
            confidence = max(0.0, 1.0 - (std_dev / 5.0))  # Normalize by max possible std
        else:
            confidence = 0.5  # Low confidence with single model

        # Prepare result
        result = {
            "overall": float(ensemble_score),
            "confidence": float(confidence),
            "model_type": "ensemble",
            "num_models": len(scores)
        }

        # Add multi-dimensional scores from multi-task model
        if multitask_result:
            result["dimensions"] = {
                "novelty": multitask_result["novelty_quality"],
                "methodology": multitask_result["methodology_quality"],
                "clarity": multitask_result["clarity_quality"],
                "significance": multitask_result["significance_quality"]
            }

        # Add individual model scores if requested
        if return_individual:
            result["individual_scores"] = {
                "gpt4": gpt4_result["overall"] if gpt4_result else None,
                "hybrid": hybrid_result["overall_quality"] if hybrid_result else None,
                "multitask": multitask_result["overall_quality"] if multitask_result else None
            }

            if gpt4_result:
                result["gpt4_analysis"] = gpt4_result["analysis"]

        # Add agreement analysis
        if len(scores) >= 2:
            max_diff = max(scores) - min(scores)
            result["agreement"] = {
                "max_difference": float(max_diff),
                "std_deviation": float(std_dev),
                "interpretation": self._interpret_agreement(max_diff)
            }

        return result

    def _interpret_agreement(self, max_diff: float) -> str:
        """Interpret model agreement level.

        Args:
            max_diff: Maximum difference between model scores

        Returns:
            Human-readable interpretation
        """
        if max_diff < 1.0:
            return "Strong agreement - all models aligned"
        elif max_diff < 2.0:
            return "Good agreement - minor differences"
        elif max_diff < 3.0:
            return "Moderate agreement - some uncertainty"
        else:
            return "Low agreement - significant uncertainty, recommend human review"


async def test_ensemble():
    """Test ensemble scorer."""
    print("=" * 80)
    print("ENSEMBLE SCORER TEST")
    print("=" * 80)
    print()

    # Initialize ensemble
    ensemble = EnsemblePaperScorer(
        gpt4_weight=0.4,
        hybrid_weight=0.3,
        multitask_weight=0.3,
        use_gpt4=True  # Set to False if no API key
    )

    # Sample paper
    sample_text = """
    Deep Learning for Natural Language Processing: A Comprehensive Survey

    Abstract:
    This paper provides a comprehensive survey of deep learning methods for natural language processing.
    We review recent advances in neural architectures, including transformers, attention mechanisms,
    and pre-trained language models. Our analysis covers both theoretical foundations and practical
    applications across various NLP tasks. We identify key challenges and future research directions
    in the field.

    Introduction:
    Natural language processing has undergone a paradigm shift with the advent of deep learning.
    Traditional feature-based methods have been largely superseded by end-to-end neural approaches
    that learn representations directly from data. This survey examines the key developments that
    have driven this transformation and their implications for future research.

    The rise of transformer architectures, particularly models like BERT and GPT, has revolutionized
    the field. These models leverage self-attention mechanisms to capture long-range dependencies
    and contextual information effectively. Pre-training on large corpora followed by fine-tuning
    on specific tasks has become the dominant paradigm.
    """

    print("📝 Sample Paper:")
    print(sample_text[:200] + "...\n")

    # Score
    result = await ensemble.score_paper(sample_text, return_individual=True)

    print("📊 ENSEMBLE RESULTS")
    print("=" * 80)
    print(f"\n🎯 Overall Quality: {result['overall']:.2f} / 10")
    print(f"🎲 Confidence:      {result['confidence']:.2f}")
    print(f"🤝 Models Used:     {result['num_models']}")

    if "dimensions" in result:
        print("\n📐 Quality Dimensions:")
        for dim, score in result["dimensions"].items():
            print(f"   {dim.capitalize():15s}: {score:.2f}")

    if "individual_scores" in result:
        print("\n🔍 Individual Model Scores:")
        for model, score in result["individual_scores"].items():
            if score is not None:
                print(f"   {model.upper():12s}: {score:.2f}")

    if "agreement" in result:
        print(f"\n📊 Model Agreement:")
        print(f"   Max Difference: {result['agreement']['max_difference']:.2f}")
        print(f"   Interpretation: {result['agreement']['interpretation']}")

    if "gpt4_analysis" in result:
        print(f"\n💬 GPT-4 Analysis:")
        print(f"   {result['gpt4_analysis']}")

    print("\n" + "=" * 80)
    print("✅ Ensemble test complete!")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    asyncio.run(test_ensemble())
