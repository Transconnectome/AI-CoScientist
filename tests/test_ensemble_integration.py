import pytest
from src.services.paper.ensemble_scorer import EnsemblePaperScorer

class TestEnsembleAdvancedMetrics:
    """Test integration of advanced metrics into ensemble scorer."""

    @pytest.mark.asyncio
    async def test_ensemble_includes_advanced_metrics(self):
        """Test that ensemble scorer includes reproducibility and narrative scores."""
        ensemble = EnsemblePaperScorer(
            gpt4_weight=0.0,  # Disable GPT-4 for faster testing
            hybrid_weight=0.5,
            multitask_weight=0.5,
            use_gpt4=False
        )
        
        # Sample paper with reproducibility indicators
        paper_text = """
        Deep Learning for Natural Language Processing
        
        Abstract: This novel approach achieves breakthrough results.
        We provide code at https://github.com/example/repo.
        Statistical power analysis was conducted with G*Power.
        Effect size (Cohen's d = 0.8) demonstrates strong impact.
        
        Introduction:
        NLP has advanced rapidly. However, reproducibility remains a challenge.
        """
        
        # Score paper
        result = await ensemble.score_paper(paper_text, return_individual=True)
        
        # Verify advanced metrics are included
        assert "advanced_metrics" in result
        assert "reproducibility" in result["advanced_metrics"]
        assert "narrative" in result["advanced_metrics"]
        
        # Verify scores are reasonable
        assert result["advanced_metrics"]["reproducibility"] >= 7.0  # High due to code + stats
        assert result["advanced_metrics"]["narrative"]["hook_score"] >= 5.0

    @pytest.mark.asyncio
    async def test_ensemble_weights_advanced_metrics(self):
        """Test that advanced metrics influence the overall score."""
        ensemble = EnsemblePaperScorer(
            use_gpt4=False,
            include_advanced_metrics=True,
            advanced_metrics_weight=0.15  # 15% weight
        )
        
        # Paper with excellent reproducibility
        excellent_paper = """
        Research Paper with Full Reproducibility
        Code: https://github.com/example/repo
        Data: https://zenodo.org/record/12345
        Power analysis: G*Power, alpha=0.05, power=0.8
        Effect size: Cohen's d = 0.85, 95% CI [0.7, 1.0]
        """
        
        result = await ensemble.score_paper(excellent_paper)
        
        # Should have high overall score boosted by reproducibility
        assert result["overall"] >= 6.0
