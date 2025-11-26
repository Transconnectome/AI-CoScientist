import pytest
from unittest.mock import AsyncMock
from src.services.review.reviewer_agent import ReviewerAgent, ReviewResult

class TestReviewerAgent:
    """Test adversarial reviewer that finds flaws."""

    @pytest.mark.asyncio
    async def test_review_paper_finds_fatal_flaws(self):
        """Test that reviewer detects fatal flaws in weak paper."""
        agent = ReviewerAgent()
        
        # Mock LLM service
        llm_service = AsyncMock()
        llm_service.get_response = AsyncMock(return_value="""{
            "recommendation": "reject",
            "fatal_flaws": [
                "Data leakage: Authors tested on training data",
                "No statistical power analysis"
            ],
            "major_issues": [
                "Missing baselines: No comparison to standard methods"
            ],
            "minor_issues": [
                "Figure 1 caption is unclear"
            ],
            "overall_assessment": "This paper has fundamental methodological flaws."
        }""")
        
        weak_paper = """
        We used LSTM to predict Bitcoin. We got 99% accuracy on our data.
        """
        
        result = await agent.review_paper(weak_paper, llm_service)
        
        # Verify structure
        assert isinstance(result, ReviewResult)
        assert result.recommendation == "reject"
        assert len(result.fatal_flaws) >= 1
        assert "data leakage" in result.fatal_flaws[0].lower() or "training data" in result.fatal_flaws[0].lower()

    @pytest.mark.asyncio
    async def test_review_paper_accepts_strong_paper(self):
        """Test that reviewer accepts well-designed paper."""
        agent = ReviewerAgent()
        
        # Mock LLM service for strong paper
        llm_service = AsyncMock()
        llm_service.get_response = AsyncMock(return_value="""{
            "recommendation": "accept",
            "fatal_flaws": [],
            "major_issues": [],
            "minor_issues": [
                "Minor typo in abstract"
            ],
            "overall_assessment": "Well-designed study with proper methodology."
        }""")
        
        strong_paper = """
        Novel fMRI architecture. Power analysis: G*Power, alpha=0.05, power=0.8.
        Code: github.com/example/repo. Effect size: Cohen's d=0.85, 95% CI [0.7, 1.0].
        """
        
        result = await agent.review_paper(strong_paper, llm_service)
        
        assert result.recommendation == "accept"
        assert len(result.fatal_flaws) == 0

    @pytest.mark.asyncio
    async def test_review_paper_major_revision(self):
        """Test that reviewer suggests major revision for fixable issues."""
        agent = ReviewerAgent()
        
        # Mock LLM service
        llm_service = AsyncMock()
        llm_service.get_response = AsyncMock(return_value="""{
            "recommendation": "major_revision",
            "fatal_flaws": [],
            "major_issues": [
                "Missing code repository",
                "Statistical analysis needs more detail"
            ],
            "minor_issues": [],
            "overall_assessment": "Promising work but needs improvements."
        }""")
        
        paper = "Good idea but missing implementation details."
        
        result = await agent.review_paper(paper, llm_service)
        
        assert result.recommendation == "major_revision"
        assert len(result.major_issues) >= 1
