import pytest
from unittest.mock import AsyncMock, MagicMock
from src.services.paper.narrative_analyzer import NarrativeAnalyzer

class TestNarrativeAnalyzer:
    """Test LLM-based narrative analysis."""

    @pytest.mark.asyncio
    async def test_analyze_narrative_returns_correct_structure(self):
        """Test that narrative analyzer returns expected structure."""
        analyzer = NarrativeAnalyzer()
        
        # Mock LLM service
        llm_service = AsyncMock()
        llm_service.get_response = AsyncMock(return_value="""{
            "hook_score": 8.5,
            "tension_curve": [0.2, 0.7, 0.9, 0.6],
            "story_elements": {
                "hook": "Novel transformer architecture for fMRI",
                "gap": "Reproducibility remains a challenge",
                "resolution": "We provide open-source implementation"
            },
            "feedback": "Strong hook with clear gap definition. Resolution is well-stated."
        }""")
        
        abstract = """
        This paper introduces a breakthrough transformer-based architecture achieving 
        state-of-the-art performance in brain decoding tasks.
        """
        
        introduction = """
        fMRI analysis has advanced significantly. However, reproducibility remains 
        a critical challenge in the field. We address this by...
        """
        
        result = await analyzer.analyze_narrative(abstract, introduction, llm_service)
        
        # Verify structure
        assert "hook_score" in result
        assert "tension_curve" in result
        assert "story_elements" in result
        assert "feedback" in result
        
        # Verify values
        assert result["hook_score"] == 8.5
        assert len(result["tension_curve"]) == 4
        assert "gap" in result["story_elements"]

    @pytest.mark.asyncio
    async def test_analyze_narrative_handles_weak_paper(self):
        """Test that analyzer detects weak narrative."""
        analyzer = NarrativeAnalyzer()
        
        # Mock LLM service for weak paper
        llm_service = AsyncMock()
        llm_service.get_response = AsyncMock(return_value="""{
            "hook_score": 3.0,
            "tension_curve": [0.1, 0.2, 0.15, 0.1],
            "story_elements": {
                "hook": "We used machine learning",
                "gap": "Not clearly defined",
                "resolution": "Missing"
            },
            "feedback": "Weak hook. No clear problem statement or tension."
        }""")
        
        abstract = "We used machine learning to predict things."
        introduction = "Machine learning is popular."
        
        result = await analyzer.analyze_narrative(abstract, introduction, llm_service)
        
        assert result["hook_score"] < 5.0
        assert max(result["tension_curve"]) < 0.5

    @pytest.mark.asyncio
    async def test_analyze_narrative_with_real_llm(self):
        """Test with actual LLM service (integration test)."""
        # This test requires actual API keys
        pytest.skip("Integration test - requires API keys")
        
        from src.services.llm.service import LLMService
        
        analyzer = NarrativeAnalyzer()
        llm_service = LLMService()
        
        abstract = """
        Novel deep learning architecture for fMRI analysis achieving breakthrough results.
        Our method demonstrates 30% improvement over previous approaches.
        """
        
        introduction = """
        fMRI analysis has advanced significantly. However, reproducibility remains 
        a critical challenge. We address this fundamental gap...
        """
        
        result = await analyzer.analyze_narrative(abstract, introduction, llm_service)
        
        # Should return valid structure
        assert 0 <= result["hook_score"] <= 10
        assert len(result["tension_curve"]) > 0
