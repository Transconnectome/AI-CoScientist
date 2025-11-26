import pytest
from unittest.mock import Mock, AsyncMock
from src.services.paper.style_extractor import StyleExtractor, StyleMetrics

@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.generate = AsyncMock()
    return llm

@pytest.fixture
def style_extractor(mock_llm):
    return StyleExtractor(llm=mock_llm)

@pytest.mark.asyncio
async def test_analyze_text_structure(style_extractor):
    """Test analysis of text structure (sentence length, paragraph length)."""
    text = """
    This is a short sentence. This is a slightly longer sentence that has more words.
    
    This is a new paragraph. It contains some scientific terminology like "neural networks" and "optimization".
    """
    
    metrics = await style_extractor.analyze_style(text)
    
    assert isinstance(metrics, StyleMetrics)
    assert metrics.avg_sentence_length > 0
    assert metrics.avg_paragraph_length > 0
    assert metrics.vocabulary_richness > 0

@pytest.mark.asyncio
async def test_extract_transition_phrases(style_extractor, mock_llm):
    """Test extraction of transition phrases using LLM."""
    text = "However, the results were inconclusive. Therefore, we propose a new method."
    
    # Mock LLM response
    mock_llm.generate.return_value = (
        '["However", "Therefore"]',
        "mock_provider"
    )
    
    transitions = await style_extractor.extract_transitions(text)
    
    assert "However" in transitions
    assert "Therefore" in transitions
    mock_llm.generate.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_tone_and_voice(style_extractor, mock_llm):
    """Test analysis of tone and voice."""
    text = "We demonstrate that the proposed method significantly outperforms baselines."
    
    mock_llm.generate.return_value = (
        '{"tone": "objective", "voice": "active", "confidence": "high"}',
        "mock_provider"
    )
    
    tone_analysis = await style_extractor.analyze_tone(text)
    
    assert tone_analysis["tone"] == "objective"
    assert tone_analysis["voice"] == "active"
