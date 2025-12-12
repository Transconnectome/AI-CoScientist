import pytest
from unittest.mock import Mock, AsyncMock
from src.services.paper.generator import SectionGenerator, GenerationRequest

@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.generate = AsyncMock()
    return llm

@pytest.fixture
def mock_rag():
    rag = Mock()
    rag.search = AsyncMock()
    return rag

@pytest.fixture
def generator(mock_llm, mock_rag):
    return SectionGenerator(llm=mock_llm, rag_client=mock_rag)

@pytest.mark.asyncio
async def test_generate_introduction(generator, mock_llm, mock_rag):
    """Test introduction generation with RAG context."""
    request = GenerationRequest(
        section_type="introduction",
        topic="Neural Networks",
        key_points=["Deep learning is powerful", "Transformers are new"],
        target_journal="Nature"
    )
    
    # Mock RAG response
    mock_rag.search.return_value = ["Reference 1: Deep learning...", "Reference 2: Transformers..."]
    
    # Mock LLM response
    mock_llm.generate.return_value = (
        "## Introduction\n\nDeep learning has revolutionized...",
        "mock_provider"
    )
    
    section = await generator.generate_section(request)
    
    assert "Introduction" in section
    mock_rag.search.assert_called_once()
    mock_llm.generate.assert_called_once()

@pytest.mark.asyncio
async def test_generate_methods_structure(generator, mock_llm):
    """Test methods section follows structured format."""
    request = GenerationRequest(
        section_type="methods",
        topic="Experiment 1",
        key_points=["Data collection", "Analysis"],
        target_journal="Nature"
    )
    
    mock_llm.generate.return_value = (
        "## Methods\n\n### Data Collection\nWe collected...",
        "mock_provider"
    )
    
    section = await generator.generate_section(request)
    
    assert "Methods" in section
    assert "Data Collection" in section

@pytest.mark.asyncio
async def test_style_transfer_prompt(generator, mock_llm):
    """Test that style instructions are included in the prompt."""
    request = GenerationRequest(
        section_type="results",
        topic="Accuracy",
        key_points=["95% accuracy"],
        style_guide="Use passive voice and concise sentences."
    )
    
    mock_llm.generate.return_value = ("Results...", "mock")
    
    await generator.generate_section(request)
    
    # Check if style guide was used in prompt
    call_args = mock_llm.generate.call_args[0][0]
    assert "passive voice" in call_args
