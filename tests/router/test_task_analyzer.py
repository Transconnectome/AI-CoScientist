import pytest
from src.router.analyzer import TaskAnalyzer
from src.router.types import ResearchTask, TaskProfile

@pytest.fixture
def mock_llm_service():
    return None  # Mock for now

@pytest.fixture
def task_analyzer(mock_llm_service):
    return TaskAnalyzer(mock_llm_service)

@pytest.mark.asyncio
async def test_analyze_simple_task(task_analyzer):
    """Analyze a simple research task"""
    task = ResearchTask(
        description="Search for fMRI papers on emotion recognition",
        task_type="literature_search"
    )

    profile = await task_analyzer.analyze_task(task)

    assert profile is not None
    assert "neuroscience" in profile.domains
    assert profile.complexity in ["simple", "medium", "high"]
    assert len(profile.required_expertise) > 0

@pytest.mark.asyncio
async def test_analyze_complex_task(task_analyzer):
    """Analyze complex multi-domain task"""
    task = ResearchTask(
        description="""Generate novel hypothesis for fMRI emotion recognition
                       using deep learning with ethical considerations""",
        task_type="hypothesis_generation"
    )

    profile = await task_analyzer.analyze_task(task)

    assert profile.complexity == "high"
    assert len(profile.domains) >= 3  # neuroscience, ML, ethics
    assert "neuroscience" in profile.domains
    assert "machine_learning" in profile.domains
