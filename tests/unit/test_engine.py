import pytest
from unittest.mock import Mock, AsyncMock
from uuid import uuid4
from src.core.engine import CoScientistEngine
from src.models.project import Project, ProjectStatus

@pytest.fixture
def mock_components():
    return {
        "hypothesis_generator": Mock(),
        "experiment_designer": Mock(),
        "paper_generator": Mock(),
        "db": AsyncMock()
    }

@pytest.fixture
def engine(mock_components):
    return CoScientistEngine(
        hypothesis_generator=mock_components["hypothesis_generator"],
        experiment_designer=mock_components["experiment_designer"],
        paper_generator=mock_components["paper_generator"],
        db=mock_components["db"]
    )

@pytest.mark.asyncio
async def test_start_project(engine, mock_components):
    """Test starting a new research project."""
    topic = "Test Topic"
    mock_components["db"].add = Mock()
    mock_components["db"].commit = AsyncMock()
    mock_components["db"].refresh = AsyncMock()
    
    project = await engine.start_project(topic)
    
    assert project.research_question == topic
    assert project.status == ProjectStatus.ACTIVE
    mock_components["db"].add.assert_called_once()

@pytest.mark.asyncio
async def test_run_research_phase(engine, mock_components):
    """Test execution of the research phase."""
    project_id = uuid4()
    # Use a Mock object instead of real Project to avoid SQLAlchemy instrumentation
    mock_project = Mock(spec=Project)
    mock_project.id = project_id
    mock_project.research_question = "Test Question"
    mock_project.status = ProjectStatus.ACTIVE
    
    # Mock DB execution result
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = mock_project
    mock_components["db"].execute = AsyncMock(return_value=mock_result)
    mock_components["db"].commit = AsyncMock()
    
    # Mock Hypothesis Generator
    mock_components["hypothesis_generator"].generate_hypotheses = AsyncMock(return_value=["Hypothesis 1"])
    
    await engine.run_research_phase(project_id)
    
    mock_components["hypothesis_generator"].generate_hypotheses.assert_called_once()
    assert mock_project.status == ProjectStatus.ACTIVE

@pytest.mark.asyncio
async def test_run_experiment_phase(engine, mock_components):
    """Test execution of the experiment phase."""
    project_id = uuid4()
    mock_project = Mock(spec=Project)
    mock_project.id = project_id
    mock_project.research_question = "Test Question"
    mock_project.status = ProjectStatus.ACTIVE
    
    # Mock hypotheses
    hyp1 = Mock(id=uuid4(), content="Hypothesis 1")
    mock_project.hypotheses = [hyp1]
    
    # Mock DB execution result
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = mock_project
    mock_components["db"].execute = AsyncMock(return_value=mock_result)
    mock_components["db"].commit = AsyncMock()
    
    # Mock Experiment Designer
    mock_components["experiment_designer"].design_experiment = AsyncMock(return_value=Mock(id=uuid4()))
    
    await engine.run_experiment_phase(project_id)
    
    mock_components["experiment_designer"].design_experiment.assert_called_once()
    assert mock_project.status == ProjectStatus.ACTIVE

@pytest.mark.asyncio
async def test_run_paper_phase(engine, mock_components):
    """Test execution of the paper generation phase."""
    project_id = uuid4()
    mock_project = Mock(spec=Project)
    mock_project.id = project_id
    mock_project.status = ProjectStatus.ACTIVE
    
    # Mock DB execution result
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = mock_project
    mock_components["db"].execute = AsyncMock(return_value=mock_result)
    mock_components["db"].commit = AsyncMock()
    
    # Mock Paper Generator
    mock_components["paper_generator"].generate_from_project = AsyncMock(return_value="Paper Content")
    
    await engine.run_paper_phase(project_id)
    
    mock_components["paper_generator"].generate_from_project.assert_called_once()
    assert mock_project.status == ProjectStatus.ACTIVE

@pytest.mark.asyncio
async def test_run_discovery_loop(engine, mock_components):
    """Test the full discovery loop."""
    topic = "Full Loop Test"
    project_id = uuid4()
    mock_project = Mock(spec=Project)
    mock_project.id = project_id
    mock_project.research_question = topic
    mock_project.status = ProjectStatus.ACTIVE
    mock_project.hypotheses = [Mock(id=uuid4(), content="Hypothesis 1")]
    
    # Mock DB interactions
    mock_components["db"].add = Mock()
    mock_components["db"].commit = AsyncMock()
    mock_components["db"].refresh = AsyncMock()
    
    # Mock _get_project (via db.execute)
    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = mock_project
    mock_components["db"].execute = AsyncMock(return_value=mock_result)
    
    # Mock engine methods (we can't easily mock internal methods, so we mock the components they call)
    mock_components["hypothesis_generator"].generate_hypotheses = AsyncMock(return_value=["Hypothesis 1"])
    mock_components["experiment_designer"].design_experiment = AsyncMock(return_value=Mock(id=uuid4()))
    mock_components["paper_generator"].generate_from_project = AsyncMock(return_value="Paper Content")
    
    # We need to mock start_project to return our mock_project
    # But start_project is an instance method. We can mock the db.add to verify it was called.
    # However, start_project creates a NEW project instance.
    # To make this test easier, we can mock the internal calls if we want, but testing the flow is better.
    # The issue is start_project returns a new object, but subsequent calls use _get_project which returns mock_project.
    # This mismatch might be confusing but should work if we don't assert identity between start_project result and later results.
    
    # Actually, let's just run it and verify the component calls.
    
    await engine.run_discovery_loop(topic)
    
    mock_components["db"].add.assert_called() # From start_project
    mock_components["hypothesis_generator"].generate_hypotheses.assert_called_once()
    mock_components["experiment_designer"].design_experiment.assert_called()
    mock_components["paper_generator"].generate_from_project.assert_called_once()
