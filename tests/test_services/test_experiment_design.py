"""Unit tests for experiment design service."""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.experiment.design import ExperimentDesigner
from src.models.project import Hypothesis, HypothesisStatus


@pytest.fixture
def mock_db():
    """Mock database session."""
    db = AsyncMock()
    return db


@pytest.fixture
def mock_llm_service():
    """Mock LLM service."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="""{
        "title": "Test Experiment",
        "protocol": "Test protocol steps",
        "methods": ["Method 1", "Method 2"],
        "materials": ["Material 1"],
        "variables": {
            "independent": ["IV1"],
            "dependent": ["DV1"],
            "controlled": ["CV1"]
        },
        "data_collection": "Test data collection",
        "statistical_analysis": "Test analysis plan",
        "potential_confounds": ["Confound 1"],
        "mitigation_strategies": ["Strategy 1"],
        "estimated_duration": "4 weeks",
        "resource_requirements": {"participants": 100}
    }""")
    return llm


@pytest.fixture
def mock_kb_search():
    """Mock knowledge base search."""
    kb = AsyncMock()
    kb.semantic_search = AsyncMock(return_value=[])
    return kb


@pytest.fixture
def experiment_designer(mock_llm_service, mock_kb_search, mock_db):
    """Create experiment designer instance."""
    return ExperimentDesigner(
        llm_service=mock_llm_service,
        knowledge_base=mock_kb_search,
        db=mock_db
    )


class TestSampleSizeCalculation:
    """Tests for sample size calculation."""

    def test_calculate_sample_size_medium_effect(self, experiment_designer):
        """Test sample size calculation with medium effect size."""
        sample_size = experiment_designer._calculate_sample_size(
            effect_size=0.5,
            power=0.8,
            alpha=0.05
        )

        # For medium effect (d=0.5), power=0.8, alpha=0.05
        # Expected ~64 per group, with 10% buffer ~70
        assert 60 <= sample_size <= 80

    def test_calculate_sample_size_large_effect(self, experiment_designer):
        """Test sample size calculation with large effect size."""
        sample_size = experiment_designer._calculate_sample_size(
            effect_size=0.8,
            power=0.8,
            alpha=0.05
        )

        # Larger effect size requires fewer participants
        assert sample_size < 50

    def test_calculate_sample_size_small_effect(self, experiment_designer):
        """Test sample size calculation with small effect size."""
        sample_size = experiment_designer._calculate_sample_size(
            effect_size=0.2,
            power=0.8,
            alpha=0.05
        )

        # Smaller effect size requires more participants
        assert sample_size > 300


class TestPowerCalculation:
    """Tests for power calculation."""

    def test_calculate_power_adequate_sample(self, experiment_designer):
        """Test power calculation with adequate sample size."""
        power = experiment_designer.calculate_power(
            effect_size=0.5,
            sample_size=64,
            alpha=0.05
        )

        # Should achieve ~0.8 power
        assert 0.75 <= power <= 0.85

    def test_calculate_power_large_sample(self, experiment_designer):
        """Test power calculation with large sample size."""
        power = experiment_designer.calculate_power(
            effect_size=0.5,
            sample_size=200,
            alpha=0.05
        )

        # Larger sample should give higher power
        assert power > 0.95

    def test_calculate_power_small_sample(self, experiment_designer):
        """Test power calculation with small sample size."""
        power = experiment_designer.calculate_power(
            effect_size=0.5,
            sample_size=20,
            alpha=0.05
        )

        # Small sample should give low power
        assert power < 0.5


@pytest.mark.asyncio
class TestExperimentDesign:
    """Tests for experiment design functionality."""

    async def test_design_experiment_success(
        self,
        experiment_designer,
        mock_db
    ):
        """Test successful experiment design."""
        hypothesis_id = uuid4()

        # Mock hypothesis query
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = Hypothesis(
            id=hypothesis_id,
            project_id=uuid4(),
            content="Test hypothesis",
            novelty_score=0.8,
            status=HypothesisStatus.VALIDATED
        )
        mock_db.execute = AsyncMock(return_value=mock_result)

        experiment = await experiment_designer.design_experiment(
            hypothesis_id=hypothesis_id,
            research_question="Test question",
            hypothesis_content="Test hypothesis",
            desired_power=0.8,
            significance_level=0.05,
            expected_effect_size=0.5
        )

        assert experiment is not None
        assert experiment.hypothesis_id == hypothesis_id
        assert experiment.title == "Test Experiment"
        assert experiment.sample_size > 0
        assert experiment.power == 0.8
        assert experiment.effect_size == 0.5

    async def test_design_experiment_hypothesis_not_found(
        self,
        experiment_designer,
        mock_db
    ):
        """Test experiment design with non-existent hypothesis."""
        hypothesis_id = uuid4()

        # Mock hypothesis not found
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        with pytest.raises(ValueError, match="not found"):
            await experiment_designer.design_experiment(
                hypothesis_id=hypothesis_id,
                research_question="Test question",
                hypothesis_content="Test hypothesis"
            )

    async def test_design_experiment_with_constraints(
        self,
        experiment_designer,
        mock_db
    ):
        """Test experiment design with resource constraints."""
        hypothesis_id = uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = Hypothesis(
            id=hypothesis_id,
            project_id=uuid4(),
            content="Test hypothesis",
            novelty_score=0.8,
            status=HypothesisStatus.VALIDATED
        )
        mock_db.execute = AsyncMock(return_value=mock_result)

        constraints = {
            "max_participants": 50,
            "max_duration_days": 30,
            "budget": 10000
        }

        experiment = await experiment_designer.design_experiment(
            hypothesis_id=hypothesis_id,
            research_question="Test question",
            hypothesis_content="Test hypothesis",
            constraints=constraints
        )

        assert experiment is not None
        # LLM should consider constraints in protocol


class TestProtocolGeneration:
    """Tests for protocol generation."""

    def test_build_protocol_prompt(self, experiment_designer):
        """Test protocol prompt building."""
        prompt = experiment_designer._build_protocol_prompt(
            research_question="Test question",
            hypothesis="Test hypothesis",
            methodology_context="Test context",
            sample_size=64,
            constraints={"budget": 10000},
            approach="lab experiment"
        )

        assert "Test question" in prompt
        assert "Test hypothesis" in prompt
        assert "64" in prompt
        assert "budget" in prompt
        assert "lab experiment" in prompt

    def test_parse_protocol_response_valid_json(self, experiment_designer):
        """Test parsing valid JSON protocol response."""
        response = """{
            "title": "Test Experiment",
            "protocol": "Test protocol",
            "methods": ["Method 1"]
        }"""

        data = experiment_designer._parse_protocol_response(response)

        assert data["title"] == "Test Experiment"
        assert data["protocol"] == "Test protocol"
        assert "Method 1" in data["methods"]

    def test_parse_protocol_response_invalid_json(self, experiment_designer):
        """Test parsing invalid JSON falls back gracefully."""
        response = "This is not JSON"

        data = experiment_designer._parse_protocol_response(response)

        assert "protocol" in data
        assert data["protocol"] == response


@pytest.mark.asyncio
class TestMethodologySearch:
    """Tests for methodology search."""

    async def test_search_methodologies_with_results(
        self,
        experiment_designer,
        mock_kb_search
    ):
        """Test methodology search with results."""
        # Mock search results
        mock_result = MagicMock()
        mock_result.title = "Test Paper"
        mock_result.abstract = "Test abstract with methodology"
        mock_kb_search.semantic_search = AsyncMock(
            return_value=[mock_result]
        )

        context = await experiment_designer._search_methodologies(
            research_question="Test question",
            hypothesis="Test hypothesis"
        )

        assert "Test Paper" in context
        assert "Test abstract" in context

    async def test_search_methodologies_no_results(
        self,
        experiment_designer,
        mock_kb_search
    ):
        """Test methodology search with no results."""
        mock_kb_search.semantic_search = AsyncMock(return_value=[])

        context = await experiment_designer._search_methodologies(
            research_question="Test question",
            hypothesis="Test hypothesis"
        )

        assert "No specific methodology found" in context
