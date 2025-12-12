import pytest
from src.quality.validator import QualityValidationLoop
from src.quality.types import QualityScores, ValidationResult


@pytest.fixture
def validator():
    # Mock critics for now
    return QualityValidationLoop(quality_critics=[], threshold_config=None)


@pytest.mark.asyncio
async def test_validates_high_quality_output(validator):
    """High quality output passes without iterations"""
    from src.router.execution import ExecutionResult
    from src.agents.types import AgentResult

    result = ExecutionResult(
        status="success",
        agent_results=[
            AgentResult("agent1", "t1", "output", confidence=0.9)
        ],
        quality_score=0.9,
        execution_time_ms=100
    )

    validation = await validator.validate_and_refine(result, {})

    assert validation.status == "APPROVED"
    assert validation.iterations == 1


@pytest.mark.asyncio
async def test_refines_low_quality_output(validator):
    """Low quality triggers refinement iterations"""
    from src.router.execution import ExecutionResult
    from src.agents.types import AgentResult

    result = ExecutionResult(
        status="partial_success",
        agent_results=[
            AgentResult("agent1", "t1", "weak output", confidence=0.5)
        ],
        quality_score=0.5,
        execution_time_ms=100
    )

    validation = await validator.validate_and_refine(result, {})

    # Should attempt improvement
    assert validation.iterations > 1 or validation.status == "APPROVED_WITH_CONDITIONS"
