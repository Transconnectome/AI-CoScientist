import pytest
from unittest.mock import Mock, AsyncMock
from src.services.review.loop import AdversarialReviewLoop, ReviewResult, RevisionProposal

@pytest.fixture
def mock_reviewer():
    reviewer = Mock()
    reviewer.review = AsyncMock()
    return reviewer

@pytest.fixture
def mock_defense():
    defense = Mock()
    defense.propose_revision = AsyncMock()
    return defense

@pytest.fixture
def review_loop(mock_reviewer, mock_defense):
    return AdversarialReviewLoop(reviewer=mock_reviewer, defense=mock_defense)

@pytest.mark.asyncio
async def test_review_improvement_loop(review_loop, mock_reviewer, mock_defense):
    """Test that the loop iterates and improves the paper."""
    initial_draft = "Draft v1"
    
    # Iteration 1: Poor score
    review_1 = ReviewResult(score=5.0, comments="Too vague", passed=False)
    revision_1 = RevisionProposal(revised_text="Draft v2", explanation="Added details")
    
    # Iteration 2: Good score
    review_2 = ReviewResult(score=9.0, comments="Excellent", passed=True)
    
    # Mock sequence of returns
    mock_reviewer.review.side_effect = [review_1, review_2]
    mock_defense.propose_revision.side_effect = [revision_1]
    
    final_draft, history = await review_loop.run_loop(initial_draft, max_iterations=3)
    
    assert final_draft == "Draft v2"
    assert len(history) == 2
    assert history[0]['score'] == 5.0
    assert history[1]['score'] == 9.0
    
    # Verify interactions
    assert mock_reviewer.review.call_count == 2
    assert mock_defense.propose_revision.call_count == 1

@pytest.mark.asyncio
async def test_max_iterations_reached(review_loop, mock_reviewer, mock_defense):
    """Test that loop stops at max iterations even if not passed."""
    mock_reviewer.review.return_value = ReviewResult(score=4.0, comments="Bad", passed=False)
    mock_defense.propose_revision.return_value = RevisionProposal(revised_text="New Draft", explanation="Fix")
    
    final_draft, history = await review_loop.run_loop("Draft", max_iterations=2)
    
    assert len(history) == 2
    assert history[-1]['passed'] is False
