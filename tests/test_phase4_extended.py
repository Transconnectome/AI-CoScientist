"""Extended tests for Phase 4 Smart Suggestions and Iterative Improvement.

Tests RAG-powered suggestions and iterative improvement loops.
"""

import pytest


class TestSmartSuggestions:
    """Test smart suggestion engine."""

    def test_rag_context_building(self):
        """Test RAG context building from patterns."""
        from src.services.paper.improvement_service import ImprovementService

        # Mock service instance
        class MockDB:
            pass

        service = ImprovementService(MockDB())

        # Test with patterns
        patterns = [
            {
                "metadata": {
                    "changes": ["Fixed grammar", "Improved clarity"],
                    "improvement_score": 8.5,
                }
            },
            {"metadata": {"changes": ["Added citations"], "improvement_score": 7.8}},
        ]

        exemplars = [
            {"metadata": {"overall_score": 9.2}},
            {"metadata": {"overall_score": 8.7}},
        ]

        context = service._build_rag_context(patterns, exemplars)

        assert "Reference successful improvements" in context
        assert "Fixed grammar" in context
        assert "High-quality examples" in context
        assert "9.2" in context

    def test_rag_context_empty(self):
        """Test RAG context with no patterns."""
        from src.services.paper.improvement_service import ImprovementService

        class MockDB:
            pass

        service = ImprovementService(MockDB())

        context = service._build_rag_context([], [])

        assert "Reference successful improvements" in context
        # Should handle empty gracefully


class TestIterativeImprovement:
    """Test iterative improvement loop."""

    def test_iteration_request_schema(self):
        """Test IterativeImprovementRequest schema."""
        from src.schemas.improvement import IterativeImprovementRequest

        request = IterativeImprovementRequest(
            target_score=8.5, max_iterations=3, focus_areas=["clarity", "methodology"]
        )

        assert request.target_score == 8.5
        assert request.max_iterations == 3
        assert len(request.focus_areas) == 2

    def test_iteration_response_schema(self):
        """Test IterativeImprovementResponse schema."""
        from src.schemas.improvement import IterativeImprovementResponse

        response = IterativeImprovementResponse(
            session_id="test-session",
            iterations_completed=3,
            improvements_applied=9,
            initial_score=6.5,
            final_score=8.2,
            score_improvement=1.7,
            target_reached=False,
        )

        assert response.iterations_completed == 3
        assert response.improvements_applied == 9
        assert response.score_improvement == 1.7
        assert response.target_reached is False


class TestSmartSuggestionSchemas:
    """Test smart suggestion schemas."""

    def test_smart_suggestion_item(self):
        """Test SmartSuggestionItem schema."""
        from src.schemas.improvement import SmartSuggestionItem

        item = SmartSuggestionItem(
            section_name="Introduction",
            improved_content="Improved text",
            metadata={"type": "clarity", "changes": ["Fixed grammar"]},
            expected_improvement=1.5,
            similar_patterns_used=3,
            exemplars_referenced=2,
        )

        assert item.section_name == "Introduction"
        assert item.similar_patterns_used == 3
        assert item.exemplars_referenced == 2

    def test_smart_suggestion_response(self):
        """Test SmartSuggestionResponse schema."""
        from src.schemas.improvement import (
            SmartSuggestionResponse,
            SmartSuggestionItem,
        )

        suggestions = [
            SmartSuggestionItem(
                section_name="Abstract",
                improved_content="Better abstract",
                metadata={},
                expected_improvement=1.0,
            )
        ]

        response = SmartSuggestionResponse(
            suggestions=suggestions, total_suggestions=1, rag_enhanced=True
        )

        assert response.total_suggestions == 1
        assert response.rag_enhanced is True


class TestAPIEndpoints:
    """Test API endpoint definitions."""

    def test_improvements_router_exists(self):
        """Test that improvements router is defined."""
        from src.api.v1 import improvements

        assert improvements.router is not None

    def test_smart_suggestions_endpoint_defined(self):
        """Test smart suggestions endpoint is defined."""
        from src.api.v1.improvements import get_smart_suggestions

        assert get_smart_suggestions is not None
        assert callable(get_smart_suggestions)

    def test_iterative_improvement_endpoint_defined(self):
        """Test iterative improvement endpoint is defined."""
        from src.api.v1.improvements import start_iterative_improvement

        assert start_iterative_improvement is not None
        assert callable(start_iterative_improvement)


def test_extended_imports():
    """Test that all extended Phase 4 modules can be imported."""
    # Service methods
    from src.services.paper.improvement_service import ImprovementService

    # Schemas
    from src.schemas.improvement import (
        SmartSuggestionResponse,
        SmartSuggestionItem,
        IterativeImprovementRequest,
        IterativeImprovementResponse,
    )

    # API endpoints
    from src.api.v1.improvements import (
        get_smart_suggestions,
        start_iterative_improvement,
    )

    # All imports successful
    assert ImprovementService is not None
    assert SmartSuggestionResponse is not None
    assert SmartSuggestionItem is not None
    assert IterativeImprovementRequest is not None
    assert IterativeImprovementResponse is not None
    assert get_smart_suggestions is not None
    assert start_iterative_improvement is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
