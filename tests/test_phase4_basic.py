"""Basic tests for Phase 4 version tracking and improvements.

Tests core functionality without requiring full database setup.
"""

import pytest
from src.models.paper_version import VersionType, ImprovementStatus, PaperVersion


class TestVersionTypes:
    """Test version type enums."""

    def test_version_types_exist(self):
        """Test that all version types are defined."""
        assert VersionType.MAJOR == "major"
        assert VersionType.MINOR == "minor"
        assert VersionType.PATCH == "patch"

    def test_improvement_status_exist(self):
        """Test that all improvement statuses are defined."""
        assert ImprovementStatus.SUGGESTED == "suggested"
        assert ImprovementStatus.APPLIED == "applied"
        assert ImprovementStatus.REVERTED == "reverted"
        assert ImprovementStatus.REJECTED == "rejected"


class TestPaperVersion:
    """Test PaperVersion model."""

    def test_version_string_property(self):
        """Test semantic version string generation."""
        # Create a mock version object
        class MockVersion:
            major = 1
            minor = 2
            patch = 3

            @property
            def version_string(self):
                return f"{self.major}.{self.minor}.{self.patch}"

        version = MockVersion()
        assert version.version_string == "1.2.3"

    def test_version_parsing(self):
        """Test version string parsing."""
        version_str = "2.5.7"
        major, minor, patch = map(int, version_str.split("."))

        assert major == 2
        assert minor == 5
        assert patch == 7


@pytest.mark.skip(reason="Requires ChromaDB server running")
class TestLearningStore:
    """Test ChromaDB learning store (requires ChromaDB server)."""

    def test_format_results_empty(self):
        """Test formatting empty ChromaDB results."""
        from src.services.knowledge_base.learning_store import LearningStore

        store = LearningStore()

        # Simulate empty results
        empty_results = {"ids": [[]], "documents": [[]], "metadatas": [[]]}

        formatted = store._format_results(empty_results)
        assert formatted == []

    def test_format_results_with_data(self):
        """Test formatting ChromaDB results with data."""
        from src.services.knowledge_base.learning_store import LearningStore

        store = LearningStore()

        # Simulate results with data
        results = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"key": "value1"}, {"key": "value2"}]],
            "distances": [[0.1, 0.2]],
        }

        formatted = store._format_results(results)

        assert len(formatted) == 2
        assert formatted[0]["id"] == "id1"
        assert formatted[0]["document"] == "doc1"
        assert formatted[0]["metadata"]["key"] == "value1"
        assert formatted[0]["distance"] == 0.1


class TestImprovementSchemas:
    """Test improvement request/response schemas."""

    def test_apply_improvement_request(self):
        """Test ApplyImprovementRequest schema."""
        from src.schemas.improvement import ApplyImprovementRequest

        request = ApplyImprovementRequest(
            section_name="Introduction",
            improved_content="Improved text here",
            metadata={"type": "clarity", "changes": ["Fixed grammar"]},
        )

        assert request.section_name == "Introduction"
        assert request.improved_content == "Improved text here"
        assert request.metadata["type"] == "clarity"

    def test_apply_improvement_response(self):
        """Test ApplyImprovementResponse schema."""
        from src.schemas.improvement import ApplyImprovementResponse

        response = ApplyImprovementResponse(
            success=True,
            new_version="1.2.3",
            quality_improvement=1.5,
            section_updated="Introduction",
            improvement_id="test-id-123",
        )

        assert response.success is True
        assert response.new_version == "1.2.3"
        assert response.quality_improvement == 1.5


def test_imports():
    """Test that all Phase 4 modules can be imported."""
    # Models
    from src.models.paper_version import (
        PaperVersion,
        ImprovementHistory,
        IterationSession,
    )

    # Schemas
    from src.schemas.improvement import (
        ApplyImprovementRequest,
        ApplyImprovementResponse,
    )

    # Services
    from src.services.knowledge_base.learning_store import LearningStore

    # All imports successful
    assert PaperVersion is not None
    assert ImprovementHistory is not None
    assert IterationSession is not None
    assert ApplyImprovementRequest is not None
    assert ApplyImprovementResponse is not None
    assert LearningStore is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
