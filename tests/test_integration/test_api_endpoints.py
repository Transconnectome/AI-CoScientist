"""Integration tests for API endpoints."""

import pytest
from httpx import AsyncClient
from uuid import uuid4
from datetime import datetime

from src.main import app
from src.models.project import ProjectStatus, HypothesisStatus, ExperimentStatus


@pytest.fixture
async def client():
    """Create async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def test_project(client):
    """Create a test project for integration tests."""
    project_data = {
        "name": "Integration Test Project",
        "description": "Project for integration testing",
        "research_domain": "Computer Science",
        "objectives": ["Test objective 1", "Test objective 2"]
    }

    response = await client.post("/api/v1/projects/", json=project_data)
    assert response.status_code == 200
    return response.json()


@pytest.fixture
async def test_hypothesis(client, test_project):
    """Create a test hypothesis for integration tests."""
    hypothesis_data = {
        "project_id": test_project["id"],
        "content": "Test hypothesis content",
        "rationale": "Test rationale",
        "expected_outcomes": ["Outcome 1", "Outcome 2"]
    }

    response = await client.post(
        f"/api/v1/projects/{test_project['id']}/hypotheses/",
        json=hypothesis_data
    )
    assert response.status_code == 200
    return response.json()


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Tests for health check endpoint."""

    async def test_health_check(self, client):
        """Test health check endpoint."""
        response = await client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data


@pytest.mark.asyncio
class TestProjectsAPI:
    """Integration tests for Projects API."""

    async def test_create_project(self, client):
        """Test creating a new project."""
        project_data = {
            "name": "Test Project",
            "description": "A test research project",
            "research_domain": "Biology",
            "objectives": ["Understand X", "Test Y"]
        }

        response = await client.post("/api/v1/projects/", json=project_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == project_data["name"]
        assert data["status"] == ProjectStatus.ACTIVE
        assert "id" in data
        assert "created_at" in data

    async def test_get_project(self, client, test_project):
        """Test retrieving a specific project."""
        response = await client.get(f"/api/v1/projects/{test_project['id']}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_project["id"]
        assert data["name"] == test_project["name"]

    async def test_list_projects(self, client, test_project):
        """Test listing all projects."""
        response = await client.get("/api/v1/projects/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert any(p["id"] == test_project["id"] for p in data)

    async def test_update_project(self, client, test_project):
        """Test updating a project."""
        update_data = {
            "name": "Updated Project Name",
            "description": "Updated description"
        }

        response = await client.patch(
            f"/api/v1/projects/{test_project['id']}",
            json=update_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]
        assert data["description"] == update_data["description"]

    async def test_delete_project(self, client):
        """Test deleting a project."""
        # Create a project to delete
        project_data = {
            "name": "Project to Delete",
            "description": "This will be deleted",
            "research_domain": "Physics"
        }
        create_response = await client.post("/api/v1/projects/", json=project_data)
        project_id = create_response.json()["id"]

        # Delete the project
        response = await client.delete(f"/api/v1/projects/{project_id}")

        assert response.status_code == 204

        # Verify it's deleted
        get_response = await client.get(f"/api/v1/projects/{project_id}")
        assert get_response.status_code == 404


@pytest.mark.asyncio
class TestLiteratureAPI:
    """Integration tests for Literature API."""

    async def test_ingest_literature(self, client, test_project):
        """Test ingesting literature for a project."""
        literature_data = {
            "query": "machine learning",
            "max_results": 5,
            "source": "arxiv"
        }

        response = await client.post(
            f"/api/v1/projects/{test_project['id']}/literature/ingest",
            json=literature_data
        )

        assert response.status_code == 200
        data = response.json()
        assert "papers_ingested" in data
        assert data["papers_ingested"] >= 0

    async def test_search_literature(self, client, test_project):
        """Test searching literature."""
        search_data = {
            "query": "neural networks",
            "limit": 10
        }

        response = await client.post(
            f"/api/v1/projects/{test_project['id']}/literature/search",
            json=search_data
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    async def test_get_literature_item(self, client, test_project):
        """Test retrieving a specific literature item."""
        # First ingest some literature
        await client.post(
            f"/api/v1/projects/{test_project['id']}/literature/ingest",
            json={"query": "test", "max_results": 1, "source": "arxiv"}
        )

        # Then try to get literature list
        response = await client.get(
            f"/api/v1/projects/{test_project['id']}/literature/"
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
class TestHypothesesAPI:
    """Integration tests for Hypotheses API."""

    async def test_generate_hypotheses(self, client, test_project):
        """Test generating hypotheses from literature."""
        response = await client.post(
            f"/api/v1/projects/{test_project['id']}/hypotheses/generate",
            json={"max_hypotheses": 3}
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    async def test_create_hypothesis_manually(self, client, test_project):
        """Test creating a hypothesis manually."""
        hypothesis_data = {
            "content": "Manual test hypothesis",
            "rationale": "Testing manual creation",
            "expected_outcomes": ["Expected outcome 1"]
        }

        response = await client.post(
            f"/api/v1/projects/{test_project['id']}/hypotheses/",
            json=hypothesis_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == hypothesis_data["content"]
        assert data["project_id"] == test_project["id"]

    async def test_get_hypothesis(self, client, test_project, test_hypothesis):
        """Test retrieving a specific hypothesis."""
        response = await client.get(
            f"/api/v1/hypotheses/{test_hypothesis['id']}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_hypothesis["id"]

    async def test_list_hypotheses(self, client, test_project, test_hypothesis):
        """Test listing hypotheses for a project."""
        response = await client.get(
            f"/api/v1/projects/{test_project['id']}/hypotheses/"
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert any(h["id"] == test_hypothesis["id"] for h in data)

    async def test_validate_hypothesis(self, client, test_hypothesis):
        """Test validating a hypothesis."""
        response = await client.post(
            f"/api/v1/hypotheses/{test_hypothesis['id']}/validate"
        )

        assert response.status_code == 200
        data = response.json()
        assert "validation_score" in data
        assert "status" in data


@pytest.mark.asyncio
class TestExperimentsAPI:
    """Integration tests for Experiments API."""

    async def test_design_experiment(self, client, test_hypothesis):
        """Test designing an experiment for a hypothesis."""
        design_data = {
            "research_question": "Test research question",
            "desired_power": 0.8,
            "significance_level": 0.05,
            "expected_effect_size": 0.5
        }

        response = await client.post(
            f"/api/v1/hypotheses/{test_hypothesis['id']}/experiments/design",
            json=design_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["hypothesis_id"] == test_hypothesis["id"]
        assert "sample_size" in data
        assert "protocol" in data

    async def test_get_experiment(self, client, test_hypothesis):
        """Test retrieving a specific experiment."""
        # First create an experiment
        design_data = {
            "research_question": "Test question",
            "desired_power": 0.8
        }
        create_response = await client.post(
            f"/api/v1/hypotheses/{test_hypothesis['id']}/experiments/design",
            json=design_data
        )
        experiment = create_response.json()

        # Then retrieve it
        response = await client.get(
            f"/api/v1/experiments/{experiment['id']}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == experiment["id"]

    async def test_list_experiments(self, client, test_hypothesis):
        """Test listing experiments for a hypothesis."""
        response = await client.get(
            f"/api/v1/hypotheses/{test_hypothesis['id']}/experiments/"
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    async def test_update_experiment_status(self, client, test_hypothesis):
        """Test updating experiment status."""
        # Create an experiment
        design_data = {"research_question": "Test"}
        create_response = await client.post(
            f"/api/v1/hypotheses/{test_hypothesis['id']}/experiments/design",
            json=design_data
        )
        experiment_id = create_response.json()["id"]

        # Update status
        update_data = {"status": "in_progress"}
        response = await client.patch(
            f"/api/v1/experiments/{experiment_id}",
            json=update_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == ExperimentStatus.IN_PROGRESS


@pytest.mark.asyncio
class TestCompleteWorkflow:
    """Integration tests for complete research workflows."""

    async def test_complete_research_pipeline(self, client):
        """Test complete research workflow from project to experiment."""
        # 1. Create project
        project_data = {
            "name": "Complete Workflow Test",
            "description": "Testing complete pipeline",
            "research_domain": "Computer Science"
        }
        project_response = await client.post(
            "/api/v1/projects/",
            json=project_data
        )
        assert project_response.status_code == 200
        project = project_response.json()

        # 2. Ingest literature
        literature_response = await client.post(
            f"/api/v1/projects/{project['id']}/literature/ingest",
            json={"query": "artificial intelligence", "max_results": 3}
        )
        assert literature_response.status_code == 200

        # 3. Generate hypotheses
        hypotheses_response = await client.post(
            f"/api/v1/projects/{project['id']}/hypotheses/generate",
            json={"max_hypotheses": 2}
        )
        assert hypotheses_response.status_code == 200
        hypotheses = hypotheses_response.json()
        assert len(hypotheses) > 0

        # 4. Validate hypothesis
        hypothesis_id = hypotheses[0]["id"]
        validate_response = await client.post(
            f"/api/v1/hypotheses/{hypothesis_id}/validate"
        )
        assert validate_response.status_code == 200

        # 5. Design experiment
        experiment_response = await client.post(
            f"/api/v1/hypotheses/{hypothesis_id}/experiments/design",
            json={
                "research_question": "Test question",
                "desired_power": 0.8
            }
        )
        assert experiment_response.status_code == 200
        experiment = experiment_response.json()

        # 6. Verify complete chain
        assert experiment["hypothesis_id"] == hypothesis_id
        assert "sample_size" in experiment
        assert experiment["status"] == ExperimentStatus.DESIGNED

    async def test_error_handling_invalid_ids(self, client):
        """Test error handling for invalid IDs."""
        invalid_id = str(uuid4())

        # Invalid project ID
        response = await client.get(f"/api/v1/projects/{invalid_id}")
        assert response.status_code == 404

        # Invalid hypothesis ID
        response = await client.get(f"/api/v1/hypotheses/{invalid_id}")
        assert response.status_code == 404

        # Invalid experiment ID
        response = await client.get(f"/api/v1/experiments/{invalid_id}")
        assert response.status_code == 404

    async def test_validation_errors(self, client):
        """Test validation error handling."""
        # Missing required fields
        response = await client.post(
            "/api/v1/projects/",
            json={"description": "Missing name field"}
        )
        assert response.status_code == 422

        # Invalid data types
        response = await client.post(
            "/api/v1/projects/",
            json={"name": 123, "description": "Invalid type"}
        )
        assert response.status_code == 422


@pytest.mark.asyncio
class TestConcurrency:
    """Integration tests for concurrent operations."""

    async def test_concurrent_project_creation(self, client):
        """Test creating multiple projects concurrently."""
        import asyncio

        async def create_project(name: str):
            return await client.post(
                "/api/v1/projects/",
                json={
                    "name": name,
                    "description": f"Concurrent test {name}",
                    "research_domain": "Test"
                }
            )

        # Create 5 projects concurrently
        responses = await asyncio.gather(*[
            create_project(f"Concurrent Project {i}")
            for i in range(5)
        ])

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

        # All should have unique IDs
        project_ids = [r.json()["id"] for r in responses]
        assert len(set(project_ids)) == 5

    async def test_concurrent_reads(self, client, test_project):
        """Test concurrent read operations."""
        import asyncio

        async def get_project():
            return await client.get(f"/api/v1/projects/{test_project['id']}")

        # Perform 10 concurrent reads
        responses = await asyncio.gather(*[get_project() for _ in range(10)])

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

        # All should return same data
        data_list = [r.json() for r in responses]
        assert all(d["id"] == test_project["id"] for d in data_list)
