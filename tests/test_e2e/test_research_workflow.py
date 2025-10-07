"""End-to-end tests for complete research workflows."""

import pytest
from httpx import AsyncClient
from uuid import uuid4
import asyncio

from src.main import app


@pytest.fixture
async def client():
    """Create async HTTP client for E2E testing."""
    async with AsyncClient(app=app, base_url="http://test", timeout=60.0) as ac:
        yield ac


@pytest.mark.e2e
@pytest.mark.asyncio
class TestCompleteResearchPipeline:
    """E2E tests for complete research workflow from project creation to analysis."""

    async def test_full_research_lifecycle(self, client):
        """Test complete research workflow: Project → Literature → Hypothesis → Experiment → Analysis."""

        # Step 1: Create a new research project
        print("\n1. Creating research project...")
        project_data = {
            "name": "AI in Drug Discovery E2E Test",
            "description": "End-to-end test of AI applications in drug discovery research",
            "research_domain": "Computational Biology",
            "objectives": [
                "Investigate ML models for molecular property prediction",
                "Identify novel drug candidates",
                "Validate computational predictions"
            ]
        }

        project_response = await client.post("/api/v1/projects/", json=project_data)
        assert project_response.status_code == 200
        project = project_response.json()
        project_id = project["id"]

        print(f"✓ Project created: {project['name']} (ID: {project_id})")
        assert project["status"] == "active"

        # Step 2: Ingest relevant literature
        print("\n2. Ingesting literature...")
        literature_data = {
            "query": "machine learning drug discovery molecular properties",
            "max_results": 10,
            "source": "arxiv"
        }

        ingest_response = await client.post(
            f"/api/v1/projects/{project_id}/literature/ingest",
            json=literature_data
        )
        assert ingest_response.status_code == 200
        ingest_result = ingest_response.json()

        print(f"✓ Literature ingested: {ingest_result.get('papers_ingested', 0)} papers")

        # Step 3: Search and verify literature
        print("\n3. Searching literature...")
        search_response = await client.post(
            f"/api/v1/projects/{project_id}/literature/search",
            json={"query": "neural networks molecular", "limit": 5}
        )
        assert search_response.status_code == 200
        literature = search_response.json()

        print(f"✓ Literature search: Found {len(literature)} relevant papers")
        assert len(literature) > 0

        # Step 4: Generate hypotheses from literature
        print("\n4. Generating hypotheses...")
        hypothesis_gen_response = await client.post(
            f"/api/v1/projects/{project_id}/hypotheses/generate",
            json={"max_hypotheses": 3}
        )
        assert hypothesis_gen_response.status_code == 200
        hypotheses = hypothesis_gen_response.json()

        print(f"✓ Hypotheses generated: {len(hypotheses)} hypotheses")
        assert len(hypotheses) > 0

        # Pick the highest novelty hypothesis
        best_hypothesis = max(hypotheses, key=lambda h: h.get("novelty_score", 0))
        hypothesis_id = best_hypothesis["id"]

        print(f"  - Selected hypothesis (novelty: {best_hypothesis.get('novelty_score', 'N/A')})")
        print(f"    {best_hypothesis['content'][:100]}...")

        # Step 5: Validate the hypothesis
        print("\n5. Validating hypothesis...")
        validation_response = await client.post(
            f"/api/v1/hypotheses/{hypothesis_id}/validate"
        )
        assert validation_response.status_code == 200
        validation = validation_response.json()

        print(f"✓ Hypothesis validated: Score = {validation.get('validation_score', 'N/A')}")
        assert "validation_score" in validation

        # Step 6: Design experiment
        print("\n6. Designing experiment...")
        experiment_data = {
            "research_question": f"Can we validate: {best_hypothesis['content'][:50]}?",
            "desired_power": 0.8,
            "significance_level": 0.05,
            "expected_effect_size": 0.5,
            "constraints": {
                "max_participants": 100,
                "max_duration_days": 60,
                "budget": 50000
            }
        }

        experiment_response = await client.post(
            f"/api/v1/hypotheses/{hypothesis_id}/experiments/design",
            json=experiment_data
        )
        assert experiment_response.status_code == 200
        experiment = experiment_response.json()
        experiment_id = experiment["id"]

        print(f"✓ Experiment designed:")
        print(f"  - Title: {experiment.get('title', 'N/A')}")
        print(f"  - Sample size: {experiment.get('sample_size', 'N/A')}")
        print(f"  - Power: {experiment.get('power', 'N/A')}")

        assert experiment["hypothesis_id"] == hypothesis_id
        assert "sample_size" in experiment
        assert "protocol" in experiment

        # Step 7: Update experiment status to in_progress
        print("\n7. Starting experiment execution...")
        status_update_response = await client.patch(
            f"/api/v1/experiments/{experiment_id}",
            json={"status": "in_progress"}
        )
        assert status_update_response.status_code == 200

        updated_experiment = status_update_response.json()
        print(f"✓ Experiment status: {updated_experiment['status']}")
        assert updated_experiment["status"] == "in_progress"

        # Step 8: Verify complete chain
        print("\n8. Verifying complete research chain...")

        # Verify project has hypotheses
        project_hypotheses_response = await client.get(
            f"/api/v1/projects/{project_id}/hypotheses/"
        )
        assert project_hypotheses_response.status_code == 200
        project_hypotheses = project_hypotheses_response.json()
        assert len(project_hypotheses) > 0

        # Verify hypothesis has experiments
        hypothesis_experiments_response = await client.get(
            f"/api/v1/hypotheses/{hypothesis_id}/experiments/"
        )
        assert hypothesis_experiments_response.status_code == 200
        hypothesis_experiments = hypothesis_experiments_response.json()
        assert len(hypothesis_experiments) > 0

        print("✓ Complete research chain verified:")
        print(f"  Project → {len(project_hypotheses)} hypotheses → {len(hypothesis_experiments)} experiments")

        # Step 9: Get final project state
        print("\n9. Final project state...")
        final_project_response = await client.get(f"/api/v1/projects/{project_id}")
        assert final_project_response.status_code == 200
        final_project = final_project_response.json()

        print(f"✓ Project: {final_project['name']}")
        print(f"  Status: {final_project['status']}")
        print(f"  Created: {final_project['created_at']}")

        print("\n✅ E2E Research Pipeline Test PASSED")

        # Return data for potential further testing
        return {
            "project_id": project_id,
            "hypothesis_id": hypothesis_id,
            "experiment_id": experiment_id
        }


@pytest.mark.e2e
@pytest.mark.asyncio
class TestMultiProjectWorkflow:
    """E2E tests for managing multiple research projects."""

    async def test_parallel_projects_workflow(self, client):
        """Test managing multiple research projects in parallel."""

        print("\n=== Parallel Projects Workflow Test ===")

        # Create multiple projects
        projects = []
        domains = ["Biology", "Chemistry", "Physics"]

        print("\n1. Creating multiple projects...")
        for i, domain in enumerate(domains):
            project_data = {
                "name": f"{domain} Research Project {i+1}",
                "description": f"E2E test for {domain}",
                "research_domain": domain,
                "objectives": [f"Objective 1 for {domain}", f"Objective 2 for {domain}"]
            }

            response = await client.post("/api/v1/projects/", json=project_data)
            assert response.status_code == 200
            projects.append(response.json())
            print(f"✓ Created: {project_data['name']}")

        assert len(projects) == 3

        # Ingest literature for all projects concurrently
        print("\n2. Ingesting literature for all projects...")

        async def ingest_for_project(project):
            response = await client.post(
                f"/api/v1/projects/{project['id']}/literature/ingest",
                json={
                    "query": f"{project['research_domain']} research",
                    "max_results": 5,
                    "source": "arxiv"
                }
            )
            return response.json()

        ingest_results = await asyncio.gather(*[
            ingest_for_project(p) for p in projects
        ])

        for i, result in enumerate(ingest_results):
            print(f"✓ Project {i+1}: {result.get('papers_ingested', 0)} papers ingested")

        # Generate hypotheses for all projects
        print("\n3. Generating hypotheses for all projects...")

        async def generate_hypotheses_for_project(project):
            response = await client.post(
                f"/api/v1/projects/{project['id']}/hypotheses/generate",
                json={"max_hypotheses": 2}
            )
            return response.json()

        all_hypotheses = await asyncio.gather(*[
            generate_hypotheses_for_project(p) for p in projects
        ])

        for i, hypotheses in enumerate(all_hypotheses):
            print(f"✓ Project {i+1}: {len(hypotheses)} hypotheses generated")
            assert len(hypotheses) > 0

        # List all projects and verify
        print("\n4. Verifying all projects...")
        list_response = await client.get("/api/v1/projects/")
        assert list_response.status_code == 200
        all_projects = list_response.json()

        created_project_ids = {p["id"] for p in projects}
        listed_project_ids = {p["id"] for p in all_projects}

        assert created_project_ids.issubset(listed_project_ids)
        print(f"✓ All {len(projects)} projects verified in system")

        print("\n✅ Parallel Projects Workflow Test PASSED")


@pytest.mark.e2e
@pytest.mark.asyncio
class TestErrorRecoveryWorkflow:
    """E2E tests for error handling and recovery."""

    async def test_workflow_with_failures(self, client):
        """Test workflow recovery from various failure scenarios."""

        print("\n=== Error Recovery Workflow Test ===")

        # Create project
        print("\n1. Creating project...")
        project_response = await client.post(
            "/api/v1/projects/",
            json={
                "name": "Error Recovery Test",
                "description": "Testing error handling",
                "research_domain": "Test"
            }
        )
        assert project_response.status_code == 200
        project = project_response.json()
        project_id = project["id"]
        print(f"✓ Project created: {project_id}")

        # Test invalid hypothesis ID
        print("\n2. Testing invalid hypothesis ID error...")
        invalid_id = str(uuid4())
        invalid_response = await client.get(f"/api/v1/hypotheses/{invalid_id}")
        assert invalid_response.status_code == 404
        print("✓ Correctly handled invalid hypothesis ID (404)")

        # Test invalid project update
        print("\n3. Testing invalid project update...")
        invalid_update_response = await client.patch(
            f"/api/v1/projects/{project_id}",
            json={"invalid_field": "invalid_value"}
        )
        # Should either ignore invalid fields or return error
        print(f"✓ Invalid update handled (status: {invalid_update_response.status_code})")

        # Create hypothesis manually
        print("\n4. Creating hypothesis manually...")
        hypothesis_response = await client.post(
            f"/api/v1/projects/{project_id}/hypotheses/",
            json={
                "content": "Manual hypothesis for error testing",
                "rationale": "Testing error recovery",
                "expected_outcomes": ["Outcome 1"]
            }
        )
        assert hypothesis_response.status_code == 200
        hypothesis = hypothesis_response.json()
        hypothesis_id = hypothesis["id"]
        print(f"✓ Hypothesis created: {hypothesis_id}")

        # Test experiment design with missing required fields
        print("\n5. Testing experiment design with invalid data...")
        invalid_experiment_response = await client.post(
            f"/api/v1/hypotheses/{hypothesis_id}/experiments/design",
            json={"invalid_field": "test"}  # Missing required fields
        )
        # Should return validation error
        assert invalid_experiment_response.status_code in [400, 422]
        print(f"✓ Invalid experiment data rejected (status: {invalid_experiment_response.status_code})")

        # Test valid experiment design after error
        print("\n6. Creating valid experiment after error...")
        valid_experiment_response = await client.post(
            f"/api/v1/hypotheses/{hypothesis_id}/experiments/design",
            json={
                "research_question": "Test question",
                "desired_power": 0.8
            }
        )
        assert valid_experiment_response.status_code == 200
        experiment = valid_experiment_response.json()
        print(f"✓ Valid experiment created after error: {experiment['id']}")

        # Verify system state is consistent
        print("\n7. Verifying system state consistency...")
        project_check = await client.get(f"/api/v1/projects/{project_id}")
        assert project_check.status_code == 200

        hypothesis_check = await client.get(f"/api/v1/hypotheses/{hypothesis_id}")
        assert hypothesis_check.status_code == 200

        experiment_check = await client.get(f"/api/v1/experiments/{experiment['id']}")
        assert experiment_check.status_code == 200

        print("✓ System state is consistent after errors")

        print("\n✅ Error Recovery Workflow Test PASSED")


@pytest.mark.e2e
@pytest.mark.asyncio
class TestPerformanceWorkflow:
    """E2E tests for system performance under load."""

    async def test_high_volume_workflow(self, client):
        """Test system performance with high volume of operations."""

        print("\n=== High Volume Performance Test ===")

        # Create a project
        print("\n1. Creating project for load testing...")
        project_response = await client.post(
            "/api/v1/projects/",
            json={
                "name": "Performance Test Project",
                "description": "Load testing",
                "research_domain": "Test"
            }
        )
        assert project_response.status_code == 200
        project = project_response.json()
        project_id = project["id"]
        print(f"✓ Project created: {project_id}")

        # Create multiple hypotheses concurrently
        print("\n2. Creating 10 hypotheses concurrently...")
        import time
        start_time = time.time()

        async def create_hypothesis(index):
            return await client.post(
                f"/api/v1/projects/{project_id}/hypotheses/",
                json={
                    "content": f"Performance test hypothesis {index}",
                    "rationale": f"Load testing hypothesis {index}",
                    "expected_outcomes": [f"Outcome {index}"]
                }
            )

        hypothesis_responses = await asyncio.gather(*[
            create_hypothesis(i) for i in range(10)
        ])

        creation_time = time.time() - start_time
        successful_creates = sum(1 for r in hypothesis_responses if r.status_code == 200)

        print(f"✓ Created {successful_creates}/10 hypotheses in {creation_time:.2f}s")
        print(f"  Average: {creation_time/10:.3f}s per hypothesis")

        assert successful_creates >= 8  # At least 80% success rate
        assert creation_time < 30  # Should complete within 30 seconds

        # List all hypotheses and measure response time
        print("\n3. Listing all hypotheses...")
        start_time = time.time()

        list_response = await client.get(
            f"/api/v1/projects/{project_id}/hypotheses/"
        )

        list_time = time.time() - start_time
        assert list_response.status_code == 200
        hypotheses = list_response.json()

        print(f"✓ Listed {len(hypotheses)} hypotheses in {list_time:.3f}s")
        assert list_time < 2.0  # Should be fast with indexes
        assert len(hypotheses) >= 8

        # Concurrent reads
        print("\n4. Testing concurrent read performance...")
        start_time = time.time()

        read_responses = await asyncio.gather(*[
            client.get(f"/api/v1/projects/{project_id}")
            for _ in range(20)
        ])

        concurrent_read_time = time.time() - start_time
        successful_reads = sum(1 for r in read_responses if r.status_code == 200)

        print(f"✓ {successful_reads}/20 concurrent reads in {concurrent_read_time:.3f}s")
        print(f"  Average: {concurrent_read_time/20:.4f}s per read")

        assert successful_reads == 20  # All should succeed
        assert concurrent_read_time < 5.0  # Should be fast

        print("\n✅ High Volume Performance Test PASSED")


@pytest.mark.e2e
@pytest.mark.asyncio
class TestDataIntegrityWorkflow:
    """E2E tests for data integrity and consistency."""

    async def test_data_consistency_across_operations(self, client):
        """Test that data remains consistent across multiple operations."""

        print("\n=== Data Integrity Workflow Test ===")

        # Create project with specific data
        print("\n1. Creating project with specific data...")
        original_data = {
            "name": "Data Integrity Test",
            "description": "Testing data consistency",
            "research_domain": "Computer Science",
            "objectives": ["Objective 1", "Objective 2", "Objective 3"]
        }

        create_response = await client.post("/api/v1/projects/", json=original_data)
        assert create_response.status_code == 200
        project = create_response.json()
        project_id = project["id"]

        print(f"✓ Project created with {len(original_data['objectives'])} objectives")

        # Verify data immediately after creation
        print("\n2. Verifying data immediately after creation...")
        get_response = await client.get(f"/api/v1/projects/{project_id}")
        assert get_response.status_code == 200
        retrieved_project = get_response.json()

        assert retrieved_project["name"] == original_data["name"]
        assert retrieved_project["description"] == original_data["description"]
        assert retrieved_project["research_domain"] == original_data["research_domain"]
        print("✓ Data matches original input")

        # Update project
        print("\n3. Updating project...")
        update_data = {
            "name": "Updated Data Integrity Test",
            "description": "Updated description"
        }

        update_response = await client.patch(
            f"/api/v1/projects/{project_id}",
            json=update_data
        )
        assert update_response.status_code == 200
        updated_project = update_response.json()

        assert updated_project["name"] == update_data["name"]
        assert updated_project["description"] == update_data["description"]
        # Original domain should remain unchanged
        assert updated_project["research_domain"] == original_data["research_domain"]
        print("✓ Update preserved unchanged fields")

        # Create hypothesis and verify relationship
        print("\n4. Creating hypothesis and verifying relationship...")
        hypothesis_response = await client.post(
            f"/api/v1/projects/{project_id}/hypotheses/",
            json={
                "content": "Test hypothesis",
                "rationale": "Testing relationships"
            }
        )
        assert hypothesis_response.status_code == 200
        hypothesis = hypothesis_response.json()

        assert hypothesis["project_id"] == project_id
        print("✓ Hypothesis correctly linked to project")

        # Verify hypothesis appears in project's hypotheses
        print("\n5. Verifying hypothesis in project collection...")
        hypotheses_response = await client.get(
            f"/api/v1/projects/{project_id}/hypotheses/"
        )
        assert hypotheses_response.status_code == 200
        hypotheses = hypotheses_response.json()

        hypothesis_ids = [h["id"] for h in hypotheses]
        assert hypothesis["id"] in hypothesis_ids
        print("✓ Hypothesis appears in project's collection")

        print("\n✅ Data Integrity Workflow Test PASSED")
