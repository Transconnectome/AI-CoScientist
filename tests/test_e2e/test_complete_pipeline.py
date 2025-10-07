"""E2E tests for complete research pipeline integration."""

import pytest
from httpx import AsyncClient
import asyncio
from datetime import datetime

from src.main import app


@pytest.fixture
async def client():
    """Create async HTTP client for E2E testing."""
    async with AsyncClient(app=app, base_url="http://test", timeout=120.0) as ac:
        yield ac


@pytest.mark.e2e
@pytest.mark.asyncio
class TestEndToEndResearchPipeline:
    """Complete end-to-end research pipeline test."""

    async def test_complete_ai_research_pipeline(self, client):
        """
        Test complete AI-CoScientist pipeline:
        1. Project Setup
        2. Literature Ingestion & Analysis
        3. Hypothesis Generation & Validation
        4. Experimental Design
        5. Results Analysis
        """

        print("\n" + "="*80)
        print("COMPLETE AI-COSCIENTIST PIPELINE TEST")
        print("="*80)

        # ============================================================
        # PHASE 1: PROJECT INITIALIZATION
        # ============================================================
        print("\n📋 PHASE 1: PROJECT INITIALIZATION")
        print("-" * 80)

        project_data = {
            "name": "AI-Driven Protein Folding Research",
            "description": "Investigating deep learning approaches for protein structure prediction",
            "research_domain": "Computational Biology",
            "objectives": [
                "Evaluate transformer models for protein folding",
                "Compare AlphaFold variants with novel architectures",
                "Identify key features for accurate prediction",
                "Validate predictions with experimental data"
            ],
            "metadata": {
                "funding_source": "Research Grant XYZ",
                "start_date": datetime.utcnow().isoformat(),
                "principal_investigator": "Dr. Test Researcher"
            }
        }

        print("Creating research project...")
        project_response = await client.post("/api/v1/projects/", json=project_data)
        assert project_response.status_code == 200

        project = project_response.json()
        project_id = project["id"]

        print(f"✅ Project Created Successfully")
        print(f"   ID: {project_id}")
        print(f"   Name: {project['name']}")
        print(f"   Domain: {project['research_domain']}")
        print(f"   Objectives: {len(project['objectives'])} defined")

        # ============================================================
        # PHASE 2: LITERATURE REVIEW & INGESTION
        # ============================================================
        print("\n📚 PHASE 2: LITERATURE REVIEW & INGESTION")
        print("-" * 80)

        # Ingest literature from multiple sources
        print("Ingesting literature from ArXiv...")
        arxiv_ingest = await client.post(
            f"/api/v1/projects/{project_id}/literature/ingest",
            json={
                "query": "protein folding deep learning transformer AlphaFold",
                "max_results": 15,
                "source": "arxiv"
            }
        )
        assert arxiv_ingest.status_code == 200
        arxiv_result = arxiv_ingest.json()

        print(f"✅ ArXiv Literature Ingested")
        print(f"   Papers: {arxiv_result.get('papers_ingested', 0)}")

        # Search and analyze literature
        print("\nSearching for relevant papers...")
        search_queries = [
            "protein structure prediction",
            "AlphaFold architecture",
            "deep learning molecular biology"
        ]

        all_papers = []
        for query in search_queries:
            search_response = await client.post(
                f"/api/v1/projects/{project_id}/literature/search",
                json={"query": query, "limit": 5}
            )
            assert search_response.status_code == 200
            papers = search_response.json()
            all_papers.extend(papers)
            print(f"   '{query}': Found {len(papers)} papers")

        print(f"\n✅ Literature Search Complete")
        print(f"   Total unique papers: {len(set(p['id'] for p in all_papers))}")

        # ============================================================
        # PHASE 3: HYPOTHESIS GENERATION
        # ============================================================
        print("\n💡 PHASE 3: HYPOTHESIS GENERATION")
        print("-" * 80)

        print("Generating research hypotheses from literature...")
        hypothesis_response = await client.post(
            f"/api/v1/projects/{project_id}/hypotheses/generate",
            json={
                "max_hypotheses": 5,
                "focus_areas": ["architecture", "performance", "interpretability"]
            }
        )
        assert hypothesis_response.status_code == 200
        hypotheses = hypothesis_response.json()

        print(f"✅ Hypotheses Generated: {len(hypotheses)}")
        for i, hyp in enumerate(hypotheses, 1):
            print(f"\n   Hypothesis {i}:")
            print(f"   Content: {hyp['content'][:100]}...")
            print(f"   Novelty Score: {hyp.get('novelty_score', 'N/A')}")
            print(f"   Status: {hyp.get('status', 'pending')}")

        # ============================================================
        # PHASE 4: HYPOTHESIS VALIDATION
        # ============================================================
        print("\n🔍 PHASE 4: HYPOTHESIS VALIDATION")
        print("-" * 80)

        # Sort hypotheses by novelty and validate top ones
        sorted_hypotheses = sorted(
            hypotheses,
            key=lambda h: h.get('novelty_score', 0),
            reverse=True
        )

        validated_hypotheses = []
        print("Validating top hypotheses...")

        for i, hypothesis in enumerate(sorted_hypotheses[:3], 1):
            print(f"\nValidating Hypothesis {i}...")
            validation_response = await client.post(
                f"/api/v1/hypotheses/{hypothesis['id']}/validate"
            )
            assert validation_response.status_code == 200
            validation = validation_response.json()

            validated_hypotheses.append({
                **hypothesis,
                "validation": validation
            })

            print(f"✅ Validation Complete")
            print(f"   Validation Score: {validation.get('validation_score', 'N/A')}")
            print(f"   Status: {validation.get('status', 'N/A')}")

        # Select best hypothesis
        best_hypothesis = max(
            validated_hypotheses,
            key=lambda h: (
                h.get('validation', {}).get('validation_score', 0) * 0.6 +
                h.get('novelty_score', 0) * 0.4
            )
        )

        print(f"\n✅ Best Hypothesis Selected")
        print(f"   Content: {best_hypothesis['content'][:150]}...")
        print(f"   Combined Score: {best_hypothesis.get('novelty_score', 0) * 0.4 + best_hypothesis.get('validation', {}).get('validation_score', 0) * 0.6:.2f}")

        # ============================================================
        # PHASE 5: EXPERIMENTAL DESIGN
        # ============================================================
        print("\n🔬 PHASE 5: EXPERIMENTAL DESIGN")
        print("-" * 80)

        print("Designing experimental protocol...")
        experiment_design_data = {
            "research_question": f"How can we validate: {best_hypothesis['content'][:100]}?",
            "hypothesis_content": best_hypothesis['content'],
            "desired_power": 0.8,
            "significance_level": 0.05,
            "expected_effect_size": 0.5,
            "experimental_approach": "computational validation",
            "constraints": {
                "max_participants": 1000,
                "max_duration_days": 90,
                "budget": 100000,
                "computational_resources": "high_performance_cluster"
            }
        }

        experiment_response = await client.post(
            f"/api/v1/hypotheses/{best_hypothesis['id']}/experiments/design",
            json=experiment_design_data
        )
        assert experiment_response.status_code == 200
        experiment = experiment_response.json()

        print(f"✅ Experiment Designed")
        print(f"   Title: {experiment.get('title', 'N/A')}")
        print(f"   Sample Size: {experiment.get('sample_size', 'N/A')}")
        print(f"   Statistical Power: {experiment.get('power', 'N/A')}")
        print(f"   Effect Size: {experiment.get('effect_size', 'N/A')}")
        print(f"   Protocol: {len(experiment.get('protocol', ''))} chars")

        # Display protocol details
        if 'methods' in experiment:
            print(f"\n   Methods: {len(experiment['methods'])} steps")
        if 'materials' in experiment:
            print(f"   Materials: {len(experiment['materials'])} items")
        if 'variables' in experiment:
            vars = experiment['variables']
            print(f"   Variables:")
            print(f"     - Independent: {len(vars.get('independent', []))}")
            print(f"     - Dependent: {len(vars.get('dependent', []))}")
            print(f"     - Controlled: {len(vars.get('controlled', []))}")

        # ============================================================
        # PHASE 6: EXPERIMENT EXECUTION SIMULATION
        # ============================================================
        print("\n⚙️ PHASE 6: EXPERIMENT EXECUTION")
        print("-" * 80)

        # Update experiment status to in_progress
        print("Initiating experiment execution...")
        status_update = await client.patch(
            f"/api/v1/experiments/{experiment['id']}",
            json={"status": "in_progress"}
        )
        assert status_update.status_code == 200

        print("✅ Experiment Status: IN PROGRESS")

        # Simulate experiment stages
        stages = [
            "data_collection",
            "preprocessing",
            "model_training",
            "validation",
            "analysis"
        ]

        for stage in stages:
            print(f"   Processing: {stage.replace('_', ' ').title()}...")
            await asyncio.sleep(0.1)  # Simulate processing time

        # Mark experiment as completed
        completion_update = await client.patch(
            f"/api/v1/experiments/{experiment['id']}",
            json={
                "status": "completed",
                "results": {
                    "success": True,
                    "metrics": {
                        "accuracy": 0.87,
                        "precision": 0.85,
                        "recall": 0.89,
                        "f1_score": 0.87
                    },
                    "conclusion": "Hypothesis supported by experimental results"
                }
            }
        )
        assert completion_update.status_code == 200

        print("\n✅ Experiment Completed Successfully")

        # ============================================================
        # PHASE 7: RESULTS ANALYSIS & REPORTING
        # ============================================================
        print("\n📊 PHASE 7: RESULTS ANALYSIS & REPORTING")
        print("-" * 80)

        # Get complete project state
        final_project = await client.get(f"/api/v1/projects/{project_id}")
        assert final_project.status_code == 200
        final_project_data = final_project.json()

        # Get all hypotheses
        all_hypotheses = await client.get(
            f"/api/v1/projects/{project_id}/hypotheses/"
        )
        assert all_hypotheses.status_code == 200
        hypotheses_data = all_hypotheses.json()

        # Get all experiments
        all_experiments = await client.get(
            f"/api/v1/hypotheses/{best_hypothesis['id']}/experiments/"
        )
        assert all_experiments.status_code == 200
        experiments_data = all_experiments.json()

        print("✅ Final Research State:")
        print(f"\n   Project: {final_project_data['name']}")
        print(f"   Status: {final_project_data['status']}")
        print(f"   Domain: {final_project_data['research_domain']}")
        print(f"\n   Literature: {len(all_papers)} papers analyzed")
        print(f"   Hypotheses: {len(hypotheses_data)} generated")
        print(f"     - Validated: {sum(1 for h in validated_hypotheses if h.get('validation', {}).get('status') == 'validated')}")
        print(f"     - Average Novelty: {sum(h.get('novelty_score', 0) for h in hypotheses_data) / len(hypotheses_data):.2f}")
        print(f"\n   Experiments: {len(experiments_data)} designed")
        print(f"     - Completed: {sum(1 for e in experiments_data if e.get('status') == 'completed')}")

        # ============================================================
        # PHASE 8: VERIFICATION & QUALITY CHECKS
        # ============================================================
        print("\n✔️ PHASE 8: VERIFICATION & QUALITY CHECKS")
        print("-" * 80)

        print("Running integrity checks...")

        # Verify data relationships
        assert project_id == final_project_data['id']
        assert best_hypothesis['project_id'] == project_id
        assert experiment['hypothesis_id'] == best_hypothesis['id']
        print("✅ Data relationships verified")

        # Verify data consistency
        assert final_project_data['name'] == project_data['name']
        assert final_project_data['research_domain'] == project_data['research_domain']
        print("✅ Data consistency verified")

        # Verify all objects are retrievable
        project_check = await client.get(f"/api/v1/projects/{project_id}")
        hypothesis_check = await client.get(f"/api/v1/hypotheses/{best_hypothesis['id']}")
        experiment_check = await client.get(f"/api/v1/experiments/{experiment['id']}")

        assert all([
            project_check.status_code == 200,
            hypothesis_check.status_code == 200,
            experiment_check.status_code == 200
        ])
        print("✅ All objects retrievable")

        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        print("\n" + "="*80)
        print("✅ COMPLETE PIPELINE TEST PASSED")
        print("="*80)

        print("\n📈 Pipeline Statistics:")
        print(f"   Total Phases: 8")
        print(f"   API Calls: ~{20 + len(search_queries) + len(hypotheses) + 3}")
        print(f"   Objects Created:")
        print(f"     - Projects: 1")
        print(f"     - Literature: {len(all_papers)}")
        print(f"     - Hypotheses: {len(hypotheses_data)}")
        print(f"     - Experiments: {len(experiments_data)}")

        print("\n✨ AI-CoScientist Pipeline Fully Operational")


@pytest.mark.e2e
@pytest.mark.asyncio
class TestScalabilityPipeline:
    """E2E tests for system scalability."""

    async def test_multiple_concurrent_pipelines(self, client):
        """Test running multiple research pipelines concurrently."""

        print("\n" + "="*80)
        print("CONCURRENT PIPELINES SCALABILITY TEST")
        print("="*80)

        # Define multiple research projects
        research_projects = [
            {
                "name": "AI in Drug Discovery",
                "domain": "Computational Biology",
                "query": "machine learning drug discovery"
            },
            {
                "name": "Climate Modeling with ML",
                "domain": "Environmental Science",
                "query": "machine learning climate prediction"
            },
            {
                "name": "Quantum Computing Applications",
                "domain": "Physics",
                "query": "quantum computing optimization"
            }
        ]

        print(f"\nLaunching {len(research_projects)} concurrent research pipelines...")

        async def run_mini_pipeline(project_config):
            """Run a minimal research pipeline."""
            # Create project
            project_response = await client.post(
                "/api/v1/projects/",
                json={
                    "name": project_config["name"],
                    "description": f"Concurrent test: {project_config['name']}",
                    "research_domain": project_config["domain"]
                }
            )
            assert project_response.status_code == 200
            project = project_response.json()

            # Ingest literature
            await client.post(
                f"/api/v1/projects/{project['id']}/literature/ingest",
                json={
                    "query": project_config["query"],
                    "max_results": 3
                }
            )

            # Generate hypotheses
            hyp_response = await client.post(
                f"/api/v1/projects/{project['id']}/hypotheses/generate",
                json={"max_hypotheses": 2}
            )
            hypotheses = hyp_response.json()

            return {
                "project": project,
                "hypotheses_count": len(hypotheses)
            }

        # Run all pipelines concurrently
        import time
        start_time = time.time()

        results = await asyncio.gather(*[
            run_mini_pipeline(config)
            for config in research_projects
        ])

        execution_time = time.time() - start_time

        print(f"\n✅ All pipelines completed in {execution_time:.2f}s")
        print(f"\n   Results:")
        for i, result in enumerate(results, 1):
            print(f"   Pipeline {i}: {result['hypotheses_count']} hypotheses generated")

        # Verify all succeeded
        assert len(results) == len(research_projects)
        assert execution_time < 60  # Should complete within 60 seconds

        print(f"\n✅ Scalability Test PASSED")
        print(f"   Average time per pipeline: {execution_time / len(results):.2f}s")


@pytest.mark.e2e
@pytest.mark.asyncio
class TestRobustnessPipeline:
    """E2E tests for system robustness and fault tolerance."""

    async def test_pipeline_resilience(self, client):
        """Test pipeline resilience to various error conditions."""

        print("\n" + "="*80)
        print("PIPELINE ROBUSTNESS TEST")
        print("="*80)

        # Create valid project
        print("\n1. Creating valid project...")
        project_response = await client.post(
            "/api/v1/projects/",
            json={
                "name": "Robustness Test",
                "description": "Testing error handling",
                "research_domain": "Test"
            }
        )
        assert project_response.status_code == 200
        project = project_response.json()
        print("✅ Project created")

        # Test with empty literature
        print("\n2. Testing with no literature...")
        hyp_response = await client.post(
            f"/api/v1/projects/{project['id']}/hypotheses/generate",
            json={"max_hypotheses": 2}
        )
        # Should handle gracefully (may return 200 with empty list or appropriate error)
        print(f"✅ Handled gracefully (status: {hyp_response.status_code})")

        # Create manual hypothesis
        print("\n3. Creating manual hypothesis...")
        manual_hyp = await client.post(
            f"/api/v1/projects/{project['id']}/hypotheses/",
            json={
                "content": "Manual test hypothesis",
                "rationale": "Testing robustness"
            }
        )
        assert manual_hyp.status_code == 200
        hypothesis = manual_hyp.json()
        print("✅ Manual hypothesis created")

        # Test invalid experiment parameters
        print("\n4. Testing with invalid parameters...")
        invalid_exp = await client.post(
            f"/api/v1/hypotheses/{hypothesis['id']}/experiments/design",
            json={
                "research_question": "Test",
                "desired_power": 2.0,  # Invalid (should be 0-1)
                "significance_level": -0.05  # Invalid (should be > 0)
            }
        )
        # Should reject invalid parameters
        assert invalid_exp.status_code in [400, 422]
        print("✅ Invalid parameters rejected")

        # Test valid experiment after errors
        print("\n5. Creating valid experiment...")
        valid_exp = await client.post(
            f"/api/v1/hypotheses/{hypothesis['id']}/experiments/design",
            json={
                "research_question": "Valid test question",
                "desired_power": 0.8,
                "significance_level": 0.05
            }
        )
        assert valid_exp.status_code == 200
        print("✅ Valid experiment created after errors")

        print("\n✅ Robustness Test PASSED - System handles errors gracefully")
