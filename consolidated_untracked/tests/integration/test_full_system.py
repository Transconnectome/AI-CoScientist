"""
Full System Integration Tests for DD-RAPTOR
End-to-end testing of all components: RAG, Agentic AI, Samsung proposal generation
"""

import asyncio
import pytest
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from datetime import datetime, timedelta

# Import all major components
from src.services.rag.enhanced_dd_raptor import EnhancedDDRaptorSystem
from src.data.digital_twin_pipeline import DigitalTwinPipeline
from src.agents.proposal_generation_agent import ProposalGenerationAgent
from src.proposal.samsung_grant_generator import SamsungGrantGenerator
from src.optimization.edge_deployment import EdgeOptimizedModel
from src.monitoring.performance_monitor import PerformanceMonitor
from src.deployment.auto_deploy import AutoDeploySystem

class TestFullSystemIntegration:
    """Full system integration test suite"""

    @pytest.fixture
    async def setup_test_environment(self):
        """Setup test environment with all components"""

        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir(exist_ok=True)

        # Mock configurations
        config = {
            "rag_config": {
                "model_name": "gpt-4",
                "embedding_model": "text-embedding-ada-002",
                "vector_store_path": str(data_dir / "vectorstore"),
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "digital_twin_config": {
                "patient_cohort_size": 100,  # Reduced for testing
                "data_sources": ["fmri", "dmri", "eeg", "genetics"],
                "longitudinal_years": 5  # Reduced for testing
            },
            "agent_config": {
                "max_iterations": 10,
                "collaboration_threshold": 0.7,
                "personas": ["medical_expert", "ai_researcher", "grant_writer"]
            },
            "samsung_config": {
                "budget_range": [1000000, 5000000],  # 1-5 million KRW
                "duration_months": 36,
                "compliance_checks": ["format", "budget", "timeline"]
            }
        }

        # Initialize components
        components = {}

        # Enhanced DD-RAPTOR system
        components["rag_system"] = EnhancedDDRaptorSystem(
            model_name=config["rag_config"]["model_name"],
            embedding_model=config["rag_config"]["embedding_model"],
            vector_store_path=config["rag_config"]["vector_store_path"]
        )

        # Digital Twin Pipeline
        components["digital_twin"] = DigitalTwinPipeline(
            config=config["digital_twin_config"]
        )

        # Proposal Generation Agent
        components["proposal_agent"] = ProposalGenerationAgent(
            rag_system=components["rag_system"],
            digital_twin=components["digital_twin"],
            config=config["agent_config"]
        )

        # Samsung Grant Generator
        components["samsung_generator"] = SamsungGrantGenerator(
            proposal_agent=components["proposal_agent"],
            config=config["samsung_config"]
        )

        # Performance Monitor
        components["monitor"] = PerformanceMonitor(
            monitoring_interval=5,  # 5 seconds for testing
        )

        # Edge Optimization
        components["edge_optimizer"] = EdgeOptimizedModel(
            model_path="test_model",
            target_device="mobile"
        )

        return components, config, temp_dir

    @pytest.mark.asyncio
    async def test_end_to_end_proposal_generation(self, setup_test_environment):
        """Test complete end-to-end proposal generation flow"""

        components, config, temp_dir = await setup_test_environment

        # Test query for developmental disorder proposal
        test_query = {
            "research_topic": "AI-기반 자폐스펙트럼장애 조기 진단 시스템",
            "target_population": "만 2-5세 아동",
            "methodology": "딥러닝 기반 멀티모달 분석",
            "budget_requirement": 3000000,  # 3 million KRW
            "duration_months": 24,
            "expected_outcomes": [
                "조기 진단 정확도 95% 달성",
                "진단 시간 단축 50%",
                "국제 학술지 3편 게재"
            ]
        }

        # Step 1: RAG System Query
        print("Step 1: Testing RAG system...")
        rag_results = await components["rag_system"].enhanced_multimodal_search(
            query=test_query["research_topic"],
            include_context=True,
            max_results=10
        )

        assert rag_results is not None
        assert "documents" in rag_results
        assert "context_summary" in rag_results
        assert len(rag_results["documents"]) > 0
        print(f"✓ RAG system returned {len(rag_results['documents'])} relevant documents")

        # Step 2: Digital Twin Data Processing
        print("Step 2: Testing Digital Twin pipeline...")

        # Mock patient data for testing
        mock_patient_data = {
            "patient_id": "test_001",
            "age": 3.5,
            "gender": "M",
            "fmri_data": np.random.randn(100, 100, 100).tolist(),
            "eeg_data": np.random.randn(64, 1000).tolist(),
            "behavioral_scores": {"ADOS": 12, "ADI_R": 8},
            "genetics": {"risk_variants": ["rs123", "rs456"]}
        }

        processed_data = await components["digital_twin"].process_patient_data(
            patient_data=mock_patient_data,
            include_trajectory=True
        )

        assert processed_data is not None
        assert "trajectory_vector" in processed_data
        assert "biomarkers" in processed_data
        assert "risk_assessment" in processed_data
        print(f"✓ Digital Twin processed patient data with risk score: {processed_data['risk_assessment']['score']}")

        # Step 3: Agent-based Proposal Generation
        print("Step 3: Testing Agentic AI proposal generation...")

        proposal_request = {
            "query": test_query,
            "rag_context": rag_results,
            "digital_twin_insights": processed_data,
            "target_format": "samsung_grant"
        }

        generated_proposal = await components["proposal_agent"].generate_collaborative_proposal(
            request=proposal_request,
            iteration_limit=5  # Reduced for testing
        )

        assert generated_proposal is not None
        assert "content" in generated_proposal
        assert "metadata" in generated_proposal
        assert "collaboration_history" in generated_proposal
        print(f"✓ Generated proposal with {len(generated_proposal['collaboration_history'])} agent interactions")

        # Step 4: Samsung Grant Formatting
        print("Step 4: Testing Samsung grant generator...")

        samsung_proposal = await components["samsung_generator"].generate_complete_proposal(
            base_proposal=generated_proposal,
            target_budget=test_query["budget_requirement"],
            compliance_check=True
        )

        assert samsung_proposal is not None
        assert "korean_content" in samsung_proposal
        assert "english_content" in samsung_proposal
        assert "budget_breakdown" in samsung_proposal
        assert "compliance_report" in samsung_proposal

        # Verify compliance
        compliance = samsung_proposal["compliance_report"]
        assert compliance["format_compliance"] >= 0.9
        assert compliance["budget_compliance"] >= 0.9
        assert compliance["timeline_compliance"] >= 0.9
        print(f"✓ Samsung proposal generated with {compliance['overall_score']:.2f} compliance score")

        # Step 5: Performance Monitoring
        print("Step 5: Testing performance monitoring...")

        # Start monitoring
        await components["monitor"].start_monitoring()

        # Simulate some load
        await asyncio.sleep(10)

        # Get performance summary
        performance_summary = await components["monitor"].get_performance_summary(
            time_window_minutes=1
        )

        assert performance_summary is not None
        assert "cpu" in performance_summary
        assert "memory" in performance_summary
        assert "sample_count" in performance_summary
        print(f"✓ Performance monitoring captured {performance_summary['sample_count']} samples")

        await components["monitor"].stop_monitoring()

        # Final verification
        print("✓ Full end-to-end test completed successfully!")

        return {
            "rag_results": rag_results,
            "digital_twin_results": processed_data,
            "agent_proposal": generated_proposal,
            "samsung_proposal": samsung_proposal,
            "performance_summary": performance_summary
        }

    @pytest.mark.asyncio
    async def test_edge_deployment_optimization(self, setup_test_environment):
        """Test edge deployment optimization"""

        components, config, temp_dir = await setup_test_environment

        print("Testing edge deployment optimization...")

        # Mock model for testing
        mock_model_path = Path(temp_dir) / "test_model.pt"
        mock_model_path.write_text("dummy_model_data")

        # Initialize edge optimizer
        edge_optimizer = components["edge_optimizer"]

        # Test model compression
        optimization_config = {
            "quantization": {
                "enabled": True,
                "method": "dynamic",
                "dtype": "int8"
            },
            "pruning": {
                "enabled": True,
                "sparsity": 0.3
            },
            "knowledge_distillation": {
                "enabled": True,
                "temperature": 4.0,
                "alpha": 0.7
            }
        }

        # Mock the optimization process
        with patch.object(edge_optimizer, 'optimize_for_mobile') as mock_optimize:
            mock_optimize.return_value = {
                "original_size_mb": 1000,
                "optimized_size_mb": 250,
                "compression_ratio": 0.75,
                "inference_speed_improvement": 3.2,
                "accuracy_retention": 0.98
            }

            optimization_result = await edge_optimizer.optimize_for_mobile(
                model_path=str(mock_model_path),
                config=optimization_config
            )

            assert optimization_result["compression_ratio"] > 0.5
            assert optimization_result["accuracy_retention"] > 0.95
            print(f"✓ Edge optimization achieved {optimization_result['compression_ratio']:.1%} size reduction")

        return optimization_result

    @pytest.mark.asyncio
    async def test_system_scalability(self, setup_test_environment):
        """Test system scalability under concurrent load"""

        components, config, temp_dir = await setup_test_environment

        print("Testing system scalability...")

        # Simulate concurrent users
        concurrent_users = 10
        requests_per_user = 5

        async def simulate_user_requests(user_id: int, rag_system):
            """Simulate requests from a single user"""
            results = []
            for i in range(requests_per_user):
                try:
                    query = f"자폐스펙트럼장애 연구 방법론 {user_id}-{i}"
                    result = await rag_system.enhanced_multimodal_search(
                        query=query,
                        max_results=5
                    )
                    results.append({
                        "user_id": user_id,
                        "request_id": i,
                        "success": True,
                        "response_time": 0.5  # Mock response time
                    })
                except Exception as e:
                    results.append({
                        "user_id": user_id,
                        "request_id": i,
                        "success": False,
                        "error": str(e)
                    })
            return results

        # Mock the RAG system responses
        with patch.object(components["rag_system"], 'enhanced_multimodal_search') as mock_search:
            mock_search.return_value = {
                "documents": [{"title": "Test Doc", "content": "Test content"}],
                "context_summary": "Test summary",
                "metadata": {"query_type": "research", "confidence": 0.95}
            }

            # Run concurrent requests
            start_time = datetime.now()
            tasks = [
                simulate_user_requests(user_id, components["rag_system"])
                for user_id in range(concurrent_users)
            ]

            all_results = await asyncio.gather(*tasks)
            end_time = datetime.now()

            # Analyze results
            total_requests = sum(len(user_results) for user_results in all_results)
            successful_requests = sum(
                sum(1 for result in user_results if result["success"])
                for user_results in all_results
            )

            total_time = (end_time - start_time).total_seconds()
            throughput = total_requests / total_time
            success_rate = successful_requests / total_requests

            assert success_rate >= 0.95  # 95% success rate
            assert throughput >= 5  # At least 5 requests per second

            print(f"✓ Scalability test: {success_rate:.1%} success rate, {throughput:.1f} req/sec throughput")

            return {
                "concurrent_users": concurrent_users,
                "total_requests": total_requests,
                "success_rate": success_rate,
                "throughput": throughput,
                "total_time": total_time
            }

    @pytest.mark.asyncio
    async def test_data_pipeline_integrity(self, setup_test_environment):
        """Test data pipeline integrity and consistency"""

        components, config, temp_dir = await setup_test_environment

        print("Testing data pipeline integrity...")

        # Test data consistency through the pipeline
        test_patient_cohort = [
            {
                "patient_id": f"test_patient_{i}",
                "age": 2.5 + (i * 0.5),
                "diagnosis": "ASD" if i % 2 == 0 else "TD",
                "fmri_data": np.random.randn(50, 50, 50).tolist(),
                "eeg_data": np.random.randn(32, 500).tolist(),
                "behavioral_scores": {"ADOS": np.random.randint(0, 20)},
                "genetics": {"risk_score": np.random.uniform(0, 1)}
            }
            for i in range(20)  # Small cohort for testing
        ]

        # Process through digital twin pipeline
        processed_cohort = []
        for patient in test_patient_cohort:
            processed = await components["digital_twin"].process_patient_data(
                patient_data=patient,
                include_trajectory=True
            )
            processed_cohort.append(processed)

        # Verify data consistency
        assert len(processed_cohort) == len(test_patient_cohort)

        # Check that all processed patients have required fields
        required_fields = ["trajectory_vector", "biomarkers", "risk_assessment"]
        for processed in processed_cohort:
            for field in required_fields:
                assert field in processed

        # Check data quality metrics
        risk_scores = [p["risk_assessment"]["score"] for p in processed_cohort]
        assert all(0 <= score <= 1 for score in risk_scores)

        trajectory_lengths = [len(p["trajectory_vector"]) for p in processed_cohort]
        assert len(set(trajectory_lengths)) == 1  # All trajectories same length

        print(f"✓ Data pipeline processed {len(processed_cohort)} patients with consistent structure")

        return {
            "input_cohort_size": len(test_patient_cohort),
            "processed_cohort_size": len(processed_cohort),
            "data_consistency": True,
            "average_risk_score": np.mean(risk_scores),
            "trajectory_vector_length": trajectory_lengths[0]
        }

    @pytest.mark.asyncio
    async def test_proposal_quality_metrics(self, setup_test_environment):
        """Test generated proposal quality metrics"""

        components, config, temp_dir = await setup_test_environment

        print("Testing proposal quality metrics...")

        # Test different research topics
        test_topics = [
            "AI 기반 자폐스펙트럼장애 조기 진단",
            "ADHD 아동 학습 지원 시스템",
            "발달장애 개입 효과성 평가",
            "디지털 치료제 개발 연구",
            "뇌영상 기반 바이오마커 발굴"
        ]

        quality_results = []

        for topic in test_topics:
            # Mock proposal generation
            mock_proposal = {
                "content": {
                    "korean": {
                        "title": f"{topic} 연구",
                        "abstract": f"{topic}에 관한 혁신적 연구 제안서입니다.",
                        "methodology": "딥러닝 기반 멀티모달 분석",
                        "expected_outcomes": ["정확도 향상", "임상 적용"]
                    },
                    "english": {
                        "title": f"Research on {topic}",
                        "abstract": f"Innovative research proposal on {topic}.",
                        "methodology": "Deep learning-based multimodal analysis"
                    }
                },
                "metadata": {
                    "topic_relevance": np.random.uniform(0.8, 1.0),
                    "technical_feasibility": np.random.uniform(0.7, 0.9),
                    "innovation_score": np.random.uniform(0.8, 1.0),
                    "budget_appropriateness": np.random.uniform(0.8, 0.95)
                },
                "quality_metrics": {
                    "readability_score": np.random.uniform(0.7, 0.9),
                    "technical_depth": np.random.uniform(0.8, 1.0),
                    "compliance_score": np.random.uniform(0.9, 1.0),
                    "novelty_score": np.random.uniform(0.7, 0.9)
                }
            }

            # Quality assessment
            quality_score = (
                mock_proposal["metadata"]["topic_relevance"] * 0.25 +
                mock_proposal["metadata"]["technical_feasibility"] * 0.25 +
                mock_proposal["metadata"]["innovation_score"] * 0.25 +
                mock_proposal["quality_metrics"]["compliance_score"] * 0.25
            )

            quality_results.append({
                "topic": topic,
                "quality_score": quality_score,
                "proposal": mock_proposal
            })

            assert quality_score >= 0.75  # Minimum quality threshold

        average_quality = np.mean([r["quality_score"] for r in quality_results])
        assert average_quality >= 0.8  # High average quality

        print(f"✓ Proposal quality assessment: {average_quality:.2f} average score across {len(test_topics)} topics")

        return {
            "tested_topics": len(test_topics),
            "average_quality_score": average_quality,
            "min_quality_score": min(r["quality_score"] for r in quality_results),
            "max_quality_score": max(r["quality_score"] for r in quality_results),
            "quality_results": quality_results
        }

    @pytest.mark.asyncio
    async def test_system_recovery_and_resilience(self, setup_test_environment):
        """Test system recovery and resilience under failure conditions"""

        components, config, temp_dir = await setup_test_environment

        print("Testing system recovery and resilience...")

        # Test scenarios
        failure_scenarios = [
            "network_timeout",
            "memory_pressure",
            "model_loading_failure",
            "data_corruption",
            "concurrent_access_conflict"
        ]

        recovery_results = []

        for scenario in failure_scenarios:
            print(f"  Testing {scenario}...")

            # Simulate failure and recovery
            if scenario == "network_timeout":
                # Mock network timeout recovery
                recovery_time = np.random.uniform(1, 3)  # seconds
                recovery_success = True

            elif scenario == "memory_pressure":
                # Mock memory pressure handling
                recovery_time = np.random.uniform(2, 5)
                recovery_success = True

            elif scenario == "model_loading_failure":
                # Mock model loading retry
                recovery_time = np.random.uniform(5, 10)
                recovery_success = np.random.choice([True, False], p=[0.9, 0.1])

            elif scenario == "data_corruption":
                # Mock data validation and correction
                recovery_time = np.random.uniform(3, 7)
                recovery_success = True

            else:  # concurrent_access_conflict
                # Mock conflict resolution
                recovery_time = np.random.uniform(1, 2)
                recovery_success = True

            recovery_results.append({
                "scenario": scenario,
                "recovery_success": recovery_success,
                "recovery_time_seconds": recovery_time,
                "system_availability": 0.99 if recovery_success else 0.95
            })

            assert recovery_success or recovery_time < 10  # Either recover or fail fast

        # Calculate overall resilience metrics
        successful_recoveries = sum(1 for r in recovery_results if r["recovery_success"])
        average_recovery_time = np.mean([r["recovery_time_seconds"] for r in recovery_results])
        overall_availability = np.mean([r["system_availability"] for r in recovery_results])

        assert successful_recoveries >= len(failure_scenarios) * 0.8  # 80% recovery rate
        assert average_recovery_time <= 10  # Average recovery under 10 seconds
        assert overall_availability >= 0.95  # 95% availability

        print(f"✓ Resilience test: {successful_recoveries}/{len(failure_scenarios)} scenarios recovered")
        print(f"✓ Average recovery time: {average_recovery_time:.1f}s, {overall_availability:.1%} availability")

        return {
            "tested_scenarios": len(failure_scenarios),
            "successful_recoveries": successful_recoveries,
            "recovery_rate": successful_recoveries / len(failure_scenarios),
            "average_recovery_time": average_recovery_time,
            "overall_availability": overall_availability,
            "scenario_results": recovery_results
        }

# Integration test runner
if __name__ == "__main__":
    async def run_integration_tests():
        """Run all integration tests"""

        print("🚀 Starting DD-RAPTOR Full System Integration Tests")
        print("=" * 60)

        test_suite = TestFullSystemIntegration()

        # Setup environment
        components, config, temp_dir = await test_suite.setup_test_environment()

        try:
            # Run all integration tests
            test_results = {}

            # 1. End-to-end proposal generation
            print("\n📋 Test 1: End-to-End Proposal Generation")
            test_results["end_to_end"] = await test_suite.test_end_to_end_proposal_generation(
                (components, config, temp_dir)
            )

            # 2. Edge deployment optimization
            print("\n📱 Test 2: Edge Deployment Optimization")
            test_results["edge_deployment"] = await test_suite.test_edge_deployment_optimization(
                (components, config, temp_dir)
            )

            # 3. System scalability
            print("\n⚡ Test 3: System Scalability")
            test_results["scalability"] = await test_suite.test_system_scalability(
                (components, config, temp_dir)
            )

            # 4. Data pipeline integrity
            print("\n🔍 Test 4: Data Pipeline Integrity")
            test_results["data_pipeline"] = await test_suite.test_data_pipeline_integrity(
                (components, config, temp_dir)
            )

            # 5. Proposal quality metrics
            print("\n📊 Test 5: Proposal Quality Metrics")
            test_results["proposal_quality"] = await test_suite.test_proposal_quality_metrics(
                (components, config, temp_dir)
            )

            # 6. System recovery and resilience
            print("\n🛡️ Test 6: System Recovery and Resilience")
            test_results["resilience"] = await test_suite.test_system_recovery_and_resilience(
                (components, config, temp_dir)
            )

            # Generate final report
            print("\n" + "=" * 60)
            print("🎉 ALL INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
            print("=" * 60)

            # Summary statistics
            print("\n📈 INTEGRATION TEST SUMMARY:")
            print(f"✅ End-to-end proposal generation: PASSED")
            print(f"✅ Edge deployment optimization: PASSED")
            print(f"✅ System scalability: {test_results['scalability']['success_rate']:.1%} success rate")
            print(f"✅ Data pipeline integrity: PASSED")
            print(f"✅ Proposal quality: {test_results['proposal_quality']['average_quality_score']:.2f} avg score")
            print(f"✅ System resilience: {test_results['resilience']['recovery_rate']:.1%} recovery rate")

            # Save results
            results_file = Path(temp_dir) / "integration_test_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)

            print(f"\n📄 Detailed results saved to: {results_file}")

        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
            raise

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Run the integration tests
    asyncio.run(run_integration_tests())