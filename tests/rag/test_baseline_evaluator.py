"""
Tests for baseline RAG evaluation system.

Following TDD methodology (RED → GREEN → REFACTOR):
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Improve code quality
"""

import json
from pathlib import Path

import pytest

from src.services.rag.baseline_evaluator import BaselineEvaluator
from src.services.rag.evaluation_dataset import EvaluationDataset


class TestBaselineEvaluator:
    """베이스라인 평가 시스템 테스트"""

    @pytest.fixture
    def evaluator(self):
        """평가자 인스턴스 생성"""
        return BaselineEvaluator(
            output_dir="tests/fixtures/baseline_results"
        )

    @pytest.fixture
    def sample_dataset(self):
        """샘플 평가 데이터셋"""
        dataset = EvaluationDataset()
        return dataset.create_test_cases(count=5)

    def test_evaluator_initialization(self, evaluator):
        """평가자 초기화 검증"""
        assert evaluator is not None
        assert hasattr(evaluator, 'evaluate_system')
        assert hasattr(evaluator, 'generate_report')
        assert hasattr(evaluator, 'save_results')

    def test_system_evaluation(self, evaluator, sample_dataset):
        """시스템 평가 기능 검증"""
        # Mock RAG system response function
        def mock_rag_system(query: str) -> tuple[str, list[str]]:
            """Mock RAG system that returns answer and contexts"""
            return (
                "This is a test answer based on the query.",
                ["Context 1 from retrieval", "Context 2 from retrieval"]
            )

        results = evaluator.evaluate_system(
            test_cases=sample_dataset[:3],  # Use first 3 cases
            rag_system=mock_rag_system
        )

        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'test_cases' in results
        assert len(results['test_cases']) == 3

    def test_metrics_aggregation(self, evaluator):
        """메트릭 집계 검증"""
        individual_results = [
            {
                'faithfulness': 0.8,
                'answer_relevancy': 0.7,
                'context_precision': 0.9,
                'context_recall': 0.6
            },
            {
                'faithfulness': 0.9,
                'answer_relevancy': 0.8,
                'context_precision': 0.85,
                'context_recall': 0.7
            },
            {
                'faithfulness': 0.85,
                'answer_relevancy': 0.75,
                'context_precision': 0.88,
                'context_recall': 0.65
            }
        ]

        aggregated = evaluator.aggregate_metrics(individual_results)

        assert 'mean' in aggregated
        assert 'std' in aggregated
        assert 'min' in aggregated
        assert 'max' in aggregated

        # Check that all metrics are aggregated
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
            assert metric in aggregated['mean']
            assert metric in aggregated['std']

    def test_report_generation(self, evaluator):
        """리포트 생성 검증"""
        evaluation_results = {
            'metrics': {
                'mean': {
                    'faithfulness': 0.85,
                    'answer_relevancy': 0.75,
                    'context_precision': 0.88,
                    'context_recall': 0.67
                },
                'std': {
                    'faithfulness': 0.05,
                    'answer_relevancy': 0.06,
                    'context_precision': 0.03,
                    'context_recall': 0.05
                },
                'min': {
                    'faithfulness': 0.80,
                    'answer_relevancy': 0.69,
                    'context_precision': 0.85,
                    'context_recall': 0.62
                },
                'max': {
                    'faithfulness': 0.90,
                    'answer_relevancy': 0.81,
                    'context_precision': 0.91,
                    'context_recall': 0.72
                }
            },
            'test_cases': [
                {
                    'id': 'test_1',
                    'metrics': {'faithfulness': 0.8, 'answer_relevancy': 0.7}
                }
            ],
            'timestamp': '2025-10-21T10:00:00',
            'total_cases': 5
        }

        report = evaluator.generate_report(evaluation_results)

        assert isinstance(report, str)
        assert 'faithfulness' in report.lower()
        assert 'answer_relevancy' in report.lower()
        assert '0.85' in report or '85' in report  # Mean faithfulness

    def test_results_persistence(self, evaluator, tmp_path):
        """결과 저장 및 로드 검증"""
        evaluator.output_dir = tmp_path

        results = {
            'metrics': {
                'mean': {'faithfulness': 0.85, 'answer_relevancy': 0.75},
                'std': {'faithfulness': 0.05, 'answer_relevancy': 0.06}
            },
            'test_cases': [{'id': 'test_1', 'metrics': {}}],
            'timestamp': '2025-10-21T10:00:00',
            'total_cases': 1
        }

        # Save results
        file_path = evaluator.save_results(results, "baseline_test")

        # Verify file exists
        assert file_path.exists()
        assert file_path.suffix == '.json'

        # Verify content
        with open(file_path) as f:
            loaded_results = json.load(f)
            assert loaded_results['metrics']['mean']['faithfulness'] == 0.85

    def test_baseline_comparison(self, evaluator):
        """베이스라인 비교 기능 검증"""
        baseline_results = {
            'metrics': {
                'mean': {
                    'faithfulness': 0.80,
                    'answer_relevancy': 0.70,
                    'context_precision': 0.85,
                    'context_recall': 0.65
                }
            }
        }

        new_results = {
            'metrics': {
                'mean': {
                    'faithfulness': 0.85,
                    'answer_relevancy': 0.75,
                    'context_precision': 0.88,
                    'context_recall': 0.67
                }
            }
        }

        comparison = evaluator.compare_to_baseline(new_results, baseline_results)

        assert 'improvements' in comparison
        assert 'regressions' in comparison
        assert 'deltas' in comparison

        # All metrics improved in this case
        assert len(comparison['improvements']) == 4
        assert len(comparison['regressions']) == 0

    def test_automated_evaluation_pipeline(self, evaluator, sample_dataset):
        """자동 평가 파이프라인 검증"""
        def mock_rag_system(query: str) -> tuple[str, list[str]]:
            return (
                "Answer for: " + query,
                ["Context 1", "Context 2"]
            )

        # Run full pipeline
        pipeline_results = evaluator.run_evaluation_pipeline(
            test_cases=sample_dataset[:2],
            rag_system=mock_rag_system,
            save_results=False
        )

        assert 'metrics' in pipeline_results
        assert 'report' in pipeline_results
        assert 'timestamp' in pipeline_results
        assert isinstance(pipeline_results['report'], str)

    def test_error_handling_in_evaluation(self, evaluator, sample_dataset):
        """평가 중 에러 처리 검증"""
        def failing_rag_system(query: str) -> tuple[str, list[str]]:
            raise ValueError("Simulated RAG system failure")

        # Should handle errors gracefully
        with pytest.raises(ValueError):
            evaluator.evaluate_system(
                test_cases=sample_dataset[:1],
                rag_system=failing_rag_system
            )
