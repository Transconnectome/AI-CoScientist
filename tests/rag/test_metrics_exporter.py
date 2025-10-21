"""
Tests for Prometheus metrics exporter.

Following TDD methodology (RED → GREEN → REFACTOR):
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Improve code quality
"""

import pytest

from src.services.rag.metrics_exporter import PrometheusMetricsExporter


class TestPrometheusMetricsExporter:
    """Prometheus 메트릭 익스포터 테스트"""

    @pytest.fixture
    def exporter(self):
        """익스포터 인스턴스 생성"""
        return PrometheusMetricsExporter()

    def test_exporter_initialization(self, exporter):
        """익스포터 초기화 검증"""
        assert exporter is not None
        assert hasattr(exporter, 'export_metrics')
        assert hasattr(exporter, 'get_prometheus_format')

    def test_metric_registration(self, exporter):
        """메트릭 등록 검증"""
        evaluation_results = {
            'metrics': {
                'mean': {
                    'faithfulness': 0.85,
                    'answer_relevancy': 0.75,
                    'context_precision': 0.88,
                    'context_recall': 0.67
                }
            },
            'timestamp': '2025-10-21T10:00:00'
        }

        exporter.export_metrics(evaluation_results)

        # Check metrics were registered
        assert exporter.has_metric('rag_faithfulness')
        assert exporter.has_metric('rag_answer_relevancy')
        assert exporter.has_metric('rag_context_precision')
        assert exporter.has_metric('rag_context_recall')

    def test_prometheus_format_output(self, exporter):
        """Prometheus 형식 출력 검증"""
        evaluation_results = {
            'metrics': {
                'mean': {
                    'faithfulness': 0.85,
                    'answer_relevancy': 0.75
                }
            },
            'timestamp': '2025-10-21T10:00:00'
        }

        exporter.export_metrics(evaluation_results)
        prometheus_text = exporter.get_prometheus_format()

        # Check format
        assert isinstance(prometheus_text, str)
        assert 'rag_faithfulness' in prometheus_text
        assert 'rag_answer_relevancy' in prometheus_text
        assert '0.85' in prometheus_text
        assert '0.75' in prometheus_text

    def test_metric_labels(self, exporter):
        """메트릭 레이블 검증"""
        evaluation_results = {
            'metrics': {
                'mean': {
                    'faithfulness': 0.85
                }
            },
            'timestamp': '2025-10-21T10:00:00',
            'environment': 'test',
            'model': 'gpt-4'
        }

        exporter.export_metrics(evaluation_results, labels={'env': 'test', 'model': 'gpt-4'})
        prometheus_text = exporter.get_prometheus_format()

        # Check labels in output
        assert 'env="test"' in prometheus_text
        assert 'model="gpt-4"' in prometheus_text

    def test_histogram_metrics(self, exporter):
        """히스토그램 메트릭 검증"""
        evaluation_results = {
            'metrics': {
                'mean': {'faithfulness': 0.85},
                'std': {'faithfulness': 0.05},
                'min': {'faithfulness': 0.75},
                'max': {'faithfulness': 0.95}
            },
            'test_cases': [
                {'metrics': {'faithfulness': 0.80}},
                {'metrics': {'faithfulness': 0.85}},
                {'metrics': {'faithfulness': 0.90}}
            ]
        }

        exporter.export_metrics(evaluation_results)

        # Check histogram was created
        assert exporter.has_metric('rag_faithfulness_histogram')

    def test_metric_update(self, exporter):
        """메트릭 업데이트 검증"""
        results1 = {
            'metrics': {'mean': {'faithfulness': 0.80}},
            'timestamp': '2025-10-21T10:00:00'
        }
        results2 = {
            'metrics': {'mean': {'faithfulness': 0.85}},
            'timestamp': '2025-10-21T11:00:00'
        }

        exporter.export_metrics(results1)
        value1 = exporter.get_metric_value('rag_faithfulness')

        exporter.export_metrics(results2)
        value2 = exporter.get_metric_value('rag_faithfulness')

        # Value should be updated
        assert value2 == 0.85
        assert value2 > value1

    def test_counter_metrics(self, exporter):
        """카운터 메트릭 검증"""
        for i in range(3):
            results = {
                'metrics': {'mean': {'faithfulness': 0.85}},
                'total_cases': 10
            }
            exporter.export_metrics(results)

        # Check evaluation counter
        counter_value = exporter.get_metric_value('rag_evaluations_total')
        assert counter_value == 3
