"""
Prometheus metrics exporter for RAG evaluation.

This module implements Prometheus metrics export for Grafana visualization
following the TDD methodology.
"""

import logging
from typing import Any

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)


class PrometheusMetricsExporter:
    """Prometheus 메트릭 익스포터"""

    def __init__(self) -> None:
        """익스포터 초기화"""
        self._metrics: dict[str, Any] = {}
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """기본 메트릭 초기화"""
        # Gauge metrics for current values
        self._metrics['rag_faithfulness'] = Gauge(
            'rag_faithfulness',
            'RAG faithfulness score (factual consistency)',
            ['env', 'model']
        )
        self._metrics['rag_answer_relevancy'] = Gauge(
            'rag_answer_relevancy',
            'RAG answer relevancy score',
            ['env', 'model']
        )
        self._metrics['rag_context_precision'] = Gauge(
            'rag_context_precision',
            'RAG context precision score',
            ['env', 'model']
        )
        self._metrics['rag_context_recall'] = Gauge(
            'rag_context_recall',
            'RAG context recall score',
            ['env', 'model']
        )

        # Counter for total evaluations
        self._metrics['rag_evaluations_total'] = Counter(
            'rag_evaluations_total',
            'Total number of RAG evaluations',
            ['env', 'model']
        )

        # Histogram for distribution
        self._metrics['rag_faithfulness_histogram'] = Histogram(
            'rag_faithfulness_histogram',
            'Distribution of faithfulness scores',
            ['env', 'model'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )

    def export_metrics(
        self,
        evaluation_results: dict[str, Any],
        labels: dict[str, str] | None = None
    ) -> None:
        """메트릭 익스포트

        Args:
            evaluation_results: 평가 결과
            labels: 메트릭 레이블 (env, model 등)
        """
        if labels is None:
            labels = {'env': 'production', 'model': 'default'}

        metrics_data = evaluation_results.get('metrics', {})
        mean_metrics = metrics_data.get('mean', {})

        # Update gauge metrics
        for metric_name, value in mean_metrics.items():
            gauge_name = f'rag_{metric_name}'
            if gauge_name in self._metrics:
                self._metrics[gauge_name].labels(**labels).set(value)

        # Increment counter
        self._metrics['rag_evaluations_total'].labels(**labels).inc()

        # Update histogram if we have individual test case results
        test_cases = evaluation_results.get('test_cases', [])
        if test_cases:
            for test_case in test_cases:
                test_metrics = test_case.get('metrics', {})
                if 'faithfulness' in test_metrics:
                    self._metrics['rag_faithfulness_histogram'].labels(**labels).observe(
                        test_metrics['faithfulness']
                    )

        logger.info(f"Exported metrics with labels: {labels}")

    def get_prometheus_format(self) -> str:
        """Prometheus 형식으로 메트릭 가져오기

        Returns:
            Prometheus 텍스트 형식 메트릭
        """
        return generate_latest(REGISTRY).decode('utf-8')

    def has_metric(self, metric_name: str) -> bool:
        """메트릭 존재 여부 확인

        Args:
            metric_name: 메트릭 이름

        Returns:
            메트릭 존재 여부
        """
        return metric_name in self._metrics

    def get_metric_value(self, metric_name: str, labels: dict[str, str] | None = None) -> float:
        """메트릭 값 가져오기

        Args:
            metric_name: 메트릭 이름
            labels: 메트릭 레이블

        Returns:
            메트릭 현재 값
        """
        if labels is None:
            labels = {'env': 'production', 'model': 'default'}

        if metric_name not in self._metrics:
            raise ValueError(f"Metric {metric_name} not found")

        metric = self._metrics[metric_name]

        # For Gauge metrics
        if hasattr(metric, '_metrics'):
            # Get the labeled metric
            label_key = tuple(labels.values())
            if label_key in metric._metrics:
                return metric._metrics[label_key]._value._value

        # For Counter metrics
        if isinstance(metric, Counter):
            # Get counter value
            return metric.labels(**labels)._value._value

        return 0.0
