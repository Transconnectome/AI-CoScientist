"""
Performance tracking system for RAG operations.

This module implements latency and token usage tracking
following the TDD methodology.
"""

import logging
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Pricing per 1K tokens (as of 2024)
MODEL_PRICING = {
    'gpt-4': {'prompt': 0.03, 'completion': 0.06},
    'gpt-4-turbo': {'prompt': 0.01, 'completion': 0.03},
    'gpt-3.5-turbo': {'prompt': 0.0005, 'completion': 0.0015},
    'claude-3-opus': {'prompt': 0.015, 'completion': 0.075},
    'claude-3-sonnet': {'prompt': 0.003, 'completion': 0.015},
}


class PerformanceTracker:
    """RAG 성능 추적 시스템"""

    def __init__(self) -> None:
        """트래커 초기화"""
        self._latency_data: dict[str, list[float]] = defaultdict(list)
        self._token_data: dict[str, dict[str, int]] = defaultdict(
            lambda: {'prompt_tokens': 0, 'completion_tokens': 0}
        )
        self._lock = threading.Lock()

    @contextmanager
    def track_latency(self, operation: str) -> Any:
        """레이턴시 추적 컨텍스트 매니저

        Args:
            operation: 작업 이름

        Example:
            with tracker.track_latency('retrieval'):
                # ... retrieval operation
                pass
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self._record_latency(operation, elapsed)

    def _record_latency(self, operation: str, latency: float) -> None:
        """레이턴시 기록 (thread-safe)

        Args:
            operation: 작업 이름
            latency: 레이턴시 (초)
        """
        with self._lock:
            self._latency_data[operation].append(latency)
            logger.debug(f"Recorded latency for {operation}: {latency:.3f}s")

    def track_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> None:
        """토큰 사용량 추적

        Args:
            model: 모델 이름
            prompt_tokens: 프롬프트 토큰 수
            completion_tokens: 완성 토큰 수
        """
        with self._lock:
            self._token_data[model]['prompt_tokens'] += prompt_tokens
            self._token_data[model]['completion_tokens'] += completion_tokens
            logger.debug(
                f"Tracked tokens for {model}: "
                f"prompt={prompt_tokens}, completion={completion_tokens}"
            )

    def get_metrics(self) -> dict[str, Any]:
        """메트릭 가져오기

        Returns:
            딕셔너리 형태의 메트릭
            {
                'latency': {
                    'operation_name': {
                        'count': int,
                        'mean': float,
                        'min': float,
                        'max': float,
                        'p50': float,
                        'p95': float,
                        'p99': float
                    }
                },
                'tokens': {
                    'model_name': {
                        'prompt_tokens': int,
                        'completion_tokens': int,
                        'total_tokens': int,
                        'estimated_cost': float
                    }
                }
            }
        """
        with self._lock:
            metrics: dict[str, Any] = {
                'latency': {},
                'tokens': {}
            }

            # Process latency data
            for operation, latencies in self._latency_data.items():
                if latencies:
                    latency_array = np.array(latencies)
                    metrics['latency'][operation] = {
                        'count': len(latencies),
                        'mean': float(np.mean(latency_array)),
                        'min': float(np.min(latency_array)),
                        'max': float(np.max(latency_array)),
                        'p50': float(np.percentile(latency_array, 50)),
                        'p95': float(np.percentile(latency_array, 95)),
                        'p99': float(np.percentile(latency_array, 99))
                    }

            # Process token data
            for model, token_counts in self._token_data.items():
                total_tokens = (
                    token_counts['prompt_tokens'] +
                    token_counts['completion_tokens']
                )

                # Calculate cost
                cost = self._calculate_cost(
                    model,
                    token_counts['prompt_tokens'],
                    token_counts['completion_tokens']
                )

                metrics['tokens'][model] = {
                    'prompt_tokens': token_counts['prompt_tokens'],
                    'completion_tokens': token_counts['completion_tokens'],
                    'total_tokens': total_tokens,
                    'estimated_cost': cost
                }

            return metrics

    def _calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """비용 계산

        Args:
            model: 모델 이름
            prompt_tokens: 프롬프트 토큰 수
            completion_tokens: 완성 토큰 수

        Returns:
            예상 비용 (USD)
        """
        if model not in MODEL_PRICING:
            logger.warning(f"Unknown model {model}, cost estimation unavailable")
            return 0.0

        pricing = MODEL_PRICING[model]
        prompt_cost = (prompt_tokens / 1000) * pricing['prompt']
        completion_cost = (completion_tokens / 1000) * pricing['completion']

        return prompt_cost + completion_cost

    def reset_metrics(self) -> None:
        """메트릭 리셋"""
        with self._lock:
            self._latency_data.clear()
            self._token_data.clear()
            logger.info("Metrics reset")

    def export_metrics(self) -> dict[str, Any]:
        """메트릭 익스포트

        Returns:
            타임스탬프가 포함된 메트릭 딕셔너리
        """
        metrics = self.get_metrics()
        return {
            'timestamp': datetime.now().isoformat(),
            'latency': metrics['latency'],
            'tokens': metrics['tokens']
        }
