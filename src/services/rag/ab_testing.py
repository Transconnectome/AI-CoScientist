"""
A/B testing framework for RAG operations.

This module implements variant testing and statistical analysis
following the TDD methodology.
"""

import hashlib
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class Variant:
    """A/B 테스트 변형"""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        """변형 초기화

        Args:
            name: 변형 이름
            config: 변형 설정
        """
        self.name = name
        self.config = config

    def __eq__(self, other: object) -> bool:
        """변형 동등성 비교"""
        if not isinstance(other, Variant):
            return False
        return self.name == other.name and self.config == other.config

    def __repr__(self) -> str:
        return f"Variant(name='{self.name}', config={self.config})"


class ABTestConfig:
    """A/B 테스트 설정"""

    def __init__(
        self,
        name: str,
        variants: list[Variant],
        traffic_split: dict[str, float]
    ) -> None:
        """설정 초기화

        Args:
            name: 테스트 이름
            variants: 변형 리스트
            traffic_split: 트래픽 분배 비율
        """
        self.name = name
        self.variants = variants
        self.traffic_split = traffic_split

        # Validate traffic split
        total = sum(traffic_split.values())
        if abs(total - 1.0) > 0.001:
            logger.warning(f"Traffic split does not sum to 1.0: {total}")

    def get_variant(self, name: str) -> Variant | None:
        """이름으로 변형 가져오기

        Args:
            name: 변형 이름

        Returns:
            변형 또는 None
        """
        for variant in self.variants:
            if variant.name == name:
                return variant
        return None


class ABTestResult:
    """A/B 테스트 결과"""

    def __init__(
        self,
        variant_name: str,
        metrics: dict[str, float],
        cost: float
    ) -> None:
        """결과 초기화

        Args:
            variant_name: 변형 이름
            metrics: 메트릭 딕셔너리
            cost: 비용
        """
        self.variant_name = variant_name
        self.metrics = metrics
        self.cost = cost


class ABTest:
    """A/B 테스트 프레임워크"""

    def __init__(self, config: ABTestConfig) -> None:
        """A/B 테스트 초기화

        Args:
            config: 테스트 설정
        """
        self.config = config
        self._results: list[ABTestResult] = []

    def assign_variant(self, user_id: str) -> str:
        """사용자에게 변형 할당

        Args:
            user_id: 사용자 ID

        Returns:
            할당된 변형 이름
        """
        # Deterministic hash-based assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        random_value = (hash_value % 1000) / 1000.0

        # Assign based on traffic split
        cumulative = 0.0
        for variant_name, split in self.config.traffic_split.items():
            cumulative += split
            if random_value < cumulative:
                return variant_name

        # Fallback to first variant
        return list(self.config.traffic_split.keys())[0]

    def add_result(self, result: ABTestResult) -> None:
        """결과 추가

        Args:
            result: 테스트 결과
        """
        self._results.append(result)
        logger.debug(f"Added result for variant {result.variant_name}")

    def get_all_results(self) -> list[ABTestResult]:
        """모든 결과 가져오기

        Returns:
            결과 리스트
        """
        return self._results

    def aggregate_results(self) -> dict[str, dict[str, Any]]:
        """결과 집계

        Returns:
            변형별 집계 결과
        """
        aggregated: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                'count': 0,
                'total_cost': 0.0,
                'metrics': defaultdict(list)
            }
        )

        for result in self._results:
            variant = result.variant_name
            aggregated[variant]['count'] += 1
            aggregated[variant]['total_cost'] += result.cost

            for metric_name, value in result.metrics.items():
                aggregated[variant]['metrics'][metric_name].append(value)

        # Calculate averages
        for variant_data in aggregated.values():
            for metric_name, values in variant_data['metrics'].items():
                variant_data['metrics'][metric_name] = {
                    'mean': sum(values) / len(values) if values else 0.0,
                    'count': len(values),
                    'values': values
                }

        return dict(aggregated)

    def analyze_results(self, metric_name: str) -> dict[str, Any]:
        """결과 통계 분석

        Args:
            metric_name: 분석할 메트릭 이름

        Returns:
            분석 결과
        """
        aggregated = self.aggregate_results()

        # Extract metric values for each variant
        variant_values = {}
        for variant, data in aggregated.items():
            if metric_name in data['metrics']:
                variant_values[variant] = data['metrics'][metric_name]['values']

        if len(variant_values) < 2:
            return {
                'winner': None,
                'confidence': 0.0,
                'p_value': 1.0,
                'message': 'Insufficient data for comparison'
            }

        # Simple comparison: highest mean wins
        variant_means = {
            v: sum(values) / len(values)
            for v, values in variant_values.items()
        }

        winner = max(variant_means, key=variant_means.get)  # type: ignore
        winner_mean = variant_means[winner]

        # Simple confidence calculation (sample size based)
        min_samples = min(len(v) for v in variant_values.values())
        confidence = min(0.99, min_samples / 100.0)

        # Simplified p-value (not real statistical test)
        other_means = [m for v, m in variant_means.items() if v != winner]
        if other_means:
            max_other = max(other_means)
            difference = winner_mean - max_other
            p_value = max(0.01, 1.0 - confidence) if difference > 0.05 else 0.5
        else:
            p_value = 1.0

        return {
            'winner': winner,
            'confidence': confidence,
            'p_value': p_value,
            'variant_means': variant_means
        }

    def analyze_cost_effectiveness(self, metric_name: str) -> dict[str, dict[str, float]]:
        """비용 효율성 분석

        Args:
            metric_name: 분석할 메트릭 이름

        Returns:
            변형별 비용 효율성
        """
        aggregated = self.aggregate_results()
        cost_effectiveness = {}

        for variant, data in aggregated.items():
            if metric_name in data['metrics']:
                avg_quality = data['metrics'][metric_name]['mean']
                avg_cost = data['total_cost'] / data['count'] if data['count'] > 0 else 0.0

                quality_per_dollar = avg_quality / avg_cost if avg_cost > 0 else 0.0

                cost_effectiveness[variant] = {
                    'avg_quality': avg_quality,
                    'avg_cost': avg_cost,
                    'quality_per_dollar': quality_per_dollar
                }

        return cost_effectiveness

    def declare_winner(
        self,
        metric_name: str,
        min_confidence: float = 0.95
    ) -> dict[str, Any] | None:
        """승자 선언

        Args:
            metric_name: 평가 메트릭
            min_confidence: 최소 신뢰도

        Returns:
            승자 정보 또는 None
        """
        analysis = self.analyze_results(metric_name)

        if analysis['confidence'] >= min_confidence:
            return {
                'variant': analysis['winner'],
                'confidence': analysis['confidence'],
                'metric': metric_name,
                'p_value': analysis['p_value']
            }

        return None

    def export_results(self) -> dict[str, Any]:
        """결과 익스포트

        Returns:
            타임스탬프가 포함된 결과 딕셔너리
        """
        aggregated = self.aggregate_results()

        return {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'name': self.config.name,
                'variants': [v.name for v in self.config.variants],
                'traffic_split': self.config.traffic_split
            },
            'results': aggregated
        }

    def run_experiment(self) -> None:
        """실험 실행 (플레이스홀더)"""
        logger.info(f"Running experiment: {self.config.name}")
