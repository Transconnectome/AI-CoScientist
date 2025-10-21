"""
Cost optimization system for RAG operations.

This module implements budget tracking and cost optimization
following the TDD methodology.
"""

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# Model pricing from performance_tracker
MODEL_PRICING = {
    'gpt-4': {'prompt': 0.03, 'completion': 0.06},
    'gpt-4-turbo': {'prompt': 0.01, 'completion': 0.03},
    'gpt-3.5-turbo': {'prompt': 0.0005, 'completion': 0.0015},
    'claude-3-opus': {'prompt': 0.015, 'completion': 0.075},
    'claude-3-sonnet': {'prompt': 0.003, 'completion': 0.015},
}


class CostBudget:
    """비용 예산 관리"""

    def __init__(
        self,
        total: float,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95
    ) -> None:
        """예산 초기화

        Args:
            total: 총 예산
            warning_threshold: 경고 임계값 (0-1)
            critical_threshold: 위험 임계값 (0-1)
        """
        self.total = total
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.spent = 0.0
        self._expenses: dict[str, float] = defaultdict(float)

    @property
    def remaining(self) -> float:
        """남은 예산"""
        return self.total - self.spent

    def add_expense(self, amount: float, category: str) -> None:
        """비용 추가

        Args:
            amount: 비용 금액
            category: 비용 카테고리
        """
        self.spent += amount
        self._expenses[category] += amount
        logger.debug(f"Added expense: {category}=${amount:.4f}, total=${self.spent:.4f}")

    def get_status(self) -> str:
        """예산 상태 반환

        Returns:
            'normal', 'warning', 'critical'
        """
        usage_ratio = self.spent / self.total if self.total > 0 else 0

        if usage_ratio >= self.critical_threshold:
            return 'critical'
        elif usage_ratio >= self.warning_threshold:
            return 'warning'
        else:
            return 'normal'

    def is_exceeded(self) -> bool:
        """예산 초과 여부"""
        return self.spent > self.total

    def get_breakdown(self) -> dict[str, float]:
        """카테고리별 비용 분석

        Returns:
            카테고리별 비용 딕셔너리
        """
        return dict(self._expenses)


class CostOptimizer:
    """RAG 비용 최적화 시스템"""

    def __init__(self) -> None:
        """옵티마이저 초기화"""
        self._model_pricing = MODEL_PRICING

    def analyze_costs(self, usage_data: dict[str, Any]) -> dict[str, Any]:
        """비용 분석

        Args:
            usage_data: 사용량 데이터

        Returns:
            분석 결과
        """
        analysis = {
            'total_cost': 0.0,
            'by_model': {},
            'by_operation': {}
        }

        # Model-based cost analysis
        if 'tokens' in usage_data:
            for model, tokens in usage_data['tokens'].items():
                cost = tokens.get('estimated_cost', 0.0)
                analysis['total_cost'] += cost
                analysis['by_model'][model] = cost

        return analysis

    def compare_model_costs(self, usage: dict[str, dict[str, int]]) -> dict[str, float]:
        """모델별 비용 비교

        Args:
            usage: 모델별 토큰 사용량

        Returns:
            모델별 예상 비용
        """
        costs = {}

        for model, tokens in usage.items():
            if model in self._model_pricing:
                pricing = self._model_pricing[model]
                prompt_cost = (tokens['prompt_tokens'] / 1000) * pricing['prompt']
                completion_cost = (tokens['completion_tokens'] / 1000) * pricing['completion']
                costs[model] = prompt_cost + completion_cost
            else:
                costs[model] = 0.0

        return costs

    def suggest_optimizations(self, usage_data: dict[str, Any]) -> list[dict[str, Any]]:
        """최적화 제안 생성

        Args:
            usage_data: 사용량 데이터

        Returns:
            최적화 제안 리스트
        """
        suggestions = []

        # Check if using expensive models
        if 'tokens' in usage_data:
            for model, tokens in usage_data['tokens'].items():
                cost = tokens.get('estimated_cost', 0.0)

                if model == 'gpt-4' and cost > 0.5:
                    # Suggest switching to cheaper model
                    savings = self.estimate_savings(
                        {
                            'model': model,
                            'prompt_tokens': tokens['prompt_tokens'],
                            'completion_tokens': tokens['completion_tokens']
                        },
                        target_model='gpt-3.5-turbo'
                    )

                    suggestions.append({
                        'description': f'Consider switching from {model} to gpt-3.5-turbo for non-critical tasks',
                        'potential_savings': savings,
                        'priority': 'high'
                    })

        # Always suggest caching
        suggestions.append({
            'description': 'Implement response caching to reduce redundant API calls',
            'potential_savings': 0.0,  # Will be estimated separately
            'priority': 'medium'
        })

        return suggestions

    def estimate_savings(
        self,
        current_usage: dict[str, Any],
        target_model: str
    ) -> float:
        """비용 절감 추정

        Args:
            current_usage: 현재 사용량
            target_model: 목표 모델

        Returns:
            예상 절감액
        """
        current_model = current_usage['model']

        if current_model not in self._model_pricing or target_model not in self._model_pricing:
            return 0.0

        # Calculate current cost
        current_pricing = self._model_pricing[current_model]
        current_cost = (
            (current_usage['prompt_tokens'] / 1000) * current_pricing['prompt'] +
            (current_usage['completion_tokens'] / 1000) * current_pricing['completion']
        )

        # Calculate target cost
        target_pricing = self._model_pricing[target_model]
        target_cost = (
            (current_usage['prompt_tokens'] / 1000) * target_pricing['prompt'] +
            (current_usage['completion_tokens'] / 1000) * target_pricing['completion']
        )

        return current_cost - target_cost

    def estimate_cache_savings(
        self,
        total_requests: int,
        cache_hit_rate: float,
        avg_cost_per_request: float
    ) -> float:
        """캐시 히트율 기반 절감 추정

        Args:
            total_requests: 총 요청 수
            cache_hit_rate: 캐시 히트율 (0-1)
            avg_cost_per_request: 요청당 평균 비용

        Returns:
            예상 절감액
        """
        cache_hits = total_requests * cache_hit_rate
        savings = cache_hits * avg_cost_per_request
        return savings

    def estimate_batch_savings(
        self,
        total_requests: int,
        current_batch_size: int,
        optimal_batch_size: int,
        overhead_per_batch: float
    ) -> float:
        """배치 최적화 절감 추정

        Args:
            total_requests: 총 요청 수
            current_batch_size: 현재 배치 크기
            optimal_batch_size: 최적 배치 크기
            overhead_per_batch: 배치당 오버헤드

        Returns:
            예상 절감액
        """
        current_batches = total_requests / current_batch_size
        optimal_batches = total_requests / optimal_batch_size

        current_overhead = current_batches * overhead_per_batch
        optimal_overhead = optimal_batches * overhead_per_batch

        return current_overhead - optimal_overhead

    def analyze_trend(self, cost_history: list[dict[str, Any]]) -> dict[str, Any]:
        """비용 추세 분석

        Args:
            cost_history: 시간별 비용 이력

        Returns:
            추세 분석 결과
        """
        if len(cost_history) < 2:
            return {'direction': 'stable', 'rate': 0.0}

        # Simple trend: compare first and last
        first_cost = cost_history[0]['cost']
        last_cost = cost_history[-1]['cost']

        if last_cost > first_cost * 1.1:
            direction = 'increasing'
        elif last_cost < first_cost * 0.9:
            direction = 'decreasing'
        else:
            direction = 'stable'

        # Calculate rate of change
        rate = (last_cost - first_cost) / first_cost if first_cost > 0 else 0.0

        return {
            'direction': direction,
            'rate': rate
        }

    def check_budget_alerts(self, budget: CostBudget) -> list[dict[str, Any]]:
        """예산 경고 확인

        Args:
            budget: 예산 객체

        Returns:
            경고 리스트
        """
        alerts = []
        status = budget.get_status()

        if status == 'warning':
            alerts.append({
                'level': 'warning',
                'message': f'Budget usage at {(budget.spent / budget.total) * 100:.1f}%',
                'remaining': budget.remaining
            })
        elif status == 'critical':
            alerts.append({
                'level': 'critical',
                'message': f'Budget usage at {(budget.spent / budget.total) * 100:.1f}%',
                'remaining': budget.remaining
            })

        if budget.is_exceeded():
            alerts.append({
                'level': 'critical',
                'message': f'Budget exceeded by ${budget.spent - budget.total:.2f}',
                'remaining': budget.remaining
            })

        return alerts

    def rank_by_priority(
        self,
        suggestions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """최적화 제안 우선순위 정렬

        Args:
            suggestions: 최적화 제안 리스트

        Returns:
            우선순위로 정렬된 리스트
        """
        # Sort by potential_savings descending
        return sorted(
            suggestions,
            key=lambda x: x.get('potential_savings', 0),
            reverse=True
        )
