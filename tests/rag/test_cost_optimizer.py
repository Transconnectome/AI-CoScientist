"""
Tests for RAG cost optimization system.

Following TDD methodology (RED → GREEN → REFACTOR):
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Improve code quality
"""

import pytest

from src.services.rag.cost_optimizer import CostBudget, CostOptimizer


class TestCostBudget:
    """비용 예산 테스트"""

    def test_budget_initialization(self):
        """예산 초기화 검증"""
        budget = CostBudget(total=100.0, warning_threshold=0.8, critical_threshold=0.95)

        assert budget.total == 100.0
        assert budget.warning_threshold == 0.8
        assert budget.critical_threshold == 0.95
        assert budget.spent == 0.0
        assert budget.remaining == 100.0

    def test_budget_tracking(self):
        """예산 추적 검증"""
        budget = CostBudget(total=100.0)

        budget.add_expense(25.0, 'retrieval')
        assert budget.spent == 25.0
        assert budget.remaining == 75.0

        budget.add_expense(30.0, 'generation')
        assert budget.spent == 55.0
        assert budget.remaining == 45.0

    def test_budget_status(self):
        """예산 상태 검증"""
        budget = CostBudget(total=100.0, warning_threshold=0.8, critical_threshold=0.95)

        # Normal status
        budget.add_expense(50.0, 'test')
        assert budget.get_status() == 'normal'

        # Warning status (80%+)
        budget.add_expense(35.0, 'test')
        assert budget.get_status() == 'warning'

        # Critical status (95%+)
        budget.add_expense(12.0, 'test')
        assert budget.get_status() == 'critical'

    def test_budget_exceeded(self):
        """예산 초과 검증"""
        budget = CostBudget(total=100.0)

        budget.add_expense(90.0, 'test')
        assert not budget.is_exceeded()

        budget.add_expense(15.0, 'test')
        assert budget.is_exceeded()

    def test_expense_breakdown(self):
        """비용 항목별 분석 검증"""
        budget = CostBudget(total=100.0)

        budget.add_expense(30.0, 'retrieval')
        budget.add_expense(50.0, 'generation')
        budget.add_expense(10.0, 'retrieval')

        breakdown = budget.get_breakdown()
        assert breakdown['retrieval'] == 40.0
        assert breakdown['generation'] == 50.0


class TestCostOptimizer:
    """비용 최적화 시스템 테스트"""

    @pytest.fixture
    def optimizer(self):
        """옵티마이저 인스턴스 생성"""
        return CostOptimizer()

    def test_optimizer_initialization(self, optimizer):
        """옵티마이저 초기화 검증"""
        assert optimizer is not None
        assert hasattr(optimizer, 'analyze_costs')
        assert hasattr(optimizer, 'suggest_optimizations')
        assert hasattr(optimizer, 'estimate_savings')

    def test_model_cost_comparison(self, optimizer):
        """모델별 비용 비교 검증"""
        costs = {
            'gpt-4': {'prompt_tokens': 1000, 'completion_tokens': 500},
            'gpt-3.5-turbo': {'prompt_tokens': 1000, 'completion_tokens': 500},
        }

        comparison = optimizer.compare_model_costs(costs)

        assert 'gpt-4' in comparison
        assert 'gpt-3.5-turbo' in comparison
        assert comparison['gpt-4'] > comparison['gpt-3.5-turbo']

    def test_optimization_suggestions(self, optimizer):
        """최적화 제안 생성 검증"""
        usage_data = {
            'tokens': {
                'gpt-4': {
                    'prompt_tokens': 10000,
                    'completion_tokens': 5000,
                    'estimated_cost': 0.75
                }
            }
        }

        suggestions = optimizer.suggest_optimizations(usage_data)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all('description' in s for s in suggestions)
        assert all('potential_savings' in s for s in suggestions)

    def test_savings_estimation(self, optimizer):
        """비용 절감 추정 검증"""
        current_usage = {
            'model': 'gpt-4',
            'prompt_tokens': 10000,
            'completion_tokens': 5000
        }

        savings = optimizer.estimate_savings(
            current_usage,
            target_model='gpt-3.5-turbo'
        )

        assert savings > 0
        assert isinstance(savings, float)

    def test_cache_hit_savings(self, optimizer):
        """캐시 히트율 기반 절감 추정"""
        savings = optimizer.estimate_cache_savings(
            total_requests=1000,
            cache_hit_rate=0.3,
            avg_cost_per_request=0.01
        )

        # 30% cache hit = 30% savings
        expected_savings = 1000 * 0.3 * 0.01
        assert abs(savings - expected_savings) < 0.001

    def test_batch_optimization_savings(self, optimizer):
        """배치 최적화 절감 추정"""
        savings = optimizer.estimate_batch_savings(
            total_requests=1000,
            current_batch_size=1,
            optimal_batch_size=10,
            overhead_per_batch=0.001
        )

        assert savings > 0

    def test_cost_trend_analysis(self, optimizer):
        """비용 추세 분석 검증"""
        cost_history = [
            {'timestamp': '2024-01-01', 'cost': 10.0},
            {'timestamp': '2024-01-02', 'cost': 12.0},
            {'timestamp': '2024-01-03', 'cost': 15.0},
        ]

        trend = optimizer.analyze_trend(cost_history)

        assert 'direction' in trend
        assert trend['direction'] == 'increasing'
        assert 'rate' in trend

    def test_budget_alert_thresholds(self, optimizer):
        """예산 경고 임계값 검증"""
        budget = CostBudget(total=100.0, warning_threshold=0.8)

        alerts = optimizer.check_budget_alerts(budget)

        # Normal state - no alerts
        assert len(alerts) == 0

        # Add expense to trigger warning
        budget.add_expense(85.0, 'test')
        alerts = optimizer.check_budget_alerts(budget)

        assert len(alerts) > 0
        assert any('warning' in alert['level'] for alert in alerts)

    def test_optimization_priority_ranking(self, optimizer):
        """최적화 우선순위 랭킹 검증"""
        suggestions = [
            {'description': 'Switch to cheaper model', 'potential_savings': 50.0},
            {'description': 'Enable caching', 'potential_savings': 30.0},
            {'description': 'Optimize batch size', 'potential_savings': 10.0},
        ]

        ranked = optimizer.rank_by_priority(suggestions)

        # Should be sorted by potential_savings descending
        assert ranked[0]['potential_savings'] == 50.0
        assert ranked[1]['potential_savings'] == 30.0
        assert ranked[2]['potential_savings'] == 10.0
