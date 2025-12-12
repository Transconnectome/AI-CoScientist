"""
Tests for RAG A/B testing framework.

Following TDD methodology (RED → GREEN → REFACTOR):
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Improve code quality
"""

import pytest

from src.services.rag.ab_testing import ABTest, ABTestConfig, ABTestResult, Variant


class TestVariant:
    """변형(Variant) 테스트"""

    def test_variant_creation(self):
        """변형 생성 검증"""
        variant = Variant(
            name='control',
            config={'model': 'gpt-4', 'temperature': 0.7}
        )

        assert variant.name == 'control'
        assert variant.config['model'] == 'gpt-4'
        assert variant.config['temperature'] == 0.7

    def test_variant_equality(self):
        """변형 동등성 검증"""
        v1 = Variant(name='a', config={'model': 'gpt-4'})
        v2 = Variant(name='a', config={'model': 'gpt-4'})
        v3 = Variant(name='b', config={'model': 'gpt-3.5-turbo'})

        assert v1 == v2
        assert v1 != v3


class TestABTestConfig:
    """A/B 테스트 설정 테스트"""

    def test_config_creation(self):
        """설정 생성 검증"""
        config = ABTestConfig(
            name='model_comparison',
            variants=[
                Variant('control', {'model': 'gpt-4'}),
                Variant('treatment', {'model': 'gpt-3.5-turbo'})
            ],
            traffic_split={'control': 0.5, 'treatment': 0.5}
        )

        assert config.name == 'model_comparison'
        assert len(config.variants) == 2
        assert config.traffic_split['control'] == 0.5

    def test_config_validation(self):
        """설정 검증"""
        config = ABTestConfig(
            name='test',
            variants=[
                Variant('a', {}),
                Variant('b', {})
            ],
            traffic_split={'a': 0.5, 'b': 0.5}
        )

        # Traffic split should sum to 1.0
        assert abs(sum(config.traffic_split.values()) - 1.0) < 0.001

    def test_get_variant_by_name(self):
        """이름으로 변형 가져오기"""
        config = ABTestConfig(
            name='test',
            variants=[
                Variant('a', {'model': 'gpt-4'}),
                Variant('b', {'model': 'gpt-3.5-turbo'})
            ],
            traffic_split={'a': 0.5, 'b': 0.5}
        )

        variant = config.get_variant('a')
        assert variant is not None
        assert variant.name == 'a'
        assert variant.config['model'] == 'gpt-4'


class TestABTestResult:
    """A/B 테스트 결과 테스트"""

    def test_result_initialization(self):
        """결과 초기화 검증"""
        result = ABTestResult(
            variant_name='control',
            metrics={'faithfulness': 0.85, 'latency': 2.5},
            cost=0.05
        )

        assert result.variant_name == 'control'
        assert result.metrics['faithfulness'] == 0.85
        assert result.cost == 0.05

    def test_result_comparison(self):
        """결과 비교 검증"""
        r1 = ABTestResult('a', {'score': 0.8}, 0.05)
        r2 = ABTestResult('b', {'score': 0.9}, 0.03)

        # Higher score is better
        assert r2.metrics['score'] > r1.metrics['score']


class TestABTest:
    """A/B 테스트 프레임워크 테스트"""

    @pytest.fixture
    def ab_test(self):
        """A/B 테스트 인스턴스 생성"""
        config = ABTestConfig(
            name='model_comparison',
            variants=[
                Variant('gpt4', {'model': 'gpt-4'}),
                Variant('gpt35', {'model': 'gpt-3.5-turbo'})
            ],
            traffic_split={'gpt4': 0.5, 'gpt35': 0.5}
        )
        return ABTest(config)

    def test_ab_test_initialization(self, ab_test):
        """A/B 테스트 초기화 검증"""
        assert ab_test is not None
        assert hasattr(ab_test, 'run_experiment')
        assert hasattr(ab_test, 'analyze_results')

    def test_variant_assignment(self, ab_test):
        """변형 할당 검증"""
        # Test deterministic assignment
        variant1 = ab_test.assign_variant('user123')
        variant2 = ab_test.assign_variant('user123')

        # Same user should get same variant
        assert variant1 == variant2

    def test_traffic_distribution(self):
        """트래픽 분배 검증"""
        config = ABTestConfig(
            name='test',
            variants=[
                Variant('a', {}),
                Variant('b', {})
            ],
            traffic_split={'a': 0.5, 'b': 0.5}
        )
        ab_test = ABTest(config)

        # Assign 1000 users
        assignments = {}
        for i in range(1000):
            variant = ab_test.assign_variant(f'user{i}')
            assignments[variant] = assignments.get(variant, 0) + 1

        # Should be roughly 50/50 (allow 10% deviation)
        assert assignments['a'] > 400
        assert assignments['a'] < 600
        assert assignments['b'] > 400
        assert assignments['b'] < 600

    def test_add_result(self, ab_test):
        """결과 추가 검증"""
        result = ABTestResult(
            variant_name='gpt4',
            metrics={'faithfulness': 0.85},
            cost=0.05
        )

        ab_test.add_result(result)

        # Should have 1 result
        assert len(ab_test.get_all_results()) == 1

    def test_result_aggregation(self, ab_test):
        """결과 집계 검증"""
        # Add multiple results for each variant
        for i in range(10):
            ab_test.add_result(ABTestResult(
                'gpt4',
                {'faithfulness': 0.8 + i * 0.01},
                0.05
            ))
            ab_test.add_result(ABTestResult(
                'gpt35',
                {'faithfulness': 0.7 + i * 0.01},
                0.01
            ))

        aggregated = ab_test.aggregate_results()

        assert 'gpt4' in aggregated
        assert 'gpt35' in aggregated
        assert aggregated['gpt4']['count'] == 10
        assert aggregated['gpt35']['count'] == 10

    def test_statistical_significance(self, ab_test):
        """통계적 유의성 검증"""
        # Add results with clear difference
        for _ in range(100):
            ab_test.add_result(ABTestResult(
                'gpt4',
                {'faithfulness': 0.85},
                0.05
            ))
            ab_test.add_result(ABTestResult(
                'gpt35',
                {'faithfulness': 0.75},
                0.01
            ))

        analysis = ab_test.analyze_results('faithfulness')

        assert 'winner' in analysis
        assert 'confidence' in analysis
        assert 'p_value' in analysis

    def test_cost_effectiveness_analysis(self, ab_test):
        """비용 효율성 분석 검증"""
        # Variant A: Higher quality, higher cost
        for _ in range(50):
            ab_test.add_result(ABTestResult(
                'gpt4',
                {'faithfulness': 0.9},
                0.05
            ))

        # Variant B: Lower quality, lower cost
        for _ in range(50):
            ab_test.add_result(ABTestResult(
                'gpt35',
                {'faithfulness': 0.8},
                0.01
            ))

        cost_analysis = ab_test.analyze_cost_effectiveness('faithfulness')

        assert 'gpt4' in cost_analysis
        assert 'gpt35' in cost_analysis
        assert 'quality_per_dollar' in cost_analysis['gpt4']
        assert 'quality_per_dollar' in cost_analysis['gpt35']

    def test_winner_declaration(self, ab_test):
        """승자 선언 검증"""
        # Add clear winner data
        for _ in range(100):
            ab_test.add_result(ABTestResult('gpt4', {'score': 0.9}, 0.05))
            ab_test.add_result(ABTestResult('gpt35', {'score': 0.7}, 0.01))

        winner = ab_test.declare_winner('score', min_confidence=0.95)

        assert winner is not None
        assert winner['variant'] in ['gpt4', 'gpt35']
        assert 'confidence' in winner

    def test_export_results(self, ab_test):
        """결과 익스포트 검증"""
        ab_test.add_result(ABTestResult('gpt4', {'score': 0.85}, 0.05))

        export = ab_test.export_results()

        assert 'config' in export
        assert 'results' in export
        assert 'timestamp' in export
