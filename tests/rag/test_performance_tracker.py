"""
Tests for RAG performance tracking system.

Following TDD methodology (RED → GREEN → REFACTOR):
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Improve code quality
"""

import time

import pytest

from src.services.rag.performance_tracker import PerformanceTracker


class TestPerformanceTracker:
    """성능 추적 시스템 테스트"""

    @pytest.fixture
    def tracker(self):
        """트래커 인스턴스 생성"""
        return PerformanceTracker()

    def test_tracker_initialization(self, tracker):
        """트래커 초기화 검증"""
        assert tracker is not None
        assert hasattr(tracker, 'track_latency')
        assert hasattr(tracker, 'track_tokens')
        assert hasattr(tracker, 'get_metrics')

    def test_latency_tracking(self, tracker):
        """레이턴시 추적 검증"""
        # Simulate operation
        with tracker.track_latency('query_processing'):
            time.sleep(0.1)

        metrics = tracker.get_metrics()
        assert 'query_processing' in metrics['latency']
        assert metrics['latency']['query_processing']['count'] == 1
        assert metrics['latency']['query_processing']['mean'] >= 0.1
        assert metrics['latency']['query_processing']['mean'] < 0.2

    def test_token_tracking(self, tracker):
        """토큰 사용량 추적 검증"""
        tracker.track_tokens('gpt-4', prompt_tokens=100, completion_tokens=50)
        tracker.track_tokens('gpt-4', prompt_tokens=150, completion_tokens=75)

        metrics = tracker.get_metrics()
        assert 'tokens' in metrics
        assert 'gpt-4' in metrics['tokens']
        assert metrics['tokens']['gpt-4']['total_tokens'] == 375
        assert metrics['tokens']['gpt-4']['prompt_tokens'] == 250
        assert metrics['tokens']['gpt-4']['completion_tokens'] == 125

    def test_multiple_latency_measurements(self, tracker):
        """다중 레이턴시 측정 검증"""
        # Track multiple operations
        for i in range(5):
            with tracker.track_latency('retrieval'):
                time.sleep(0.05 + i * 0.01)

        metrics = tracker.get_metrics()
        retrieval_metrics = metrics['latency']['retrieval']

        assert retrieval_metrics['count'] == 5
        assert retrieval_metrics['mean'] > 0.05
        assert retrieval_metrics['min'] >= 0.05
        assert retrieval_metrics['max'] > retrieval_metrics['min']
        assert 'p50' in retrieval_metrics
        assert 'p95' in retrieval_metrics
        assert 'p99' in retrieval_metrics

    def test_operation_categories(self, tracker):
        """작업 카테고리별 추적 검증"""
        with tracker.track_latency('retrieval'):
            time.sleep(0.05)
        with tracker.track_latency('generation'):
            time.sleep(0.1)
        with tracker.track_latency('retrieval'):
            time.sleep(0.06)

        metrics = tracker.get_metrics()

        assert 'retrieval' in metrics['latency']
        assert 'generation' in metrics['latency']
        assert metrics['latency']['retrieval']['count'] == 2
        assert metrics['latency']['generation']['count'] == 1

    def test_cost_calculation(self, tracker):
        """비용 계산 검증"""
        # GPT-4 pricing: $0.03/1K prompt, $0.06/1K completion
        tracker.track_tokens('gpt-4', prompt_tokens=1000, completion_tokens=500)

        metrics = tracker.get_metrics()
        cost = metrics['tokens']['gpt-4']['estimated_cost']

        # 1000 * 0.03/1000 + 500 * 0.06/1000 = 0.03 + 0.03 = 0.06
        assert abs(cost - 0.06) < 0.001

    def test_reset_metrics(self, tracker):
        """메트릭 리셋 검증"""
        tracker.track_tokens('gpt-4', prompt_tokens=100, completion_tokens=50)
        with tracker.track_latency('test'):
            time.sleep(0.01)

        # Reset
        tracker.reset_metrics()

        metrics = tracker.get_metrics()
        assert len(metrics['latency']) == 0
        assert len(metrics['tokens']) == 0

    def test_export_to_dict(self, tracker):
        """딕셔너리 익스포트 검증"""
        tracker.track_tokens('gpt-4', prompt_tokens=100, completion_tokens=50)
        with tracker.track_latency('query'):
            time.sleep(0.05)

        export_data = tracker.export_metrics()

        assert 'timestamp' in export_data
        assert 'latency' in export_data
        assert 'tokens' in export_data
        assert isinstance(export_data['timestamp'], str)

    def test_percentile_calculations(self, tracker):
        """백분위수 계산 검증"""
        # Add known latencies
        latencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for latency in latencies:
            tracker._record_latency('test_op', latency)

        metrics = tracker.get_metrics()
        test_metrics = metrics['latency']['test_op']

        # P50 should be around 0.5
        assert abs(test_metrics['p50'] - 0.5) < 0.1
        # P95 should be around 0.95
        assert abs(test_metrics['p95'] - 0.95) < 0.1
        # P99 should be around 0.99
        assert abs(test_metrics['p99'] - 0.99) < 0.1

    def test_concurrent_tracking(self, tracker):
        """동시 추적 검증"""
        import threading

        def track_operation():
            with tracker.track_latency('concurrent_op'):
                time.sleep(0.01)

        threads = [threading.Thread(target=track_operation) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        metrics = tracker.get_metrics()
        assert metrics['latency']['concurrent_op']['count'] == 10
