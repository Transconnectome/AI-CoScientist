"""
Tests for RAG Metrics System

Testing the Prometheus metrics integration for RAG performance monitoring.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from monitoring.rag_metrics import (
    RAGMetrics, RAGMetricsManager, PrometheusMetricsCollector,
    rag_metrics_decorator, get_metrics_manager, initialize_metrics
)

class TestRAGMetrics:
    """Test RAG metrics data structures"""

    def test_rag_metrics_creation(self):
        """Test RAGMetrics dataclass creation"""
        metrics = RAGMetrics(
            latency=1.5,
            quality_score=0.85,
            tokens_processed=1200,
            retrieval_time=0.3,
            generation_time=1.2,
            context_relevance=0.9,
            faithfulness=0.8,
            answer_relevancy=0.87,
            strategy="hybrid_rag",
            timestamp=datetime.now()
        )

        assert metrics.latency == 1.5
        assert metrics.quality_score == 0.85
        assert metrics.strategy == "hybrid_rag"

class TestPrometheusMetricsCollector:
    """Test Prometheus metrics collector"""

    def test_collector_initialization(self):
        """Test collector initializes without errors"""
        collector = PrometheusMetricsCollector()
        assert collector is not None

    def test_record_request(self):
        """Test recording requests"""
        collector = PrometheusMetricsCollector()
        # Should not raise errors even without Prometheus
        collector.record_request("test_strategy", "neuroscience", 1.5, "success")

    def test_record_quality_metrics(self):
        """Test recording quality metrics"""
        collector = PrometheusMetricsCollector()
        collector.record_quality_metrics(
            "test_strategy",
            faithfulness=0.8,
            answer_relevancy=0.9,
            context_precision=0.7,
            overall_score=0.85
        )

    def test_get_metrics_summary(self):
        """Test getting metrics summary"""
        collector = PrometheusMetricsCollector()
        summary = collector.get_metrics_summary()

        assert "request_count" in summary
        assert "error_count" in summary
        assert "timestamp" in summary

class TestRAGMetricsManager:
    """Test RAG metrics manager"""

    def test_manager_initialization(self):
        """Test manager initializes correctly"""
        manager = RAGMetricsManager(enable_prometheus=False)
        assert manager is not None
        assert not manager.enable_prometheus

    def test_record_rag_request(self):
        """Test recording complete RAG request"""
        manager = RAGMetricsManager(enable_prometheus=False)

        metrics = RAGMetrics(
            latency=1.5,
            quality_score=0.85,
            tokens_processed=1200,
            retrieval_time=0.3,
            generation_time=1.2,
            context_relevance=0.9,
            faithfulness=0.8,
            answer_relevancy=0.87,
            strategy="test_strategy",
            timestamp=datetime.now()
        )

        manager.record_rag_request(metrics)

        # Check internal storage
        performance = manager.get_strategy_performance("test_strategy")
        assert performance["strategy"] == "test_strategy"
        assert performance["metrics_count"] == 1

    def test_record_error(self):
        """Test error recording"""
        manager = RAGMetricsManager(enable_prometheus=False)
        manager.record_error("test_strategy", "ValueError")

        # Should store error internally
        assert "test_strategy_errors" in manager._internal_metrics

    def test_get_strategy_performance(self):
        """Test strategy performance retrieval"""
        manager = RAGMetricsManager(enable_prometheus=False)

        # Add multiple metrics
        for i in range(3):
            metrics = RAGMetrics(
                latency=1.0 + i * 0.1,
                quality_score=0.8 + i * 0.05,
                tokens_processed=1000 + i * 100,
                retrieval_time=0.2 + i * 0.05,
                generation_time=0.8 + i * 0.1,
                context_relevance=0.9,
                faithfulness=0.8,
                answer_relevancy=0.85,
                strategy="test_strategy",
                timestamp=datetime.now()
            )
            manager.record_rag_request(metrics)

        performance = manager.get_strategy_performance("test_strategy")

        assert performance["metrics_count"] == 3
        assert "summary" in performance
        assert performance["summary"]["request_count"] == 3

    def test_get_all_metrics(self):
        """Test getting all metrics"""
        manager = RAGMetricsManager(enable_prometheus=False)

        # Add metrics for multiple strategies
        for strategy in ["strategy1", "strategy2"]:
            metrics = RAGMetrics(
                latency=1.5,
                quality_score=0.85,
                tokens_processed=1200,
                retrieval_time=0.3,
                generation_time=1.2,
                context_relevance=0.9,
                faithfulness=0.8,
                answer_relevancy=0.87,
                strategy=strategy,
                timestamp=datetime.now()
            )
            manager.record_rag_request(metrics)

        all_metrics = manager.get_all_metrics()
        assert "strategies" in all_metrics
        assert len(all_metrics["strategies"]) == 2

class TestMetricsDecorator:
    """Test metrics decorator functionality"""

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test async function decoration"""
        manager = RAGMetricsManager(enable_prometheus=False)

        @rag_metrics_decorator("test_strategy", manager)
        async def test_rag_function():
            await asyncio.sleep(0.01)  # Simulate work
            return {"result": "success"}

        result = await test_rag_function()
        assert result["result"] == "success"

    def test_sync_decorator(self):
        """Test sync function decoration"""
        manager = RAGMetricsManager(enable_prometheus=False)

        @rag_metrics_decorator("test_strategy", manager)
        def test_rag_function():
            time.sleep(0.01)  # Simulate work
            return {"result": "success"}

        result = test_rag_function()
        assert result["result"] == "success"

    @pytest.mark.asyncio
    async def test_decorator_with_metrics_result(self):
        """Test decorator with metrics in result"""
        manager = RAGMetricsManager(enable_prometheus=False)

        @rag_metrics_decorator("test_strategy", manager)
        async def test_rag_function():
            metrics = RAGMetrics(
                latency=1.0,
                quality_score=0.8,
                tokens_processed=1000,
                retrieval_time=0.2,
                generation_time=0.8,
                context_relevance=0.9,
                faithfulness=0.8,
                answer_relevancy=0.85,
                strategy="test_strategy",
                timestamp=datetime.now()
            )
            return {"result": "success", "metrics": metrics}

        result = await test_rag_function()
        assert result["result"] == "success"

        # Check that metrics were recorded
        performance = manager.get_strategy_performance("test_strategy")
        assert performance["metrics_count"] == 1

    @pytest.mark.asyncio
    async def test_decorator_error_handling(self):
        """Test decorator error handling"""
        manager = RAGMetricsManager(enable_prometheus=False)

        @rag_metrics_decorator("test_strategy", manager)
        async def test_failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await test_failing_function()

        # Check that error was recorded
        assert "test_strategy_errors" in manager._internal_metrics

class TestGlobalMetricsManager:
    """Test global metrics manager functions"""

    def test_get_metrics_manager(self):
        """Test getting global metrics manager"""
        manager = get_metrics_manager()
        assert manager is not None

    def test_initialize_metrics(self):
        """Test initializing global metrics"""
        manager = initialize_metrics(enable_prometheus=False, metrics_port=8001)
        assert manager is not None
        assert not manager.enable_prometheus

class TestMetricsIntegration:
    """Integration tests for metrics system"""

    @pytest.mark.asyncio
    async def test_complete_rag_workflow_metrics(self):
        """Test complete RAG workflow with metrics"""
        manager = RAGMetricsManager(enable_prometheus=False)

        # Simulate complete RAG request
        start_time = time.time()

        # Simulate retrieval phase
        retrieval_start = time.time()
        await asyncio.sleep(0.01)
        retrieval_time = time.time() - retrieval_start

        # Simulate generation phase
        generation_start = time.time()
        await asyncio.sleep(0.02)
        generation_time = time.time() - generation_start

        total_latency = time.time() - start_time

        # Create comprehensive metrics
        metrics = RAGMetrics(
            latency=total_latency,
            quality_score=0.88,
            tokens_processed=1500,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            context_relevance=0.92,
            faithfulness=0.85,
            answer_relevancy=0.89,
            strategy="hybrid_rag_v2",
            timestamp=datetime.now()
        )

        # Record metrics
        manager.record_rag_request(metrics)

        # Verify metrics are properly stored
        performance = manager.get_strategy_performance("hybrid_rag_v2")

        assert performance["metrics_count"] == 1
        assert "summary" in performance
        summary = performance["summary"]

        assert summary["request_count"] == 1
        assert summary["avg_latency"] > 0
        assert summary["avg_quality"] == 0.88
        assert summary["avg_retrieval_time"] == retrieval_time
        assert summary["avg_generation_time"] == generation_time

    def test_export_prometheus_metrics(self):
        """Test exporting Prometheus metrics"""
        manager = RAGMetricsManager(enable_prometheus=False)

        # Should return empty string when Prometheus disabled
        metrics_text = manager.export_prometheus_metrics()
        assert isinstance(metrics_text, str)

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])