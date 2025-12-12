"""
RAG Performance Monitoring with Prometheus Metrics

Implementation for: Prometheus metrics for RAG performance
Created: 2025-12-05

Acceptance Criteria:
- Latency tracking per RAG strategy
- Quality score distribution monitoring
- Resource utilization metrics
- Grafana dashboard integration

This module provides comprehensive monitoring for RAG system performance,
including response times, quality metrics, and resource utilization tracking.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import functools

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        CollectorRegistry, generate_latest,
        start_http_server, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Install with: pip install prometheus-client")

    # Create mock classes for when Prometheus is not available
    class CollectorRegistry:
        pass

    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def inc(self, value=1):
            pass

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def observe(self, value):
            pass

    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def set(self, value):
            pass

    class Summary:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def observe(self, value):
            pass

    def generate_latest(registry):
        return "# Prometheus not available"

    def start_http_server(port, registry=None):
        pass

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class RAGMetrics:
    """Container for RAG performance metrics"""
    latency: float
    quality_score: float
    tokens_processed: int
    retrieval_time: float
    generation_time: float
    context_relevance: float
    faithfulness: float
    answer_relevancy: float
    strategy: str
    timestamp: datetime

class PrometheusMetricsCollector:
    """Prometheus metrics collector for RAG system monitoring"""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector with Prometheus registry"""
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()

        # Internal tracking
        self._request_count = 0
        self._error_count = 0
        self._active_requests = 0

        # Performance windows for calculating percentiles
        self._latency_window = deque(maxlen=1000)
        self._quality_window = deque(maxlen=1000)

        logger.info("RAG Prometheus metrics collector initialized")

    def _setup_metrics(self):
        """Setup all Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available - metrics disabled")
            return

        # Request counters
        self.request_total = Counter(
            'rag_requests_total',
            'Total number of RAG requests',
            ['strategy', 'domain', 'status'],
            registry=self.registry
        )

        self.error_total = Counter(
            'rag_errors_total',
            'Total number of RAG errors',
            ['strategy', 'error_type'],
            registry=self.registry
        )

        # Latency metrics
        self.request_duration = Histogram(
            'rag_request_duration_seconds',
            'Time spent processing RAG requests',
            ['strategy', 'component'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry
        )

        self.retrieval_duration = Histogram(
            'rag_retrieval_duration_seconds',
            'Time spent on document retrieval',
            ['strategy', 'index_type'],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
            registry=self.registry
        )

        self.generation_duration = Histogram(
            'rag_generation_duration_seconds',
            'Time spent on answer generation',
            ['strategy', 'model'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
            registry=self.registry
        )

        # Quality metrics
        self.quality_score = Histogram(
            'rag_quality_score',
            'Distribution of RAG quality scores',
            ['strategy', 'metric_type'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry
        )

        self.faithfulness_score = Gauge(
            'rag_faithfulness_current',
            'Current faithfulness score',
            ['strategy'],
            registry=self.registry
        )

        self.answer_relevancy_score = Gauge(
            'rag_answer_relevancy_current',
            'Current answer relevancy score',
            ['strategy'],
            registry=self.registry
        )

        self.context_precision_score = Gauge(
            'rag_context_precision_current',
            'Current context precision score',
            ['strategy'],
            registry=self.registry
        )

        # Resource utilization
        self.tokens_processed_total = Counter(
            'rag_tokens_processed_total',
            'Total tokens processed',
            ['strategy', 'token_type'],
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'rag_memory_usage_bytes',
            'Current memory usage',
            ['component'],
            registry=self.registry
        )

        self.active_requests = Gauge(
            'rag_active_requests',
            'Number of currently active requests',
            ['strategy'],
            registry=self.registry
        )

        # Context and retrieval metrics
        self.documents_retrieved = Histogram(
            'rag_documents_retrieved_count',
            'Number of documents retrieved per request',
            ['strategy'],
            buckets=(1, 2, 5, 10, 20, 50, 100),
            registry=self.registry
        )

        self.context_length = Histogram(
            'rag_context_length_tokens',
            'Length of context in tokens',
            ['strategy'],
            buckets=(100, 500, 1000, 2000, 4000, 8000, 16000, 32000),
            registry=self.registry
        )

        # Model performance
        self.model_calls_total = Counter(
            'rag_model_calls_total',
            'Total model API calls',
            ['strategy', 'model', 'call_type'],
            registry=self.registry
        )

        # Summary metrics for percentiles
        self.request_latency_summary = Summary(
            'rag_request_latency_summary',
            'Summary of request latencies',
            ['strategy'],
            registry=self.registry
        )

    def record_request(
        self,
        strategy: str,
        domain: str = "unknown",
        latency: float = 0.0,
        status: str = "success"
    ):
        """Record a RAG request"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.request_total.labels(
            strategy=strategy,
            domain=domain,
            status=status
        ).inc()

        if latency > 0:
            self.request_duration.labels(
                strategy=strategy,
                component="total"
            ).observe(latency)

            self.request_latency_summary.labels(
                strategy=strategy
            ).observe(latency)

            # Update internal tracking
            self._latency_window.append(latency)

    def record_error(self, strategy: str, error_type: str):
        """Record an error"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.error_total.labels(
            strategy=strategy,
            error_type=error_type
        ).inc()

        self._error_count += 1

    def record_retrieval(
        self,
        strategy: str,
        duration: float,
        index_type: str = "vector",
        doc_count: int = 0
    ):
        """Record retrieval metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.retrieval_duration.labels(
            strategy=strategy,
            index_type=index_type
        ).observe(duration)

        if doc_count > 0:
            self.documents_retrieved.labels(
                strategy=strategy
            ).observe(doc_count)

    def record_generation(
        self,
        strategy: str,
        duration: float,
        model: str = "unknown",
        input_tokens: int = 0,
        output_tokens: int = 0
    ):
        """Record generation metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.generation_duration.labels(
            strategy=strategy,
            model=model
        ).observe(duration)

        # Record token usage
        if input_tokens > 0:
            self.tokens_processed_total.labels(
                strategy=strategy,
                token_type="input"
            ).inc(input_tokens)

        if output_tokens > 0:
            self.tokens_processed_total.labels(
                strategy=strategy,
                token_type="output"
            ).inc(output_tokens)

    def record_quality_metrics(
        self,
        strategy: str,
        faithfulness: Optional[float] = None,
        answer_relevancy: Optional[float] = None,
        context_precision: Optional[float] = None,
        overall_score: Optional[float] = None
    ):
        """Record quality metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        if faithfulness is not None:
            self.faithfulness_score.labels(strategy=strategy).set(faithfulness)
            self.quality_score.labels(
                strategy=strategy,
                metric_type="faithfulness"
            ).observe(faithfulness)

        if answer_relevancy is not None:
            self.answer_relevancy_score.labels(strategy=strategy).set(answer_relevancy)
            self.quality_score.labels(
                strategy=strategy,
                metric_type="answer_relevancy"
            ).observe(answer_relevancy)

        if context_precision is not None:
            self.context_precision_score.labels(strategy=strategy).set(context_precision)
            self.quality_score.labels(
                strategy=strategy,
                metric_type="context_precision"
            ).observe(context_precision)

        if overall_score is not None:
            self.quality_score.labels(
                strategy=strategy,
                metric_type="overall"
            ).observe(overall_score)

            # Update internal tracking
            self._quality_window.append(overall_score)

    def record_context_metrics(
        self,
        strategy: str,
        context_length: int,
        context_relevance: Optional[float] = None
    ):
        """Record context-related metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.context_length.labels(strategy=strategy).observe(context_length)

        if context_relevance is not None:
            self.quality_score.labels(
                strategy=strategy,
                metric_type="context_relevance"
            ).observe(context_relevance)

    def record_model_call(
        self,
        strategy: str,
        model: str,
        call_type: str = "completion"
    ):
        """Record model API calls"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.model_calls_total.labels(
            strategy=strategy,
            model=model,
            call_type=call_type
        ).inc()

    def update_resource_usage(self, component: str, memory_bytes: int):
        """Update resource utilization metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.memory_usage.labels(component=component).set(memory_bytes)

    def set_active_requests(self, strategy: str, count: int):
        """Set current active requests count"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.active_requests.labels(strategy=strategy).set(count)
        self._active_requests = count

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        summary = {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "active_requests": self._active_requests,
            "avg_latency": sum(self._latency_window) / len(self._latency_window) if self._latency_window else 0,
            "avg_quality": sum(self._quality_window) / len(self._quality_window) if self._quality_window else 0,
            "error_rate": self._error_count / max(self._request_count, 1),
            "timestamp": datetime.now().isoformat()
        }

        return summary

class RAGMetricsManager:
    """High-level manager for RAG metrics collection"""

    def __init__(self, enable_prometheus: bool = True, metrics_port: int = 8000):
        """Initialize metrics manager"""
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.metrics_port = metrics_port

        if self.enable_prometheus:
            self.collector = PrometheusMetricsCollector()
            self._start_metrics_server()
        else:
            self.collector = None
            logger.warning("Prometheus metrics disabled")

        # Internal metrics storage for non-Prometheus mode
        self._internal_metrics = defaultdict(list)
        self._lock = threading.Lock()

        logger.info(f"RAG metrics manager initialized (Prometheus: {self.enable_prometheus})")

    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        try:
            start_http_server(self.metrics_port, registry=self.collector.registry)
            logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            self.enable_prometheus = False

    def record_rag_request(self, metrics: RAGMetrics):
        """Record complete RAG request metrics"""
        strategy = metrics.strategy

        if self.enable_prometheus and self.collector:
            # Record all metrics
            self.collector.record_request(
                strategy=strategy,
                latency=metrics.latency,
                status="success"
            )

            self.collector.record_retrieval(
                strategy=strategy,
                duration=metrics.retrieval_time
            )

            self.collector.record_generation(
                strategy=strategy,
                duration=metrics.generation_time,
                input_tokens=metrics.tokens_processed
            )

            self.collector.record_quality_metrics(
                strategy=strategy,
                faithfulness=metrics.faithfulness,
                answer_relevancy=metrics.answer_relevancy,
                overall_score=metrics.quality_score
            )

            self.collector.record_context_metrics(
                strategy=strategy,
                context_length=metrics.tokens_processed,
                context_relevance=metrics.context_relevance
            )

        # Also store internally for analysis
        with self._lock:
            self._internal_metrics[strategy].append(metrics)

    def record_error(self, strategy: str, error_type: str):
        """Record an error"""
        if self.enable_prometheus and self.collector:
            self.collector.record_error(strategy, error_type)

        with self._lock:
            self._internal_metrics[f"{strategy}_errors"].append({
                "error_type": error_type,
                "timestamp": datetime.now()
            })

    def get_strategy_performance(self, strategy: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for a strategy"""
        cutoff = datetime.now() - timedelta(hours=hours)

        with self._lock:
            strategy_metrics = [
                m for m in self._internal_metrics[strategy]
                if m.timestamp >= cutoff
            ]

        if not strategy_metrics:
            return {"strategy": strategy, "metrics": [], "summary": {}}

        # Calculate summary statistics
        latencies = [m.latency for m in strategy_metrics]
        quality_scores = [m.quality_score for m in strategy_metrics if m.quality_score > 0]

        summary = {
            "request_count": len(strategy_metrics),
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "p95_latency": sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0,
            "avg_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "avg_retrieval_time": sum(m.retrieval_time for m in strategy_metrics) / len(strategy_metrics),
            "avg_generation_time": sum(m.generation_time for m in strategy_metrics) / len(strategy_metrics),
            "time_window_hours": hours
        }

        return {
            "strategy": strategy,
            "summary": summary,
            "metrics_count": len(strategy_metrics)
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        result = {"strategies": {}}

        with self._lock:
            for strategy in self._internal_metrics:
                if not strategy.endswith("_errors"):
                    result["strategies"][strategy] = self.get_strategy_performance(strategy)

        if self.enable_prometheus and self.collector:
            result["prometheus_summary"] = self.collector.get_metrics_summary()

        return result

    def export_prometheus_metrics(self) -> str:
        """Export Prometheus metrics in text format"""
        if self.enable_prometheus and self.collector:
            return generate_latest(self.collector.registry)
        return ""

def rag_metrics_decorator(strategy: str, manager: RAGMetricsManager):
    """Decorator to automatically track RAG function metrics"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                # Execute function
                result = await func(*args, **kwargs)

                # Extract metrics if result includes them
                if isinstance(result, dict) and "metrics" in result:
                    metrics = result["metrics"]
                    if isinstance(metrics, RAGMetrics):
                        manager.record_rag_request(metrics)

                return result

            except Exception as e:
                # Record error
                manager.record_error(strategy, type(e).__name__)
                raise

            finally:
                # Record basic latency
                latency = time.time() - start_time
                if manager.enable_prometheus and manager.collector:
                    manager.collector.record_request(strategy=strategy, latency=latency)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                if isinstance(result, dict) and "metrics" in result:
                    metrics = result["metrics"]
                    if isinstance(metrics, RAGMetrics):
                        manager.record_rag_request(metrics)

                return result

            except Exception as e:
                manager.record_error(strategy, type(e).__name__)
                raise

            finally:
                latency = time.time() - start_time
                if manager.enable_prometheus and manager.collector:
                    manager.collector.record_request(strategy=strategy, latency=latency)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator

# Global metrics manager instance
_global_metrics_manager: Optional[RAGMetricsManager] = None

def get_metrics_manager() -> RAGMetricsManager:
    """Get global metrics manager instance"""
    global _global_metrics_manager

    if _global_metrics_manager is None:
        _global_metrics_manager = RAGMetricsManager()

    return _global_metrics_manager

def initialize_metrics(enable_prometheus: bool = True, metrics_port: int = 8000) -> RAGMetricsManager:
    """Initialize global metrics manager"""
    global _global_metrics_manager

    _global_metrics_manager = RAGMetricsManager(
        enable_prometheus=enable_prometheus,
        metrics_port=metrics_port
    )

    return _global_metrics_manager

# Example usage and testing
if __name__ == "__main__":
    async def test_metrics():
        """Test the metrics system"""
        # Initialize metrics
        manager = initialize_metrics(enable_prometheus=False)  # Disable for testing

        # Create sample metrics
        sample_metrics = RAGMetrics(
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

        print("🔄 Testing RAG metrics collection...")

        # Record metrics
        manager.record_rag_request(sample_metrics)

        # Test decorator
        @rag_metrics_decorator("test_strategy", manager)
        async def sample_rag_function():
            await asyncio.sleep(0.1)  # Simulate processing
            return {
                "answer": "Test answer",
                "metrics": sample_metrics
            }

        await sample_rag_function()

        # Get performance summary
        performance = manager.get_strategy_performance("hybrid_rag")
        print(f"📊 Strategy Performance: {performance}")

        # Get all metrics
        all_metrics = manager.get_all_metrics()
        print(f"📈 All Metrics: {all_metrics}")

        print("✅ RAG metrics testing completed successfully!")

    # Run test
    asyncio.run(test_metrics())