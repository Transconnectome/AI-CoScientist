"""
Performance Benchmark System for RAG Components

Implementation for: Comprehensive performance benchmarking and regression detection
Created: 2025-12-05

Acceptance Criteria:
- Strategy performance comparison and ranking
- Response time and quality regression detection
- Automated benchmark suite with realistic workloads
- Performance trend analysis and reporting

This module provides comprehensive performance benchmarking for RAG systems
with automated regression detection and trend analysis.
"""

import asyncio
import logging
import time
import json
import statistics
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# External dependencies with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Core dependencies
from ..rag.unified_rag_orchestrator import (
    UnifiedRAGOrchestrator, QueryContext, RAGResponse, RAGStrategy,
    QueryComplexity, QueryDomain
)
from ..rag.advanced_query_classifier import MLQueryClassifier

logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    """Types of benchmarks"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    QUALITY = "quality"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    COMPREHENSIVE = "comprehensive"

class WorkloadType(Enum):
    """Types of workloads"""
    SIMPLE_QUERIES = "simple_queries"
    COMPLEX_QUERIES = "complex_queries"
    MIXED_WORKLOAD = "mixed_workload"
    CONCURRENT_LOAD = "concurrent_load"
    STRESS_TEST = "stress_test"
    DOMAIN_SPECIFIC = "domain_specific"

class RegressionSeverity(Enum):
    """Regression severity levels"""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"

@dataclass
class BenchmarkQuery:
    """Individual benchmark query"""
    query: str
    expected_strategy: Optional[RAGStrategy]
    complexity: QueryComplexity
    domain: QueryDomain
    intent: str
    expected_quality_score: float
    max_response_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """Result of a single benchmark query"""
    query: str
    strategy_used: RAGStrategy
    response_time: float
    quality_score: float
    memory_usage: int
    success: bool
    error: Optional[str] = None
    response: Optional[RAGResponse] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy"""
    strategy: RAGStrategy
    total_queries: int
    success_rate: float
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    avg_quality_score: float
    avg_memory_usage: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report"""
    benchmark_id: str
    benchmark_type: BenchmarkType
    workload_type: WorkloadType
    start_time: datetime
    end_time: datetime
    total_duration: float
    total_queries: int
    overall_success_rate: float
    strategy_metrics: Dict[RAGStrategy, StrategyMetrics]
    regressions: List['RegressionDetection']
    performance_trends: Dict[str, Any]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegressionDetection:
    """Regression detection result"""
    metric_name: str
    strategy: Optional[RAGStrategy]
    previous_value: float
    current_value: float
    change_percent: float
    severity: RegressionSeverity
    threshold_breached: str
    detected_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

class BenchmarkWorkload:
    """Base class for benchmark workloads"""

    def __init__(self, name: str, workload_type: WorkloadType):
        self.name = name
        self.workload_type = workload_type
        self.queries: List[BenchmarkQuery] = []

    def add_query(self, query: BenchmarkQuery):
        """Add query to workload"""
        self.queries.append(query)

    def get_queries(self) -> List[BenchmarkQuery]:
        """Get all queries in workload"""
        return self.queries.copy()

class SimpleQueryWorkload(BenchmarkWorkload):
    """Simple query workload for basic testing"""

    def __init__(self):
        super().__init__("Simple Queries", WorkloadType.SIMPLE_QUERIES)
        self._create_simple_queries()

    def _create_simple_queries(self):
        """Create simple benchmark queries"""
        simple_queries = [
            ("What is machine learning?", QueryDomain.GENERAL, "factual"),
            ("Define neural networks", QueryDomain.GENERAL, "factual"),
            ("What is AI?", QueryDomain.GENERAL, "factual"),
            ("Explain deep learning", QueryDomain.GENERAL, "factual"),
            ("What is data science?", QueryDomain.GENERAL, "factual"),
        ]

        for query_text, domain, intent in simple_queries:
            self.add_query(BenchmarkQuery(
                query=query_text,
                expected_strategy=None,
                complexity=QueryComplexity.SIMPLE,
                domain=domain,
                intent=intent,
                expected_quality_score=0.7,
                max_response_time=2.0
            ))

class ComplexQueryWorkload(BenchmarkWorkload):
    """Complex query workload for advanced testing"""

    def __init__(self):
        super().__init__("Complex Queries", WorkloadType.COMPLEX_QUERIES)
        self._create_complex_queries()

    def _create_complex_queries(self):
        """Create complex benchmark queries"""
        complex_queries = [
            (
                "Analyze the relationship between quantum computing and machine learning optimization algorithms",
                QueryDomain.QUANTUM_ML,
                "synthesis"
            ),
            (
                "Compare the effectiveness of different neural network architectures for image recognition",
                QueryDomain.GENERAL,
                "comparative"
            ),
            (
                "Explain the theoretical foundations of variational quantum algorithms and their advantages",
                QueryDomain.QUANTUM_ML,
                "synthesis"
            ),
            (
                "Evaluate the impact of developmental disorders on learning processes and intervention strategies",
                QueryDomain.DEVELOPMENTAL_DISORDERS,
                "synthesis"
            ),
            (
                "Describe the neurobiological mechanisms underlying fMRI signal generation and interpretation",
                QueryDomain.NEUROSCIENCE,
                "causal"
            ),
        ]

        for query_text, domain, intent in complex_queries:
            self.add_query(BenchmarkQuery(
                query=query_text,
                expected_strategy=None,
                complexity=QueryComplexity.COMPLEX,
                domain=domain,
                intent=intent,
                expected_quality_score=0.8,
                max_response_time=5.0
            ))

class MixedWorkload(BenchmarkWorkload):
    """Mixed workload combining simple and complex queries"""

    def __init__(self):
        super().__init__("Mixed Workload", WorkloadType.MIXED_WORKLOAD)

        # Add simple queries
        simple_workload = SimpleQueryWorkload()
        self.queries.extend(simple_workload.get_queries())

        # Add complex queries
        complex_workload = ComplexQueryWorkload()
        self.queries.extend(complex_workload.get_queries())

        # Add medium complexity queries
        self._add_medium_queries()

    def _add_medium_queries(self):
        """Add medium complexity queries"""
        medium_queries = [
            ("How do convolutional neural networks work?", QueryDomain.GENERAL, "procedural"),
            ("What are the symptoms of autism spectrum disorder?", QueryDomain.DEVELOPMENTAL_DISORDERS, "factual"),
            ("Explain quantum entanglement in simple terms", QueryDomain.QUANTUM_ML, "factual"),
            ("How does fMRI measure brain activity?", QueryDomain.NEUROSCIENCE, "procedural"),
        ]

        for query_text, domain, intent in medium_queries:
            self.add_query(BenchmarkQuery(
                query=query_text,
                expected_strategy=None,
                complexity=QueryComplexity.MEDIUM,
                domain=domain,
                intent=intent,
                expected_quality_score=0.75,
                max_response_time=3.5
            ))

class PerformanceBenchmark:
    """Main performance benchmark system"""

    def __init__(
        self,
        orchestrator: UnifiedRAGOrchestrator,
        classifier: Optional[MLQueryClassifier] = None
    ):
        self.orchestrator = orchestrator
        self.classifier = classifier

        # Benchmark configuration
        self.regression_thresholds = {
            "response_time": {"moderate": 20.0, "major": 50.0, "critical": 100.0},  # % increase
            "quality_score": {"moderate": 10.0, "major": 20.0, "critical": 35.0},  # % decrease
            "success_rate": {"moderate": 5.0, "major": 15.0, "critical": 30.0},    # % decrease
            "memory_usage": {"moderate": 25.0, "major": 50.0, "critical": 100.0}   # % increase
        }

        # Historical data storage
        self.historical_results: List[BenchmarkReport] = []
        self.baseline_metrics: Optional[Dict[RAGStrategy, StrategyMetrics]] = None

        # Performance tracking
        self.current_benchmark: Optional[BenchmarkReport] = None

    async def run_benchmark(
        self,
        workload: BenchmarkWorkload,
        benchmark_type: BenchmarkType = BenchmarkType.COMPREHENSIVE,
        max_concurrent: int = 5,
        timeout: float = 30.0
    ) -> BenchmarkReport:
        """Run comprehensive benchmark"""
        benchmark_id = f"benchmark_{int(time.time())}"
        start_time = datetime.now()

        logger.info(f"Starting benchmark {benchmark_id} with workload: {workload.name}")

        try:
            # Initialize benchmark report
            report = BenchmarkReport(
                benchmark_id=benchmark_id,
                benchmark_type=benchmark_type,
                workload_type=workload.workload_type,
                start_time=start_time,
                end_time=start_time,  # Will be updated
                total_duration=0.0,
                total_queries=len(workload.queries),
                overall_success_rate=0.0,
                strategy_metrics={},
                regressions=[],
                performance_trends={},
                recommendations=[]
            )

            # Run benchmark queries
            results = await self._execute_queries(
                workload.queries,
                max_concurrent=max_concurrent,
                timeout=timeout
            )

            # Calculate metrics
            strategy_metrics = self._calculate_strategy_metrics(results)
            report.strategy_metrics = strategy_metrics

            # Calculate overall success rate
            successful_queries = sum(1 for r in results if r.success)
            report.overall_success_rate = successful_queries / len(results) if results else 0.0

            # Update timing
            end_time = datetime.now()
            report.end_time = end_time
            report.total_duration = (end_time - start_time).total_seconds()

            # Detect regressions
            if self.baseline_metrics:
                report.regressions = self._detect_regressions(strategy_metrics, self.baseline_metrics)

            # Generate performance trends
            report.performance_trends = self._analyze_performance_trends(results)

            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)

            # Store results
            self.historical_results.append(report)
            self.current_benchmark = report

            logger.info(f"Benchmark {benchmark_id} completed in {report.total_duration:.2f}s")
            return report

        except Exception as e:
            logger.error(f"Benchmark {benchmark_id} failed: {e}")
            raise

    async def _execute_queries(
        self,
        queries: List[BenchmarkQuery],
        max_concurrent: int,
        timeout: float
    ) -> List[BenchmarkResult]:
        """Execute benchmark queries with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = []

        for query in queries:
            task = asyncio.create_task(self._execute_single_query(query, semaphore, timeout))
            tasks.append(task)

        # Wait for all queries to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and convert to BenchmarkResult
        benchmark_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                benchmark_results.append(BenchmarkResult(
                    query=queries[i].query,
                    strategy_used=RAGStrategy.SIMPLE_RAG,  # Default
                    response_time=timeout,
                    quality_score=0.0,
                    memory_usage=0,
                    success=False,
                    error=str(result)
                ))
            else:
                benchmark_results.append(result)

        return benchmark_results

    async def _execute_single_query(
        self,
        query: BenchmarkQuery,
        semaphore: asyncio.Semaphore,
        timeout: float
    ) -> BenchmarkResult:
        """Execute a single benchmark query"""
        async with semaphore:
            start_time = time.time()
            memory_before = self._get_memory_usage()

            try:
                # Create query context
                if self.classifier:
                    classification = await self.classifier.classify(query.query)
                    query_context = QueryContext(
                        query=query.query,
                        complexity=classification.complexity,
                        domain=classification.domain,
                        intent=classification.intent.value,
                        confidence=classification.overall_confidence,
                        metadata=query.metadata
                    )
                else:
                    query_context = QueryContext(
                        query=query.query,
                        complexity=query.complexity,
                        domain=query.domain,
                        intent=query.intent,
                        confidence=0.8,
                        metadata=query.metadata
                    )

                # Execute query with timeout
                response = await asyncio.wait_for(
                    self.orchestrator.search(query_context),
                    timeout=timeout
                )

                # Calculate metrics
                response_time = time.time() - start_time
                memory_after = self._get_memory_usage()
                memory_usage = memory_after - memory_before

                # Calculate quality score (simplified)
                quality_score = self._calculate_quality_score(response, query)

                return BenchmarkResult(
                    query=query.query,
                    strategy_used=response.strategy_used,
                    response_time=response_time,
                    quality_score=quality_score,
                    memory_usage=memory_usage,
                    success=True,
                    response=response
                )

            except asyncio.TimeoutError:
                return BenchmarkResult(
                    query=query.query,
                    strategy_used=RAGStrategy.SIMPLE_RAG,  # Default
                    response_time=timeout,
                    quality_score=0.0,
                    memory_usage=0,
                    success=False,
                    error="Query timeout"
                )

            except Exception as e:
                response_time = time.time() - start_time
                return BenchmarkResult(
                    query=query.query,
                    strategy_used=RAGStrategy.SIMPLE_RAG,  # Default
                    response_time=response_time,
                    quality_score=0.0,
                    memory_usage=0,
                    success=False,
                    error=str(e)
                )

    def _get_memory_usage(self) -> int:
        """Get current memory usage (simplified)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            # Fallback to basic estimation
            return 0

    def _calculate_quality_score(self, response: RAGResponse, query: BenchmarkQuery) -> float:
        """Calculate quality score for response"""
        try:
            # Use response confidence as base score
            base_score = response.confidence

            # Adjust based on response length (longer responses may be more comprehensive)
            if response.answer:
                length_factor = min(1.0, len(response.answer) / 500)  # Normalize around 500 chars
                base_score = (base_score * 0.8) + (length_factor * 0.2)

            # Adjust based on number of sources
            if response.sources:
                source_factor = min(1.0, len(response.sources) / 3)  # Normalize around 3 sources
                base_score = (base_score * 0.9) + (source_factor * 0.1)

            return min(1.0, max(0.0, base_score))

        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5

    def _calculate_strategy_metrics(self, results: List[BenchmarkResult]) -> Dict[RAGStrategy, StrategyMetrics]:
        """Calculate metrics for each strategy"""
        strategy_results = {}

        # Group results by strategy
        for result in results:
            strategy = result.strategy_used
            if strategy not in strategy_results:
                strategy_results[strategy] = []
            strategy_results[strategy].append(result)

        # Calculate metrics for each strategy
        metrics = {}
        for strategy, strategy_res in strategy_results.items():
            successful_results = [r for r in strategy_res if r.success]

            if not strategy_res:
                continue

            # Calculate response time percentiles
            response_times = [r.response_time for r in successful_results]
            if response_times:
                p50_time = statistics.median(response_times)
                p95_time = self._calculate_percentile(response_times, 95)
                p99_time = self._calculate_percentile(response_times, 99)
                avg_time = statistics.mean(response_times)
            else:
                p50_time = p95_time = p99_time = avg_time = 0.0

            # Calculate other metrics
            success_rate = len(successful_results) / len(strategy_res)
            avg_quality = statistics.mean([r.quality_score for r in successful_results]) if successful_results else 0.0
            avg_memory = statistics.mean([r.memory_usage for r in successful_results]) if successful_results else 0.0

            # Collect errors
            errors = [r.error for r in strategy_res if r.error]

            metrics[strategy] = StrategyMetrics(
                strategy=strategy,
                total_queries=len(strategy_res),
                success_rate=success_rate,
                avg_response_time=avg_time,
                p50_response_time=p50_time,
                p95_response_time=p95_time,
                p99_response_time=p99_time,
                avg_quality_score=avg_quality,
                avg_memory_usage=avg_memory,
                errors=errors[:10]  # Limit to first 10 errors
            )

        return metrics

    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)

        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(index)
            upper_index = min(lower_index + 1, len(sorted_values) - 1)
            weight = index - lower_index
            return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight

    def _detect_regressions(
        self,
        current_metrics: Dict[RAGStrategy, StrategyMetrics],
        baseline_metrics: Dict[RAGStrategy, StrategyMetrics]
    ) -> List[RegressionDetection]:
        """Detect performance regressions"""
        regressions = []

        for strategy in current_metrics:
            if strategy not in baseline_metrics:
                continue

            current = current_metrics[strategy]
            baseline = baseline_metrics[strategy]

            # Check response time regression
            if baseline.avg_response_time > 0:
                time_change = ((current.avg_response_time - baseline.avg_response_time) / baseline.avg_response_time) * 100
                if time_change > self.regression_thresholds["response_time"]["moderate"]:
                    severity = self._determine_severity(time_change, self.regression_thresholds["response_time"])
                    regressions.append(RegressionDetection(
                        metric_name="avg_response_time",
                        strategy=strategy,
                        previous_value=baseline.avg_response_time,
                        current_value=current.avg_response_time,
                        change_percent=time_change,
                        severity=severity,
                        threshold_breached=f"{severity.value}_threshold"
                    ))

            # Check quality score regression
            if baseline.avg_quality_score > 0:
                quality_change = ((baseline.avg_quality_score - current.avg_quality_score) / baseline.avg_quality_score) * 100
                if quality_change > self.regression_thresholds["quality_score"]["moderate"]:
                    severity = self._determine_severity(quality_change, self.regression_thresholds["quality_score"])
                    regressions.append(RegressionDetection(
                        metric_name="avg_quality_score",
                        strategy=strategy,
                        previous_value=baseline.avg_quality_score,
                        current_value=current.avg_quality_score,
                        change_percent=-quality_change,  # Negative because lower is worse
                        severity=severity,
                        threshold_breached=f"{severity.value}_threshold"
                    ))

            # Check success rate regression
            if baseline.success_rate > 0:
                success_change = ((baseline.success_rate - current.success_rate) / baseline.success_rate) * 100
                if success_change > self.regression_thresholds["success_rate"]["moderate"]:
                    severity = self._determine_severity(success_change, self.regression_thresholds["success_rate"])
                    regressions.append(RegressionDetection(
                        metric_name="success_rate",
                        strategy=strategy,
                        previous_value=baseline.success_rate,
                        current_value=current.success_rate,
                        change_percent=-success_change,  # Negative because lower is worse
                        severity=severity,
                        threshold_breached=f"{severity.value}_threshold"
                    ))

        return regressions

    def _determine_severity(self, change_percent: float, thresholds: Dict[str, float]) -> RegressionSeverity:
        """Determine regression severity based on change percentage"""
        abs_change = abs(change_percent)

        if abs_change >= thresholds["critical"]:
            return RegressionSeverity.CRITICAL
        elif abs_change >= thresholds["major"]:
            return RegressionSeverity.MAJOR
        elif abs_change >= thresholds["moderate"]:
            return RegressionSeverity.MODERATE
        else:
            return RegressionSeverity.MINOR

    def _analyze_performance_trends(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance trends across queries"""
        trends = {}

        try:
            # Response time trends
            response_times = [r.response_time for r in results if r.success]
            if response_times:
                trends["response_time"] = {
                    "mean": statistics.mean(response_times),
                    "median": statistics.median(response_times),
                    "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0.0,
                    "min": min(response_times),
                    "max": max(response_times)
                }

            # Quality trends
            quality_scores = [r.quality_score for r in results if r.success]
            if quality_scores:
                trends["quality_score"] = {
                    "mean": statistics.mean(quality_scores),
                    "median": statistics.median(quality_scores),
                    "std_dev": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0,
                    "min": min(quality_scores),
                    "max": max(quality_scores)
                }

            # Error analysis
            errors = [r.error for r in results if r.error]
            error_counts = {}
            for error in errors:
                error_type = error.split(":")[0] if ":" in error else error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

            trends["errors"] = {
                "total_errors": len(errors),
                "error_rate": len(errors) / len(results) if results else 0.0,
                "error_types": error_counts
            }

        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
            trends["error"] = str(e)

        return trends

    def _generate_recommendations(self, report: BenchmarkReport) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        try:
            # Check overall success rate
            if report.overall_success_rate < 0.9:
                recommendations.append(f"Overall success rate is {report.overall_success_rate:.1%}, consider investigating error causes")

            # Check for slow strategies
            for strategy, metrics in report.strategy_metrics.items():
                if metrics.avg_response_time > 5.0:
                    recommendations.append(f"{strategy.value} has high average response time ({metrics.avg_response_time:.2f}s)")

                if metrics.success_rate < 0.9:
                    recommendations.append(f"{strategy.value} has low success rate ({metrics.success_rate:.1%})")

                if metrics.avg_quality_score < 0.7:
                    recommendations.append(f"{strategy.value} has low quality scores ({metrics.avg_quality_score:.2f})")

            # Check for regressions
            critical_regressions = [r for r in report.regressions if r.severity == RegressionSeverity.CRITICAL]
            if critical_regressions:
                recommendations.append(f"CRITICAL: {len(critical_regressions)} critical regressions detected")

            major_regressions = [r for r in report.regressions if r.severity == RegressionSeverity.MAJOR]
            if major_regressions:
                recommendations.append(f"WARNING: {len(major_regressions)} major regressions detected")

            # Performance trends analysis
            if "response_time" in report.performance_trends:
                rt_trends = report.performance_trends["response_time"]
                if rt_trends.get("std_dev", 0) > rt_trends.get("mean", 0) * 0.5:
                    recommendations.append("High response time variability detected, consider optimization")

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to error")

        return recommendations[:10]  # Limit to top 10 recommendations

    async def set_baseline(self, workload: BenchmarkWorkload) -> BenchmarkReport:
        """Set performance baseline by running benchmark"""
        logger.info("Setting performance baseline")
        report = await self.run_benchmark(workload, BenchmarkType.COMPREHENSIVE)
        self.baseline_metrics = report.strategy_metrics.copy()
        logger.info(f"Baseline set with {len(self.baseline_metrics)} strategies")
        return report

    async def run_regression_test(self, workload: BenchmarkWorkload) -> BenchmarkReport:
        """Run regression test against baseline"""
        if not self.baseline_metrics:
            raise ValueError("No baseline metrics set. Run set_baseline() first.")

        logger.info("Running regression test against baseline")
        return await self.run_benchmark(workload, BenchmarkType.COMPREHENSIVE)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all benchmarks"""
        if not self.historical_results:
            return {"error": "No benchmark results available"}

        total_benchmarks = len(self.historical_results)
        total_queries = sum(r.total_queries for r in self.historical_results)
        avg_success_rate = statistics.mean([r.overall_success_rate for r in self.historical_results])

        # Find best and worst performing strategies
        strategy_performances = {}
        for report in self.historical_results:
            for strategy, metrics in report.strategy_metrics.items():
                if strategy not in strategy_performances:
                    strategy_performances[strategy] = []
                strategy_performances[strategy].append(metrics.avg_response_time)

        best_strategy = None
        worst_strategy = None
        if strategy_performances:
            avg_times = {s: statistics.mean(times) for s, times in strategy_performances.items()}
            best_strategy = min(avg_times.keys(), key=lambda x: avg_times[x])
            worst_strategy = max(avg_times.keys(), key=lambda x: avg_times[x])

        # Count regressions
        total_regressions = sum(len(r.regressions) for r in self.historical_results)
        critical_regressions = sum(
            len([reg for reg in r.regressions if reg.severity == RegressionSeverity.CRITICAL])
            for r in self.historical_results
        )

        return {
            "total_benchmarks": total_benchmarks,
            "total_queries_tested": total_queries,
            "avg_success_rate": avg_success_rate,
            "best_performing_strategy": best_strategy.value if best_strategy else None,
            "worst_performing_strategy": worst_strategy.value if worst_strategy else None,
            "total_regressions": total_regressions,
            "critical_regressions": critical_regressions,
            "latest_benchmark": self.current_benchmark.benchmark_id if self.current_benchmark else None,
            "baseline_set": self.baseline_metrics is not None
        }

    def export_results(self, filepath: str, format: str = "json"):
        """Export benchmark results to file"""
        try:
            if format == "json":
                with open(filepath, 'w') as f:
                    # Convert dataclasses to dict for JSON serialization
                    serializable_results = []
                    for report in self.historical_results:
                        report_dict = asdict(report)
                        # Convert datetime objects to ISO format
                        report_dict['start_time'] = report.start_time.isoformat()
                        report_dict['end_time'] = report.end_time.isoformat()

                        # Convert enum keys to strings
                        strategy_metrics = {}
                        for strategy, metrics in report.strategy_metrics.items():
                            strategy_metrics[strategy.value] = asdict(metrics)
                            strategy_metrics[strategy.value]['strategy'] = strategy.value

                        report_dict['strategy_metrics'] = strategy_metrics
                        serializable_results.append(report_dict)

                    json.dump(serializable_results, f, indent=2)

            elif format == "csv" and PANDAS_AVAILABLE:
                # Create DataFrame from results
                rows = []
                for report in self.historical_results:
                    for strategy, metrics in report.strategy_metrics.items():
                        rows.append({
                            'benchmark_id': report.benchmark_id,
                            'benchmark_type': report.benchmark_type.value,
                            'workload_type': report.workload_type.value,
                            'strategy': strategy.value,
                            'total_queries': metrics.total_queries,
                            'success_rate': metrics.success_rate,
                            'avg_response_time': metrics.avg_response_time,
                            'avg_quality_score': metrics.avg_quality_score,
                            'total_duration': report.total_duration,
                            'overall_success_rate': report.overall_success_rate
                        })

                df = pd.DataFrame(rows)
                df.to_csv(filepath, index=False)

            logger.info(f"Exported benchmark results to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export results: {e}")

def create_performance_benchmark(
    orchestrator: UnifiedRAGOrchestrator,
    classifier: Optional[MLQueryClassifier] = None
) -> PerformanceBenchmark:
    """Factory function to create performance benchmark"""
    return PerformanceBenchmark(orchestrator, classifier)

# Example usage
if __name__ == "__main__":
    async def test_performance_benchmark():
        """Test performance benchmark system"""
        # Would require actual orchestrator instance
        # This is a demonstration of the API

        from ..rag.unified_rag_orchestrator import create_unified_orchestrator

        orchestrator = create_unified_orchestrator()
        benchmark = create_performance_benchmark(orchestrator)

        # Create workload
        workload = MixedWorkload()

        try:
            # Set baseline
            baseline_report = await benchmark.set_baseline(workload)
            print(f"Baseline set: {baseline_report.overall_success_rate:.1%} success rate")

            # Run regression test
            regression_report = await benchmark.run_regression_test(workload)
            print(f"Regression test: {len(regression_report.regressions)} regressions detected")

            # Get summary
            summary = benchmark.get_performance_summary()
            print(f"Performance Summary: {summary}")

        finally:
            orchestrator.shutdown()

    # Run test
    # asyncio.run(test_performance_benchmark())