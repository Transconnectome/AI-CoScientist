"""
RL Agent Selection Performance Monitoring

This module provides comprehensive performance monitoring for RL-enhanced agent
selection, including:

- Real-time metrics collection and aggregation
- Performance dashboards and alerting
- A/B test analysis and statistical significance testing
- Model drift detection and performance degradation alerts
- Integration with existing monitoring infrastructure (Prometheus, etc.)

The monitoring system tracks key metrics like selection accuracy, latency,
quality scores, and resource utilization to ensure the RL system is
performing optimally in production.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    # Try to import Prometheus metrics
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available, using in-memory metrics only")

try:
    # Try to import scipy for statistical analysis
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available, limited statistical analysis")

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Performance monitoring alert"""
    level: AlertLevel
    metric: str
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    strategy: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricThresholds:
    """Thresholds for performance alerts"""
    success_rate_warning: float = 0.85
    success_rate_critical: float = 0.75
    latency_p95_warning_ms: float = 2000.0
    latency_p95_critical_ms: float = 5000.0
    quality_score_warning: float = 0.7
    quality_score_critical: float = 0.6
    confidence_warning: float = 0.6
    confidence_critical: float = 0.4


@dataclass
class RealTimeMetrics:
    """Real-time performance metrics"""
    timestamp: datetime
    strategy: str
    success_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_quality_score: float
    avg_confidence: float
    total_selections: int
    error_rate: float
    agent_distribution: Dict[str, int]
    task_type_distribution: Dict[str, int]


class StatisticalAnalyzer:
    """Statistical analysis for A/B testing and performance comparison"""

    @staticmethod
    def calculate_significance(
        group_a_metrics: List[float],
        group_b_metrics: List[float],
        alpha: float = 0.05
    ) -> Tuple[float, bool]:
        """
        Calculate statistical significance between two metric groups

        Returns (p_value, is_significant)
        """
        if not SCIPY_AVAILABLE or len(group_a_metrics) < 10 or len(group_b_metrics) < 10:
            # Fallback to simple comparison
            avg_a = statistics.mean(group_a_metrics) if group_a_metrics else 0
            avg_b = statistics.mean(group_b_metrics) if group_b_metrics else 0
            difference_ratio = abs(avg_a - avg_b) / max(avg_a, avg_b, 1e-6)
            return difference_ratio, difference_ratio > 0.1  # 10% difference threshold

        try:
            # Use t-test for significance
            t_stat, p_value = stats.ttest_ind(group_a_metrics, group_b_metrics)
            is_significant = p_value < alpha
            return p_value, is_significant
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return 1.0, False

    @staticmethod
    def detect_trend(metrics: List[Tuple[datetime, float]], window_hours: int = 24) -> Dict[str, Any]:
        """
        Detect trends in metric values over time

        Returns trend analysis including slope, direction, and confidence
        """
        if len(metrics) < 10:
            return {"status": "insufficient_data"}

        # Filter to time window
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent_metrics = [(t, v) for t, v in metrics if t > cutoff]

        if len(recent_metrics) < 5:
            return {"status": "insufficient_recent_data"}

        # Convert to numeric for analysis
        timestamps = [m[0].timestamp() for m in recent_metrics]
        values = [m[1] for m in recent_metrics]

        try:
            if SCIPY_AVAILABLE:
                slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, values)
                confidence = abs(r_value)
            else:
                # Simple slope calculation
                x_mean = statistics.mean(timestamps)
                y_mean = statistics.mean(values)
                numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(timestamps, values))
                denominator = sum((x - x_mean) ** 2 for x in timestamps)
                slope = numerator / denominator if denominator != 0 else 0
                confidence = 0.5  # Default confidence

            # Determine trend direction
            if abs(slope) < 1e-6:
                direction = "stable"
            elif slope > 0:
                direction = "increasing"
            else:
                direction = "decreasing"

            return {
                "status": "success",
                "slope": slope,
                "direction": direction,
                "confidence": confidence,
                "data_points": len(recent_metrics),
                "time_span_hours": (timestamps[-1] - timestamps[0]) / 3600
            }

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {"status": "error", "error": str(e)}


class PrometheusMetrics:
    """Prometheus metrics collection for RL agent selection"""

    def __init__(self, registry=None):
        self.registry = registry or CollectorRegistry()

        if PROMETHEUS_AVAILABLE:
            # Selection metrics
            self.selection_total = Counter(
                'rl_agent_selection_total',
                'Total number of agent selections',
                ['strategy', 'task_type'],
                registry=self.registry
            )

            self.selection_duration = Histogram(
                'rl_agent_selection_duration_seconds',
                'Agent selection duration',
                ['strategy'],
                registry=self.registry
            )

            self.selection_success_rate = Gauge(
                'rl_agent_selection_success_rate',
                'Agent selection success rate',
                ['strategy', 'window'],
                registry=self.registry
            )

            self.quality_score = Histogram(
                'rl_agent_selection_quality_score',
                'Quality score of agent selections',
                ['strategy'],
                registry=self.registry
            )

            self.confidence_score = Histogram(
                'rl_agent_selection_confidence',
                'Confidence score of agent selections',
                ['strategy'],
                registry=self.registry
            )

            # Agent usage metrics
            self.agent_usage = Counter(
                'rl_agent_usage_total',
                'Usage count per agent',
                ['agent_id', 'strategy'],
                registry=self.registry
            )

            # Error metrics
            self.selection_errors = Counter(
                'rl_agent_selection_errors_total',
                'Number of selection errors',
                ['strategy', 'error_type'],
                registry=self.registry
            )

        else:
            # Mock metrics for when Prometheus not available
            self._mock_metrics = defaultdict(float)

    def record_selection(self, strategy: str, task_type: str, duration: float):
        """Record an agent selection event"""
        if PROMETHEUS_AVAILABLE:
            self.selection_total.labels(strategy=strategy, task_type=task_type).inc()
            self.selection_duration.labels(strategy=strategy).observe(duration)
        else:
            self._mock_metrics[f'selections_{strategy}_{task_type}'] += 1

    def record_quality(self, strategy: str, quality_score: float):
        """Record quality score"""
        if PROMETHEUS_AVAILABLE:
            self.quality_score.labels(strategy=strategy).observe(quality_score)

    def record_confidence(self, strategy: str, confidence: float):
        """Record confidence score"""
        if PROMETHEUS_AVAILABLE:
            self.confidence_score.labels(strategy=strategy).observe(confidence)

    def record_agent_usage(self, agent_id: str, strategy: str):
        """Record agent usage"""
        if PROMETHEUS_AVAILABLE:
            self.agent_usage.labels(agent_id=agent_id, strategy=strategy).inc()

    def record_error(self, strategy: str, error_type: str):
        """Record selection error"""
        if PROMETHEUS_AVAILABLE:
            self.selection_errors.labels(strategy=strategy, error_type=error_type).inc()

    def update_success_rate(self, strategy: str, window: str, rate: float):
        """Update success rate gauge"""
        if PROMETHEUS_AVAILABLE:
            self.success_rate.labels(strategy=strategy, window=window).set(rate)

    def get_metrics_text(self) -> str:
        """Get Prometheus metrics in text format"""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry).decode('utf-8')
        else:
            return "# Prometheus not available\n"


class RLPerformanceMonitor:
    """
    Comprehensive performance monitoring for RL agent selection

    Features:
    - Real-time metrics collection and aggregation
    - Performance alerting with configurable thresholds
    - A/B test analysis and statistical significance testing
    - Trend analysis and drift detection
    - Prometheus integration for production monitoring
    """

    def __init__(self,
                 thresholds: Optional[MetricThresholds] = None,
                 enable_prometheus: bool = True):
        """
        Initialize performance monitor

        Args:
            thresholds: Alert thresholds configuration
            enable_prometheus: Whether to enable Prometheus metrics
        """
        self.thresholds = thresholds or MetricThresholds()
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE

        # Metrics storage
        self.selection_history: deque = deque(maxlen=10000)  # Store last 10k selections
        self.alerts: deque = deque(maxlen=1000)  # Store last 1k alerts
        self.real_time_cache: Dict[str, RealTimeMetrics] = {}

        # Statistical analyzer
        self.stats_analyzer = StatisticalAnalyzer()

        # Prometheus metrics
        if self.enable_prometheus:
            self.prometheus_metrics = PrometheusMetrics()
        else:
            self.prometheus_metrics = None

        # Background processing
        self._processing_lock = threading.RLock()
        self._background_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        logger.info(f"RL Performance Monitor initialized - Prometheus: {self.enable_prometheus}")

    def record_selection_event(self,
                             strategy: str,
                             agent_ids: List[str],
                             task_type: str,
                             selection_time: float,
                             confidence: float,
                             success: Optional[bool] = None,
                             quality_score: Optional[float] = None):
        """
        Record an agent selection event

        Args:
            strategy: Selection strategy used (rl_enabled, traditional, etc.)
            agent_ids: List of selected agent IDs
            task_type: Type of task (simple, complex, comprehensive)
            selection_time: Time taken for selection in seconds
            confidence: Confidence score of the selection
            success: Whether the selection was successful (if known)
            quality_score: Quality score of the selection outcome
        """
        timestamp = datetime.now()

        # Store in history
        event = {
            'timestamp': timestamp,
            'strategy': strategy,
            'agent_ids': agent_ids,
            'task_type': task_type,
            'selection_time': selection_time,
            'confidence': confidence,
            'success': success,
            'quality_score': quality_score
        }

        with self._processing_lock:
            self.selection_history.append(event)

        # Update Prometheus metrics
        if self.prometheus_metrics:
            self.prometheus_metrics.record_selection(strategy, task_type, selection_time)
            self.prometheus_metrics.record_confidence(strategy, confidence)

            for agent_id in agent_ids:
                self.prometheus_metrics.record_agent_usage(agent_id, strategy)

            if quality_score is not None:
                self.prometheus_metrics.record_quality(strategy, quality_score)

        # Update real-time cache
        self._update_real_time_metrics()

        # Check for alerts
        self._check_alerts()

    def record_selection_error(self, strategy: str, error_type: str, error_message: str):
        """Record a selection error"""
        timestamp = datetime.now()

        error_event = {
            'timestamp': timestamp,
            'strategy': strategy,
            'error_type': error_type,
            'error_message': error_message,
            'event_type': 'error'
        }

        with self._processing_lock:
            self.selection_history.append(error_event)

        if self.prometheus_metrics:
            self.prometheus_metrics.record_error(strategy, error_type)

        # Generate error alert
        alert = PerformanceAlert(
            level=AlertLevel.ERROR,
            metric='selection_error',
            message=f"Selection error in {strategy}: {error_message}",
            value=1.0,
            threshold=0.0,
            strategy=strategy,
            additional_data={'error_type': error_type}
        )

        self._add_alert(alert)

    def _update_real_time_metrics(self):
        """Update real-time metrics cache"""
        if not self.selection_history:
            return

        # Calculate metrics for different time windows
        windows = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '1h': timedelta(hours=1)
        }

        for window_name, window_duration in windows.items():
            cutoff = datetime.now() - window_duration

            # Group events by strategy
            strategy_events = defaultdict(list)
            for event in self.selection_history:
                if event.get('timestamp', datetime.min) > cutoff:
                    strategy = event.get('strategy', 'unknown')
                    strategy_events[strategy].append(event)

            # Calculate metrics for each strategy
            for strategy, events in strategy_events.items():
                if not events:
                    continue

                metrics = self._calculate_strategy_metrics(events)
                cache_key = f"{strategy}_{window_name}"
                self.real_time_cache[cache_key] = metrics

    def _calculate_strategy_metrics(self, events: List[Dict]) -> RealTimeMetrics:
        """Calculate performance metrics for a list of events"""
        if not events:
            return RealTimeMetrics(
                timestamp=datetime.now(),
                strategy="unknown",
                success_rate=0.0,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                avg_quality_score=0.0,
                avg_confidence=0.0,
                total_selections=0,
                error_rate=0.0,
                agent_distribution={},
                task_type_distribution={}
            )

        # Filter selection events (exclude errors)
        selection_events = [e for e in events if e.get('event_type') != 'error']
        error_events = [e for e in events if e.get('event_type') == 'error']

        if not selection_events:
            return RealTimeMetrics(
                timestamp=datetime.now(),
                strategy=events[0].get('strategy', 'unknown'),
                success_rate=0.0,
                avg_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                avg_quality_score=0.0,
                avg_confidence=0.0,
                total_selections=0,
                error_rate=1.0,
                agent_distribution={},
                task_type_distribution={}
            )

        # Calculate success rate
        success_events = [e for e in selection_events if e.get('success') is True]
        success_rate = len(success_events) / len(selection_events) if selection_events else 0.0

        # Calculate latency metrics
        latencies = [e.get('selection_time', 0) * 1000 for e in selection_events]  # Convert to ms
        avg_latency_ms = statistics.mean(latencies) if latencies else 0.0

        if len(latencies) > 1:
            p95_latency_ms = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            p99_latency_ms = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 10 else max(latencies)
        else:
            p95_latency_ms = p99_latency_ms = avg_latency_ms

        # Calculate quality and confidence
        quality_scores = [e.get('quality_score', 0) for e in selection_events if e.get('quality_score') is not None]
        avg_quality_score = statistics.mean(quality_scores) if quality_scores else 0.0

        confidences = [e.get('confidence', 0) for e in selection_events]
        avg_confidence = statistics.mean(confidences) if confidences else 0.0

        # Calculate error rate
        total_events = len(selection_events) + len(error_events)
        error_rate = len(error_events) / total_events if total_events > 0 else 0.0

        # Agent distribution
        agent_distribution = defaultdict(int)
        for event in selection_events:
            for agent_id in event.get('agent_ids', []):
                agent_distribution[agent_id] += 1

        # Task type distribution
        task_type_distribution = defaultdict(int)
        for event in selection_events:
            task_type = event.get('task_type', 'unknown')
            task_type_distribution[task_type] += 1

        return RealTimeMetrics(
            timestamp=datetime.now(),
            strategy=selection_events[0].get('strategy', 'unknown'),
            success_rate=success_rate,
            avg_latency_ms=avg_latency_ms,
            p95_latency_ms=p95_latency_ms,
            p99_latency_ms=p99_latency_ms,
            avg_quality_score=avg_quality_score,
            avg_confidence=avg_confidence,
            total_selections=len(selection_events),
            error_rate=error_rate,
            agent_distribution=dict(agent_distribution),
            task_type_distribution=dict(task_type_distribution)
        )

    def _check_alerts(self):
        """Check current metrics against thresholds and generate alerts"""
        for cache_key, metrics in self.real_time_cache.items():
            strategy = metrics.strategy

            # Check success rate alerts
            if metrics.success_rate < self.thresholds.success_rate_critical:
                alert = PerformanceAlert(
                    level=AlertLevel.CRITICAL,
                    metric='success_rate',
                    message=f"Critical: {strategy} success rate {metrics.success_rate:.1%} below {self.thresholds.success_rate_critical:.1%}",
                    value=metrics.success_rate,
                    threshold=self.thresholds.success_rate_critical,
                    strategy=strategy
                )
                self._add_alert(alert)
            elif metrics.success_rate < self.thresholds.success_rate_warning:
                alert = PerformanceAlert(
                    level=AlertLevel.WARNING,
                    metric='success_rate',
                    message=f"Warning: {strategy} success rate {metrics.success_rate:.1%} below {self.thresholds.success_rate_warning:.1%}",
                    value=metrics.success_rate,
                    threshold=self.thresholds.success_rate_warning,
                    strategy=strategy
                )
                self._add_alert(alert)

            # Check latency alerts
            if metrics.p95_latency_ms > self.thresholds.latency_p95_critical_ms:
                alert = PerformanceAlert(
                    level=AlertLevel.CRITICAL,
                    metric='latency_p95',
                    message=f"Critical: {strategy} P95 latency {metrics.p95_latency_ms:.0f}ms above {self.thresholds.latency_p95_critical_ms:.0f}ms",
                    value=metrics.p95_latency_ms,
                    threshold=self.thresholds.latency_p95_critical_ms,
                    strategy=strategy
                )
                self._add_alert(alert)
            elif metrics.p95_latency_ms > self.thresholds.latency_p95_warning_ms:
                alert = PerformanceAlert(
                    level=AlertLevel.WARNING,
                    metric='latency_p95',
                    message=f"Warning: {strategy} P95 latency {metrics.p95_latency_ms:.0f}ms above {self.thresholds.latency_p95_warning_ms:.0f}ms",
                    value=metrics.p95_latency_ms,
                    threshold=self.thresholds.latency_p95_warning_ms,
                    strategy=strategy
                )
                self._add_alert(alert)

            # Check quality alerts
            if metrics.avg_quality_score > 0 and metrics.avg_quality_score < self.thresholds.quality_score_critical:
                alert = PerformanceAlert(
                    level=AlertLevel.CRITICAL,
                    metric='quality_score',
                    message=f"Critical: {strategy} quality score {metrics.avg_quality_score:.2f} below {self.thresholds.quality_score_critical:.2f}",
                    value=metrics.avg_quality_score,
                    threshold=self.thresholds.quality_score_critical,
                    strategy=strategy
                )
                self._add_alert(alert)

    def _add_alert(self, alert: PerformanceAlert):
        """Add an alert with deduplication"""
        # Simple deduplication: don't add the same metric alert for the same strategy within 5 minutes
        recent_alerts = [a for a in self.alerts if
                        (alert.timestamp - a.timestamp).total_seconds() < 300 and
                        a.metric == alert.metric and
                        a.strategy == alert.strategy]

        if not recent_alerts:
            with self._processing_lock:
                self.alerts.append(alert)
            logger.log(
                logging.WARNING if alert.level == AlertLevel.WARNING else logging.ERROR,
                alert.message
            )

    def get_real_time_metrics(self, strategy: str, window: str = '5m') -> Optional[RealTimeMetrics]:
        """Get real-time metrics for a strategy and time window"""
        cache_key = f"{strategy}_{window}"
        return self.real_time_cache.get(cache_key)

    def get_all_real_time_metrics(self) -> Dict[str, RealTimeMetrics]:
        """Get all cached real-time metrics"""
        return self.real_time_cache.copy()

    def get_recent_alerts(self, level: Optional[AlertLevel] = None, limit: int = 50) -> List[PerformanceAlert]:
        """Get recent alerts, optionally filtered by level"""
        alerts = list(self.alerts)

        if level:
            alerts = [a for a in alerts if a.level == level]

        # Sort by timestamp descending and limit
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        return alerts[:limit]

    def analyze_ab_test(self, strategy_a: str, strategy_b: str,
                       metric: str = 'quality_score', window_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze A/B test results between two strategies

        Args:
            strategy_a: First strategy to compare
            strategy_b: Second strategy to compare
            metric: Metric to compare ('quality_score', 'success_rate', 'latency')
            window_hours: Time window for analysis

        Returns:
            Analysis results with statistical significance
        """
        cutoff = datetime.now() - timedelta(hours=window_hours)

        # Get events for each strategy
        events_a = [e for e in self.selection_history if
                   e.get('strategy') == strategy_a and
                   e.get('timestamp', datetime.min) > cutoff and
                   e.get('event_type') != 'error']

        events_b = [e for e in self.selection_history if
                   e.get('strategy') == strategy_b and
                   e.get('timestamp', datetime.min) > cutoff and
                   e.get('event_type') != 'error']

        if not events_a or not events_b:
            return {
                "status": "insufficient_data",
                "events_a": len(events_a),
                "events_b": len(events_b),
                "minimum_required": 10
            }

        # Extract metric values
        if metric == 'quality_score':
            values_a = [e.get('quality_score') for e in events_a if e.get('quality_score') is not None]
            values_b = [e.get('quality_score') for e in events_b if e.get('quality_score') is not None]
        elif metric == 'success_rate':
            values_a = [1.0 if e.get('success') else 0.0 for e in events_a if e.get('success') is not None]
            values_b = [1.0 if e.get('success') else 0.0 for e in events_b if e.get('success') is not None]
        elif metric == 'latency':
            values_a = [e.get('selection_time', 0) for e in events_a]
            values_b = [e.get('selection_time', 0) for e in events_b]
        else:
            return {"status": "invalid_metric", "valid_metrics": ["quality_score", "success_rate", "latency"]}

        if not values_a or not values_b:
            return {
                "status": "no_metric_data",
                "metric": metric,
                "values_a_count": len(values_a),
                "values_b_count": len(values_b)
            }

        # Calculate basic statistics
        mean_a = statistics.mean(values_a)
        mean_b = statistics.mean(values_b)
        improvement = (mean_b - mean_a) / mean_a if mean_a > 0 else 0

        # Statistical significance test
        p_value, is_significant = self.stats_analyzer.calculate_significance(values_a, values_b)

        return {
            "status": "success",
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "metric": metric,
            "window_hours": window_hours,
            "sample_size_a": len(values_a),
            "sample_size_b": len(values_b),
            "mean_a": mean_a,
            "mean_b": mean_b,
            "improvement_percentage": improvement * 100,
            "p_value": p_value,
            "is_statistically_significant": is_significant,
            "confidence_level": 95,
            "recommendation": self._get_ab_test_recommendation(improvement, is_significant)
        }

    def _get_ab_test_recommendation(self, improvement: float, is_significant: bool) -> str:
        """Generate recommendation based on A/B test results"""
        if not is_significant:
            return "No statistically significant difference detected. Continue testing or maintain current strategy."

        if improvement > 0.05:  # 5% improvement
            return "Strategy B shows significant improvement. Consider increasing traffic allocation."
        elif improvement < -0.05:  # 5% degradation
            return "Strategy B shows significant degradation. Consider reducing traffic or reverting."
        else:
            return "Strategies perform similarly. Choice can be based on other factors."

    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive data for performance dashboard"""
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "real_time_metrics": {},
            "recent_alerts": [],
            "strategy_comparison": {},
            "system_health": {},
            "prometheus_available": self.enable_prometheus
        }

        # Real-time metrics for different windows
        for window in ['1m', '5m', '1h']:
            dashboard_data["real_time_metrics"][window] = {}
            for cache_key, metrics in self.real_time_cache.items():
                if cache_key.endswith(f'_{window}'):
                    strategy = cache_key.replace(f'_{window}', '')
                    dashboard_data["real_time_metrics"][window][strategy] = asdict(metrics)

        # Recent alerts by severity
        for level in AlertLevel:
            alerts = self.get_recent_alerts(level, limit=10)
            dashboard_data["recent_alerts"].append({
                "level": level.value,
                "count": len(alerts),
                "alerts": [asdict(alert) for alert in alerts]
            })

        # System health indicators
        total_selections_1h = sum(
            metrics.total_selections for metrics in self.real_time_cache.values()
            if '_1h' in str(metrics)
        )

        dashboard_data["system_health"] = {
            "total_selections_1h": total_selections_1h,
            "strategies_active": len(set(m.strategy for m in self.real_time_cache.values())),
            "error_rate_1h": self._calculate_overall_error_rate(),
            "data_points": len(self.selection_history),
            "monitoring_duration_hours": self._get_monitoring_duration_hours()
        }

        return dashboard_data

    def _calculate_overall_error_rate(self) -> float:
        """Calculate overall error rate in the last hour"""
        cutoff = datetime.now() - timedelta(hours=1)
        recent_events = [e for e in self.selection_history if e.get('timestamp', datetime.min) > cutoff]

        if not recent_events:
            return 0.0

        error_events = [e for e in recent_events if e.get('event_type') == 'error']
        return len(error_events) / len(recent_events)

    def _get_monitoring_duration_hours(self) -> float:
        """Get how long monitoring has been active"""
        if not self.selection_history:
            return 0.0

        oldest_event = min(self.selection_history, key=lambda x: x.get('timestamp', datetime.now()))
        duration = datetime.now() - oldest_event.get('timestamp', datetime.now())
        return duration.total_seconds() / 3600

    async def start_background_monitoring(self):
        """Start background monitoring tasks"""
        if self._background_task and not self._background_task.done():
            return

        self._shutdown_event.clear()
        self._background_task = asyncio.create_task(self._background_monitor_loop())
        logger.info("Background monitoring started")

    async def stop_background_monitoring(self):
        """Stop background monitoring tasks"""
        if self._background_task:
            self._shutdown_event.set()
            try:
                await asyncio.wait_for(self._background_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._background_task.cancel()
            logger.info("Background monitoring stopped")

    async def _background_monitor_loop(self):
        """Background monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Update real-time metrics cache
                self._update_real_time_metrics()

                # Update Prometheus metrics
                if self.prometheus_metrics:
                    for cache_key, metrics in self.real_time_cache.items():
                        if '_1m' in cache_key:  # Update 1-minute window metrics
                            strategy = cache_key.replace('_1m', '')
                            self.prometheus_metrics.update_success_rate(strategy, '1m', metrics.success_rate)

                # Clean up old data
                self._cleanup_old_data()

                await asyncio.sleep(30)  # Update every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    def _cleanup_old_data(self):
        """Clean up old data beyond retention periods"""
        # Clean alerts older than 24 hours
        cutoff_alerts = datetime.now() - timedelta(hours=24)
        with self._processing_lock:
            self.alerts = deque([a for a in self.alerts if a.timestamp > cutoff_alerts], maxlen=1000)

    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if self.prometheus_metrics:
            return self.prometheus_metrics.get_metrics_text()
        else:
            return "# Prometheus not available\n"


# Factory function for easy initialization
def create_performance_monitor(enable_prometheus: bool = True,
                            custom_thresholds: Optional[Dict[str, float]] = None) -> RLPerformanceMonitor:
    """Create a performance monitor with sensible defaults"""

    thresholds = MetricThresholds()
    if custom_thresholds:
        for key, value in custom_thresholds.items():
            if hasattr(thresholds, key):
                setattr(thresholds, key, value)

    monitor = RLPerformanceMonitor(thresholds, enable_prometheus)

    return monitor


# Example usage and demo
async def demo_performance_monitoring():
    """Demonstrate performance monitoring functionality"""

    print("RL Performance Monitoring Demo")
    print("=" * 50)

    # Create monitor
    monitor = create_performance_monitor(enable_prometheus=False)

    # Simulate selection events
    strategies = ['rl_enabled', 'traditional', 'hybrid']
    task_types = ['simple', 'complex', 'comprehensive']

    import random

    print("\nSimulating agent selection events...")

    for i in range(100):
        strategy = random.choice(strategies)
        task_type = random.choice(task_types)
        agents = [f"agent_{random.randint(1, 5)}" for _ in range(random.randint(1, 3))]

        # Simulate performance characteristics
        if strategy == 'rl_enabled':
            latency = random.uniform(0.8, 1.5)
            quality = random.uniform(0.75, 0.95)
            success_rate = 0.92
        elif strategy == 'traditional':
            latency = random.uniform(0.3, 0.8)
            quality = random.uniform(0.65, 0.85)
            success_rate = 0.85
        else:  # hybrid
            latency = random.uniform(0.5, 1.2)
            quality = random.uniform(0.70, 0.90)
            success_rate = 0.88

        success = random.random() < success_rate
        confidence = random.uniform(0.6, 0.9)

        monitor.record_selection_event(
            strategy=strategy,
            agent_ids=agents,
            task_type=task_type,
            selection_time=latency,
            confidence=confidence,
            success=success,
            quality_score=quality if success else quality * 0.5
        )

        # Simulate some errors
        if random.random() < 0.05:  # 5% error rate
            monitor.record_selection_error(strategy, "timeout", "Selection timeout")

    print(f"Recorded {100} selection events and some errors")

    # Get real-time metrics
    print("\nReal-time Metrics (5 minute window):")
    for strategy in strategies:
        metrics = monitor.get_real_time_metrics(strategy, '5m')
        if metrics:
            print(f"\n{strategy}:")
            print(f"  Success Rate: {metrics.success_rate:.1%}")
            print(f"  Avg Latency: {metrics.avg_latency_ms:.0f}ms")
            print(f"  P95 Latency: {metrics.p95_latency_ms:.0f}ms")
            print(f"  Avg Quality: {metrics.avg_quality_score:.2f}")
            print(f"  Total Selections: {metrics.total_selections}")

    # A/B test analysis
    print("\nA/B Test Analysis:")
    ab_result = monitor.analyze_ab_test('rl_enabled', 'traditional', 'quality_score')
    if ab_result['status'] == 'success':
        print(f"RL vs Traditional Quality Score:")
        print(f"  RL Mean: {ab_result['mean_a']:.3f}")
        print(f"  Traditional Mean: {ab_result['mean_b']:.3f}")
        print(f"  Improvement: {ab_result['improvement_percentage']:.1f}%")
        print(f"  P-value: {ab_result['p_value']:.4f}")
        print(f"  Significant: {ab_result['is_statistically_significant']}")
        print(f"  Recommendation: {ab_result['recommendation']}")

    # Recent alerts
    print("\nRecent Alerts:")
    alerts = monitor.get_recent_alerts(limit=5)
    for alert in alerts:
        print(f"  [{alert.level.value}] {alert.message}")

    # Dashboard data
    print("\nDashboard Summary:")
    dashboard = monitor.get_performance_dashboard_data()
    health = dashboard['system_health']
    print(f"  Total Selections (1h): {health['total_selections_1h']}")
    print(f"  Active Strategies: {health['strategies_active']}")
    print(f"  Error Rate (1h): {health['error_rate_1h']:.2%}")
    print(f"  Monitoring Duration: {health['monitoring_duration_hours']:.1f}h")

    return monitor


if __name__ == "__main__":
    # Run performance monitoring demo
    asyncio.run(demo_performance_monitoring())