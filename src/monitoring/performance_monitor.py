"""
Performance Monitoring System for DD-RAPTOR
Real-time monitoring, alerting, and performance optimization
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import redis
from contextlib import contextmanager
from collections import deque, defaultdict
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    disk_io: Dict[str, float] = field(default_factory=dict)
    network_io: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    queue_depth: int = 0
    active_connections: int = 0

@dataclass
class AlertConfig:
    """Alert configuration"""
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    duration_minutes: int = 5
    severity: str = 'warning'  # 'critical', 'warning', 'info'
    enabled: bool = True

class PrometheusMetrics:
    """Prometheus metrics collection"""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # Request metrics
        self.request_duration = Histogram(
            'dd_raptor_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )

        self.request_count = Counter(
            'dd_raptor_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        # System metrics
        self.cpu_usage = Gauge(
            'dd_raptor_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'dd_raptor_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )

        self.gpu_usage = Gauge(
            'dd_raptor_gpu_usage_percent',
            'GPU usage percentage',
            registry=self.registry
        )

        # RAG specific metrics
        self.rag_latency = Histogram(
            'dd_raptor_rag_latency_seconds',
            'RAG query latency',
            ['query_type'],
            registry=self.registry
        )

        self.document_retrievals = Counter(
            'dd_raptor_document_retrievals_total',
            'Total document retrievals',
            ['source', 'success'],
            registry=self.registry
        )

        self.proposal_generations = Counter(
            'dd_raptor_proposal_generations_total',
            'Total proposal generations',
            ['status', 'type'],
            registry=self.registry
        )

class PerformanceMonitor:
    """Real-time performance monitoring system"""

    def __init__(self,
                 redis_client: Optional[redis.Redis] = None,
                 monitoring_interval: int = 30,
                 alert_configs: Optional[List[AlertConfig]] = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.monitoring_interval = monitoring_interval
        self.alert_configs = alert_configs or self._default_alert_configs()

        # Metrics storage
        self.metrics_history = deque(maxlen=1000)  # Last 1000 measurements
        self.alert_states = defaultdict(bool)
        self.alert_timestamps = defaultdict(datetime)

        # Prometheus metrics
        self.prometheus_metrics = PrometheusMetrics()

        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Performance baselines
        self.baselines = {
            'cpu_usage': 50.0,
            'memory_usage': 4000000000,  # 4GB in bytes
            'latency_ms': 500.0,
            'error_rate': 0.05  # 5%
        }

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _default_alert_configs(self) -> List[AlertConfig]:
        """Default alert configurations"""
        return [
            AlertConfig('cpu_usage', 80.0, 'gt', 5, 'warning'),
            AlertConfig('cpu_usage', 95.0, 'gt', 2, 'critical'),
            AlertConfig('memory_usage', 6000000000, 'gt', 5, 'warning'),  # 6GB
            AlertConfig('memory_usage', 7000000000, 'gt', 2, 'critical'),  # 7GB
            AlertConfig('latency_ms', 1000.0, 'gt', 3, 'warning'),
            AlertConfig('latency_ms', 5000.0, 'gt', 1, 'critical'),
            AlertConfig('error_rate', 0.10, 'gt', 5, 'warning'),  # 10%
            AlertConfig('error_rate', 0.25, 'gt', 2, 'critical'),  # 25%
        ]

    async def collect_system_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive system metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_metrics = {
                'read_bytes_per_sec': disk_io.read_bytes,
                'write_bytes_per_sec': disk_io.write_bytes
            } if disk_io else {}

            # Network I/O
            network_io = psutil.net_io_counters()
            network_metrics = {
                'bytes_sent_per_sec': network_io.bytes_sent,
                'bytes_recv_per_sec': network_io.bytes_recv
            } if network_io else {}

            # GPU usage (if available)
            gpu_usage = await self._get_gpu_usage()

            # Application-specific metrics
            app_metrics = await self._get_application_metrics()

            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_percent,
                memory_usage=memory.used,
                gpu_usage=gpu_usage,
                disk_io=disk_metrics,
                network_io=network_metrics,
                latency_ms=app_metrics.get('latency_ms', 0.0),
                throughput_rps=app_metrics.get('throughput_rps', 0.0),
                error_rate=app_metrics.get('error_rate', 0.0),
                queue_depth=app_metrics.get('queue_depth', 0),
                active_connections=app_metrics.get('active_connections', 0)
            )

            # Update Prometheus metrics
            self.prometheus_metrics.cpu_usage.set(cpu_percent)
            self.prometheus_metrics.memory_usage.set(memory.used)
            if gpu_usage is not None:
                self.prometheus_metrics.gpu_usage.set(gpu_usage)

            return metrics

        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            raise

    async def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return gpus[0].load * 100 if gpus else None
        except ImportError:
            return None
        except Exception:
            return None

    async def _get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics from Redis"""
        try:
            # Get metrics from Redis cache
            metrics_data = await self.redis_client.hgetall('app_metrics')
            if not metrics_data:
                return {}

            # Convert bytes to appropriate types
            return {
                'latency_ms': float(metrics_data.get(b'latency_ms', 0)),
                'throughput_rps': float(metrics_data.get(b'throughput_rps', 0)),
                'error_rate': float(metrics_data.get(b'error_rate', 0)),
                'queue_depth': int(metrics_data.get(b'queue_depth', 0)),
                'active_connections': int(metrics_data.get(b'active_connections', 0))
            }
        except Exception as e:
            self.logger.warning(f"Could not get application metrics: {e}")
            return {}

    @contextmanager
    def measure_latency(self, operation_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for measuring operation latency"""
        start_time = time.time()
        labels = labels or {}

        try:
            yield
            success = True
        except Exception as e:
            success = False
            raise
        finally:
            duration = time.time() - start_time

            # Record in Prometheus
            self.prometheus_metrics.rag_latency.labels(
                query_type=labels.get('query_type', 'unknown')
            ).observe(duration)

            # Store for alerting
            asyncio.create_task(self._update_latency_metric(duration * 1000))  # Convert to ms

    async def _update_latency_metric(self, latency_ms: float):
        """Update latency metric in Redis"""
        try:
            await self.redis_client.hset('app_metrics', 'latency_ms', latency_ms)
        except Exception as e:
            self.logger.warning(f"Could not update latency metric: {e}")

    async def check_alerts(self, metrics: PerformanceMetrics):
        """Check alert conditions and trigger notifications"""
        current_time = datetime.now()

        for alert_config in self.alert_configs:
            if not alert_config.enabled:
                continue

            metric_value = getattr(metrics, alert_config.metric_name, None)
            if metric_value is None:
                continue

            # Check threshold
            alert_triggered = False
            if alert_config.comparison == 'gt':
                alert_triggered = metric_value > alert_config.threshold
            elif alert_config.comparison == 'lt':
                alert_triggered = metric_value < alert_config.threshold
            elif alert_config.comparison == 'eq':
                alert_triggered = abs(metric_value - alert_config.threshold) < 0.01

            alert_key = f"{alert_config.metric_name}_{alert_config.threshold}"

            if alert_triggered:
                # Check if this is a new alert or duration threshold met
                if not self.alert_states[alert_key]:
                    self.alert_states[alert_key] = True
                    self.alert_timestamps[alert_key] = current_time
                elif (current_time - self.alert_timestamps[alert_key]).total_seconds() >= alert_config.duration_minutes * 60:
                    # Duration threshold met, send alert
                    await self._send_alert(alert_config, metric_value, metrics)
            else:
                # Clear alert state
                self.alert_states[alert_key] = False

    async def _send_alert(self, alert_config: AlertConfig, value: float, metrics: PerformanceMetrics):
        """Send alert notification"""
        alert_message = {
            'severity': alert_config.severity,
            'metric': alert_config.metric_name,
            'value': value,
            'threshold': alert_config.threshold,
            'timestamp': metrics.timestamp.isoformat(),
            'system_state': {
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'latency_ms': metrics.latency_ms,
                'error_rate': metrics.error_rate
            }
        }

        # Log alert
        self.logger.warning(f"ALERT [{alert_config.severity.upper()}]: {alert_config.metric_name} = {value} (threshold: {alert_config.threshold})")

        # Store in Redis for external systems
        await self.redis_client.lpush('alerts', json.dumps(alert_message))
        await self.redis_client.ltrim('alerts', 0, 100)  # Keep last 100 alerts

    async def get_performance_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the specified time window"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {}

        # Calculate statistics
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage for m in recent_metrics]
        latency_values = [m.latency_ms for m in recent_metrics if m.latency_ms > 0]

        summary = {
            'time_window_minutes': time_window_minutes,
            'sample_count': len(recent_metrics),
            'cpu': {
                'avg': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'p95': np.percentile(cpu_values, 95)
            },
            'memory': {
                'avg': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'p95': np.percentile(memory_values, 95)
            },
            'current_state': {
                'cpu_usage': recent_metrics[-1].cpu_usage,
                'memory_usage': recent_metrics[-1].memory_usage,
                'timestamp': recent_metrics[-1].timestamp.isoformat()
            }
        }

        if latency_values:
            summary['latency'] = {
                'avg': np.mean(latency_values),
                'max': np.max(latency_values),
                'min': np.min(latency_values),
                'p95': np.percentile(latency_values, 95),
                'p99': np.percentile(latency_values, 99)
            }

        return summary

    async def optimize_performance(self, metrics: PerformanceMetrics) -> List[str]:
        """Provide performance optimization recommendations"""
        recommendations = []

        # CPU optimization
        if metrics.cpu_usage > 80:
            recommendations.append("High CPU usage detected. Consider scaling horizontally or optimizing CPU-intensive operations.")
            if metrics.cpu_usage > 95:
                recommendations.append("CRITICAL: CPU usage above 95%. Immediate action required.")

        # Memory optimization
        if metrics.memory_usage > self.baselines['memory_usage'] * 1.5:
            recommendations.append("High memory usage detected. Consider memory profiling and optimization.")
            if metrics.memory_usage > self.baselines['memory_usage'] * 2:
                recommendations.append("CRITICAL: Memory usage doubled baseline. Check for memory leaks.")

        # Latency optimization
        if metrics.latency_ms > self.baselines['latency_ms'] * 2:
            recommendations.append("High latency detected. Consider caching, database optimization, or model optimization.")
            if metrics.latency_ms > 5000:
                recommendations.append("CRITICAL: Latency above 5 seconds. User experience severely impacted.")

        # Error rate optimization
        if metrics.error_rate > self.baselines['error_rate'] * 2:
            recommendations.append("High error rate detected. Review error logs and implement circuit breakers.")
            if metrics.error_rate > 0.25:
                recommendations.append("CRITICAL: Error rate above 25%. Service may be failing.")

        return recommendations

    async def start_monitoring(self):
        """Start the monitoring service"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        self.logger.info(f"Performance monitoring started with {self.monitoring_interval}s interval")

    async def stop_monitoring(self):
        """Stop the monitoring service"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=30)

        self.executor.shutdown(wait=True)
        self.logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop (runs in separate thread)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while self.monitoring_active:
                try:
                    # Collect metrics
                    metrics = loop.run_until_complete(self.collect_system_metrics())
                    self.metrics_history.append(metrics)

                    # Store metrics in Redis
                    loop.run_until_complete(self._store_metrics(metrics))

                    # Check alerts
                    loop.run_until_complete(self.check_alerts(metrics))

                    # Sleep for monitoring interval
                    time.sleep(self.monitoring_interval)

                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(10)  # Wait before retrying

        finally:
            loop.close()

    async def _store_metrics(self, metrics: PerformanceMetrics):
        """Store metrics in Redis for external access"""
        try:
            metrics_data = {
                'timestamp': metrics.timestamp.isoformat(),
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'latency_ms': metrics.latency_ms,
                'throughput_rps': metrics.throughput_rps,
                'error_rate': metrics.error_rate,
                'queue_depth': metrics.queue_depth,
                'active_connections': metrics.active_connections
            }

            if metrics.gpu_usage is not None:
                metrics_data['gpu_usage'] = metrics.gpu_usage

            # Store current metrics
            await self.redis_client.hset('current_metrics', mapping=metrics_data)

            # Store in time series (keep last 24 hours)
            await self.redis_client.zadd('metrics_timeseries',
                                       {json.dumps(metrics_data): time.time()})

            # Clean old entries (older than 24 hours)
            cutoff_time = time.time() - (24 * 60 * 60)
            await self.redis_client.zremrangebyscore('metrics_timeseries', 0, cutoff_time)

        except Exception as e:
            self.logger.warning(f"Could not store metrics: {e}")

# Usage decorator for automatic performance monitoring
def monitor_performance(operation_name: str, monitor_instance: Optional[PerformanceMonitor] = None):
    """Decorator for automatic performance monitoring"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            monitor = monitor_instance or getattr(args[0], 'performance_monitor', None)
            if monitor:
                with monitor.measure_latency(operation_name):
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            monitor = monitor_instance or getattr(args[0], 'performance_monitor', None)
            if monitor:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    asyncio.create_task(monitor._update_latency_metric(duration_ms))
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    asyncio.create_task(monitor._update_latency_metric(duration_ms))
                    raise
            else:
                return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

if __name__ == "__main__":
    # Example usage
    async def main():
        monitor = PerformanceMonitor(monitoring_interval=10)
        await monitor.start_monitoring()

        # Let it run for a bit
        await asyncio.sleep(60)

        # Get summary
        summary = await monitor.get_performance_summary(time_window_minutes=5)
        print("Performance Summary:", json.dumps(summary, indent=2))

        await monitor.stop_monitoring()

    asyncio.run(main())