#!/usr/bin/env python3
"""
Unified Performance Dashboard
==============================

Comprehensive performance monitoring for the Unified RAG Proposal System.
Integrates multi-strategy search metrics with real-time visualization.

Features:
- Real-time strategy performance tracking
- Cross-domain search analytics
- Quality score distributions
- Latency breakdown by strategy
- Historical trend analysis
- Alert thresholds and notifications

Usage:
    # Start dashboard server
    poetry run python src/monitoring/unified_performance_dashboard.py --serve --port 8080

    # Generate performance report
    poetry run python src/monitoring/unified_performance_dashboard.py --report --output report.json

    # Run benchmarks
    poetry run python src/monitoring/unified_performance_dashboard.py --benchmark
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class StrategyMetrics:
    """Metrics for a single strategy"""
    strategy_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_quality_score: float = 0.0
    avg_relevance_score: float = 0.0
    cross_domain_count: int = 0
    total_sources_returned: int = 0
    latency_history: List[float] = field(default_factory=list)
    quality_history: List[float] = field(default_factory=list)
    last_updated: str = ""

@dataclass
class SystemMetrics:
    """Overall system metrics"""
    total_requests: int = 0
    requests_per_minute: float = 0.0
    avg_latency_ms: float = 0.0
    success_rate: float = 0.0
    active_strategies: int = 0
    cross_domain_rate: float = 0.0
    quality_trend: str = "stable"  # improving, stable, declining
    peak_latency_ms: float = 0.0
    error_rate: float = 0.0
    uptime_hours: float = 0.0

@dataclass
class AlertConfig:
    """Alert configuration"""
    latency_threshold_ms: float = 2000.0
    error_rate_threshold: float = 0.1
    quality_score_threshold: float = 0.6
    cross_domain_target: float = 0.5

@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_type: str
    severity: str  # info, warning, critical
    message: str
    timestamp: str
    metric_value: float
    threshold: float

# ============================================================================
# Performance Tracker
# ============================================================================

class UnifiedPerformanceTracker:
    """
    Tracks and aggregates performance metrics for the unified RAG system
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self._lock = threading.Lock()

        # Per-strategy metrics
        self._strategy_metrics: Dict[str, StrategyMetrics] = {}

        # System-wide tracking
        self._start_time = datetime.now()
        self._request_timestamps: deque = deque(maxlen=1000)
        self._recent_latencies: deque = deque(maxlen=100)
        self._alerts: List[PerformanceAlert] = []

        # Historical data (last 24 hours, 5-minute buckets)
        self._historical_data: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        # Initialize strategies
        self._init_strategy_metrics()

    def _init_strategy_metrics(self):
        """Initialize metrics for all strategies"""
        strategies = [
            "hybrid", "enhanced_dd_raptor", "graph_rag",
            "golden_reference", "simple_rag", "multimodal_rag", "psychology_rag"
        ]

        for strategy in strategies:
            self._strategy_metrics[strategy] = StrategyMetrics(strategy_name=strategy)

    def record_search(
        self,
        strategy: str,
        latency_ms: float,
        success: bool,
        quality_score: float,
        sources_count: int,
        cross_domain: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a search operation"""
        with self._lock:
            # Ensure strategy exists
            if strategy not in self._strategy_metrics:
                self._strategy_metrics[strategy] = StrategyMetrics(strategy_name=strategy)

            metrics = self._strategy_metrics[strategy]

            # Update counts
            metrics.total_requests += 1
            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1

            # Update latency
            metrics.total_latency_ms += latency_ms
            metrics.avg_latency_ms = metrics.total_latency_ms / metrics.total_requests
            metrics.min_latency_ms = min(metrics.min_latency_ms, latency_ms)
            metrics.max_latency_ms = max(metrics.max_latency_ms, latency_ms)

            # Update history (keep last 100)
            metrics.latency_history.append(latency_ms)
            if len(metrics.latency_history) > 100:
                metrics.latency_history = metrics.latency_history[-100:]

            # Calculate percentiles
            if len(metrics.latency_history) >= 5:
                sorted_latencies = sorted(metrics.latency_history)
                metrics.p50_latency_ms = sorted_latencies[len(sorted_latencies) // 2]
                metrics.p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
                metrics.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]

            # Update quality
            metrics.quality_history.append(quality_score)
            if len(metrics.quality_history) > 100:
                metrics.quality_history = metrics.quality_history[-100:]
            metrics.avg_quality_score = statistics.mean(metrics.quality_history)

            # Update sources
            metrics.total_sources_returned += sources_count

            # Update cross-domain
            if cross_domain:
                metrics.cross_domain_count += 1

            metrics.last_updated = datetime.now().isoformat()

            # Record timestamp for RPM calculation
            self._request_timestamps.append(time.time())
            self._recent_latencies.append(latency_ms)

            # Check alerts
            self._check_alerts(strategy, latency_ms, quality_score)

    def _check_alerts(self, strategy: str, latency_ms: float, quality_score: float):
        """Check and generate alerts"""
        timestamp = datetime.now().isoformat()

        # Latency alert
        if latency_ms > self.config.latency_threshold_ms:
            self._alerts.append(PerformanceAlert(
                alert_type="high_latency",
                severity="warning" if latency_ms < self.config.latency_threshold_ms * 2 else "critical",
                message=f"Strategy {strategy} latency {latency_ms:.1f}ms exceeds threshold",
                timestamp=timestamp,
                metric_value=latency_ms,
                threshold=self.config.latency_threshold_ms
            ))

        # Quality alert
        if quality_score < self.config.quality_score_threshold:
            self._alerts.append(PerformanceAlert(
                alert_type="low_quality",
                severity="warning",
                message=f"Strategy {strategy} quality {quality_score:.3f} below threshold",
                timestamp=timestamp,
                metric_value=quality_score,
                threshold=self.config.quality_score_threshold
            ))

        # Keep only recent alerts (last 100)
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]

    def get_strategy_metrics(self, strategy: str) -> Optional[StrategyMetrics]:
        """Get metrics for a specific strategy"""
        return self._strategy_metrics.get(strategy)

    def get_all_strategy_metrics(self) -> Dict[str, StrategyMetrics]:
        """Get metrics for all strategies"""
        return self._strategy_metrics.copy()

    def get_system_metrics(self) -> SystemMetrics:
        """Get system-wide metrics"""
        with self._lock:
            total_requests = sum(m.total_requests for m in self._strategy_metrics.values())
            successful = sum(m.successful_requests for m in self._strategy_metrics.values())
            failed = sum(m.failed_requests for m in self._strategy_metrics.values())
            cross_domain = sum(m.cross_domain_count for m in self._strategy_metrics.values())

            # Calculate RPM
            now = time.time()
            recent_requests = sum(1 for t in self._request_timestamps if now - t < 60)

            # Calculate average latency
            avg_latency = 0.0
            if self._recent_latencies:
                avg_latency = statistics.mean(self._recent_latencies)

            # Calculate uptime
            uptime = (datetime.now() - self._start_time).total_seconds() / 3600

            # Determine quality trend
            quality_trend = "stable"
            if len(self._recent_latencies) >= 20:
                first_half = list(self._recent_latencies)[:len(self._recent_latencies)//2]
                second_half = list(self._recent_latencies)[len(self._recent_latencies)//2:]
                if statistics.mean(second_half) < statistics.mean(first_half) * 0.9:
                    quality_trend = "improving"
                elif statistics.mean(second_half) > statistics.mean(first_half) * 1.1:
                    quality_trend = "declining"

            return SystemMetrics(
                total_requests=total_requests,
                requests_per_minute=recent_requests,
                avg_latency_ms=avg_latency,
                success_rate=successful / max(total_requests, 1),
                active_strategies=sum(1 for m in self._strategy_metrics.values() if m.total_requests > 0),
                cross_domain_rate=cross_domain / max(total_requests, 1),
                quality_trend=quality_trend,
                peak_latency_ms=max((m.max_latency_ms for m in self._strategy_metrics.values()), default=0),
                error_rate=failed / max(total_requests, 1),
                uptime_hours=uptime
            )

    def get_alerts(self, severity: Optional[str] = None) -> List[PerformanceAlert]:
        """Get recent alerts, optionally filtered by severity"""
        if severity:
            return [a for a in self._alerts if a.severity == severity]
        return self._alerts.copy()

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        system = self.get_system_metrics()
        strategies = self.get_all_strategy_metrics()

        return {
            "report_timestamp": datetime.now().isoformat(),
            "system_metrics": asdict(system),
            "strategy_breakdown": {
                name: {
                    "total_requests": m.total_requests,
                    "success_rate": m.successful_requests / max(m.total_requests, 1),
                    "avg_latency_ms": m.avg_latency_ms,
                    "p95_latency_ms": m.p95_latency_ms,
                    "avg_quality_score": m.avg_quality_score,
                    "cross_domain_rate": m.cross_domain_count / max(m.total_requests, 1),
                    "sources_per_request": m.total_sources_returned / max(m.total_requests, 1)
                }
                for name, m in strategies.items()
                if m.total_requests > 0
            },
            "alerts": [asdict(a) for a in self._alerts[-20:]],
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        system = self.get_system_metrics()

        if system.avg_latency_ms > 1500:
            recommendations.append("⚠️ Average latency high. Consider caching frequent queries.")

        if system.error_rate > 0.05:
            recommendations.append("❌ Error rate above 5%. Check ChromaDB connections.")

        if system.cross_domain_rate < 0.3:
            recommendations.append("💡 Cross-domain usage low. Enable for richer results.")

        # Strategy-specific recommendations
        for name, metrics in self._strategy_metrics.items():
            if metrics.total_requests > 10:
                if metrics.avg_quality_score < 0.6:
                    recommendations.append(f"🔧 {name} quality low ({metrics.avg_quality_score:.2f}). Review relevance tuning.")

                if metrics.p95_latency_ms > 3000:
                    recommendations.append(f"⏱️ {name} p95 latency high ({metrics.p95_latency_ms:.0f}ms). Optimize query.")

        if not recommendations:
            recommendations.append("✅ System performing well within thresholds.")

        return recommendations


# ============================================================================
# Dashboard Server (Simple HTTP)
# ============================================================================

class DashboardServer:
    """Simple HTTP server for performance dashboard"""

    def __init__(self, tracker: UnifiedPerformanceTracker, port: int = 8080):
        self.tracker = tracker
        self.port = port

    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard"""
        system = self.tracker.get_system_metrics()
        report = self.tracker.get_performance_report()

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Unified RAG Performance Dashboard</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; font-size: 14px; }}
        .strategy-row {{ display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #eee; }}
        .alert {{ padding: 10px; border-radius: 4px; margin: 5px 0; }}
        .alert-warning {{ background: #fff3cd; }}
        .alert-critical {{ background: #f8d7da; }}
        .good {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .bad {{ color: #dc3545; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Unified RAG Performance Dashboard</h1>
        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="card">
            <h2>📊 System Overview</h2>
            <div class="metric">
                <div class="metric-value">{system.total_requests}</div>
                <div class="metric-label">Total Requests</div>
            </div>
            <div class="metric">
                <div class="metric-value">{system.requests_per_minute:.1f}</div>
                <div class="metric-label">Requests/min</div>
            </div>
            <div class="metric">
                <div class="metric-value {'good' if system.avg_latency_ms < 1000 else 'warning' if system.avg_latency_ms < 2000 else 'bad'}">{system.avg_latency_ms:.0f}ms</div>
                <div class="metric-label">Avg Latency</div>
            </div>
            <div class="metric">
                <div class="metric-value {'good' if system.success_rate > 0.95 else 'warning' if system.success_rate > 0.8 else 'bad'}">{system.success_rate*100:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{system.active_strategies}</div>
                <div class="metric-label">Active Strategies</div>
            </div>
            <div class="metric">
                <div class="metric-value">{system.cross_domain_rate*100:.1f}%</div>
                <div class="metric-label">Cross-Domain</div>
            </div>
        </div>

        <div class="card">
            <h2>🔧 Strategy Performance</h2>
            {''.join(self._generate_strategy_rows(report.get('strategy_breakdown', {})))}
        </div>

        <div class="card">
            <h2>⚠️ Recent Alerts</h2>
            {''.join(self._generate_alert_rows(report.get('alerts', [])))}
            {f'<p>No recent alerts</p>' if not report.get('alerts') else ''}
        </div>

        <div class="card">
            <h2>💡 Recommendations</h2>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in report.get('recommendations', []))}
            </ul>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _generate_strategy_rows(self, strategies: Dict[str, Any]) -> List[str]:
        """Generate HTML rows for strategies"""
        rows = []
        for name, data in strategies.items():
            quality_class = 'good' if data['avg_quality_score'] > 0.7 else 'warning' if data['avg_quality_score'] > 0.5 else 'bad'
            latency_class = 'good' if data['avg_latency_ms'] < 500 else 'warning' if data['avg_latency_ms'] < 1500 else 'bad'

            rows.append(f"""
            <div class="strategy-row">
                <strong>{name}</strong>
                <span>Requests: {data['total_requests']}</span>
                <span class="{latency_class}">Latency: {data['avg_latency_ms']:.0f}ms</span>
                <span class="{quality_class}">Quality: {data['avg_quality_score']:.2f}</span>
                <span>Success: {data['success_rate']*100:.1f}%</span>
            </div>
            """)
        return rows

    def _generate_alert_rows(self, alerts: List[Dict]) -> List[str]:
        """Generate HTML rows for alerts"""
        rows = []
        for alert in alerts[-10:]:  # Last 10 alerts
            severity_class = f"alert-{alert['severity']}"
            rows.append(f"""
            <div class="alert {severity_class}">
                <strong>{alert['alert_type']}</strong>: {alert['message']}
                <small>({alert['timestamp']})</small>
            </div>
            """)
        return rows


# ============================================================================
# Global Instance
# ============================================================================

_performance_tracker: Optional[UnifiedPerformanceTracker] = None

def get_performance_tracker() -> UnifiedPerformanceTracker:
    """Get or create global performance tracker"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = UnifiedPerformanceTracker()
    return _performance_tracker


# ============================================================================
# Benchmark Runner
# ============================================================================

async def run_benchmark(num_queries: int = 20) -> Dict[str, Any]:
    """Run performance benchmark"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    try:
        from src.services.rag.multi_strategy_search import create_search_engine
    except ImportError:
        logger.error("Could not import multi_strategy_search")
        return {"error": "Import failed"}

    logger.info(f"Running benchmark with {num_queries} queries...")
    tracker = get_performance_tracker()

    # Test queries
    test_queries = [
        ("ESM3 protein structure prediction", "neuroscience"),
        ("brain development neural connectivity", "neuroscience"),
        ("quantum machine learning optimization", "quantum_ml"),
        ("Samsung grant proposal research", "general"),
        ("autism spectrum disorder diagnosis", "developmental_disorders"),
        ("neural network architecture design", "general"),
        ("protein folding evolution", "neuroscience"),
        ("developmental delay early intervention", "developmental_disorders"),
        ("machine learning healthcare applications", "general"),
        ("brain imaging fMRI analysis", "neuroscience")
    ]

    # Create engine
    engine = await create_search_engine()

    results = []
    for i in range(num_queries):
        query, domain = test_queries[i % len(test_queries)]

        start = time.time()
        try:
            result = await engine.search(
                query=query,
                domain=domain,
                complexity="medium"
            )

            latency = (time.time() - start) * 1000

            # Record metrics
            for strategy in result.strategies_used:
                perf = result.performance_breakdown.get(strategy, {})
                tracker.record_search(
                    strategy=strategy,
                    latency_ms=perf.get("latency_ms", latency),
                    success=True,
                    quality_score=perf.get("confidence", 0.7),
                    sources_count=perf.get("results_count", 0),
                    cross_domain=result.cross_domain_detected
                )

            results.append({
                "query": query,
                "success": True,
                "latency_ms": latency,
                "strategies": result.strategies_used,
                "sources": result.total_sources
            })

        except Exception as e:
            latency = (time.time() - start) * 1000
            tracker.record_search(
                strategy="unknown",
                latency_ms=latency,
                success=False,
                quality_score=0,
                sources_count=0,
                cross_domain=False
            )
            results.append({
                "query": query,
                "success": False,
                "error": str(e)
            })

        await asyncio.sleep(0.1)  # Small delay between queries

    # Generate report
    report = tracker.get_performance_report()
    report["benchmark_results"] = {
        "total_queries": num_queries,
        "successful_queries": sum(1 for r in results if r.get("success")),
        "avg_latency_ms": statistics.mean(r.get("latency_ms", 0) for r in results if r.get("success")),
        "queries": results[:10]  # First 10 for brevity
    }

    return report


# ============================================================================
# CLI
# ============================================================================

async def main():
    """CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Unified Performance Dashboard")
    parser.add_argument("--serve", action="store_true", help="Start dashboard server")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--queries", type=int, default=20, help="Benchmark queries")

    args = parser.parse_args()

    tracker = get_performance_tracker()

    if args.benchmark:
        print("🏃 Running performance benchmark...")
        report = await run_benchmark(args.queries)

        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")

        if "benchmark_results" in report:
            br = report["benchmark_results"]
            print(f"Total Queries: {br['total_queries']}")
            print(f"Successful: {br['successful_queries']}")
            print(f"Avg Latency: {br['avg_latency_ms']:.1f}ms")

        print(f"\n📊 Strategy Performance:")
        for name, data in report.get("strategy_breakdown", {}).items():
            print(f"  {name}: {data['total_requests']} requests, {data['avg_latency_ms']:.0f}ms avg")

        print(f"\n💡 Recommendations:")
        for rec in report.get("recommendations", []):
            print(f"  {rec}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Report saved: {args.output}")

    elif args.report:
        report = tracker.get_performance_report()
        print(json.dumps(report, indent=2))

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Report saved: {args.output}")

    elif args.serve:
        print(f"🌐 Dashboard available at http://localhost:{args.port}")
        # In production, use proper HTTP server
        # For now, just generate static HTML
        dashboard = DashboardServer(tracker, args.port)
        html = dashboard.generate_html_dashboard()
        output_path = Path("output/dashboard.html")
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(html)
        print(f"📄 Dashboard HTML saved to: {output_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())