"""
Monitoring Dashboard and Alerts for RL Agent Selection

This module provides a comprehensive web-based monitoring dashboard and alerting
system for the RL-enhanced agent selection system. It includes:

- Real-time performance monitoring and visualization
- Safety status monitoring and alerts
- A/B testing results and statistical analysis
- Model performance tracking and comparison
- System health indicators and resource monitoring
- Alert management and notification system
- Configuration management interface
- Emergency controls and manual overrides

The dashboard is built using FastAPI for the backend with WebSocket support
for real-time updates, and provides both a web interface and REST API.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid

try:
    from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    from fastapi.websockets import WebSocketDisconnect
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available, dashboard will not be functional")

try:
    from .hybrid_agent_selector import HybridAgentSelector
    from .performance_monitor import RLPerformanceMonitor
    from .safety_mechanisms import SafetyManager, SafetyLevel
    from .continuous_learning import ContinuousLearningPipeline
    RL_COMPONENTS_AVAILABLE = True
except ImportError:
    RL_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


# Pydantic models for API
class AlertConfig(BaseModel):
    """Alert configuration model"""
    name: str
    metric: str
    threshold: float
    comparison: str  # 'greater_than', 'less_than'
    enabled: bool = True


class SystemOverride(BaseModel):
    """System override model"""
    component: str
    action: str
    reason: str
    duration_minutes: Optional[int] = None


class TrafficConfig(BaseModel):
    """Traffic configuration model"""
    strategy: str
    percentage: float
    ramp_up_duration_minutes: int = 60


@dataclass
class DashboardAlert:
    """Dashboard alert"""
    alert_id: str
    title: str
    message: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    timestamp: datetime
    source: str
    acknowledged: bool = False
    resolved: bool = False


class ConnectionManager:
    """WebSocket connection manager for real-time updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_info: Dict[str, Any]):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = client_info
        logger.info(f"Client connected: {client_info}")

    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_metadata.pop(websocket, None)
            logger.info("Client disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send message to client: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_json(self, data: Dict[str, Any]):
        """Broadcast JSON data to all connected clients"""
        message = json.dumps(data, default=str)
        await self.broadcast(message)


class MonitoringDashboard:
    """
    Comprehensive monitoring dashboard for RL agent selection system

    Provides real-time monitoring, alerting, and control capabilities
    through a web interface and REST API.
    """

    def __init__(self,
                 hybrid_selector: Optional[HybridAgentSelector] = None,
                 performance_monitor: Optional[RLPerformanceMonitor] = None,
                 safety_manager: Optional[SafetyManager] = None,
                 learning_pipeline: Optional[ContinuousLearningPipeline] = None):
        """
        Initialize monitoring dashboard

        Args:
            hybrid_selector: Hybrid agent selector instance
            performance_monitor: Performance monitor instance
            safety_manager: Safety manager instance
            learning_pipeline: Continuous learning pipeline instance
        """
        self.hybrid_selector = hybrid_selector
        self.performance_monitor = performance_monitor
        self.safety_manager = safety_manager
        self.learning_pipeline = learning_pipeline

        # Dashboard state
        self.alerts: List[DashboardAlert] = []
        self.alert_configs: List[AlertConfig] = []
        self.connection_manager = ConnectionManager()

        # Monitoring state
        self.monitoring_active = False
        self.last_update_time = datetime.now()
        self.update_interval_seconds = 10

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Initialize FastAPI app if available
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app()
        else:
            self.app = None
            logger.warning("FastAPI not available, web interface disabled")

        logger.info("Monitoring Dashboard initialized")

    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="RL Agent Selection Monitoring Dashboard",
            description="Real-time monitoring and control for RL-enhanced agent selection",
            version="1.0.0"
        )

        # Add routes
        self._add_api_routes(app)
        self._add_websocket_routes(app)
        self._add_static_routes(app)

        return app

    def _add_api_routes(self, app: FastAPI):
        """Add REST API routes"""

        @app.get("/api/status")
        async def get_system_status():
            """Get comprehensive system status"""
            return await self.get_system_status()

        @app.get("/api/performance")
        async def get_performance_metrics():
            """Get performance metrics"""
            return await self.get_performance_data()

        @app.get("/api/safety")
        async def get_safety_status():
            """Get safety status"""
            return await self.get_safety_data()

        @app.get("/api/alerts")
        async def get_alerts():
            """Get current alerts"""
            return self.get_alerts_data()

        @app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str):
            """Acknowledge an alert"""
            return self.acknowledge_alert(alert_id)

        @app.post("/api/alerts/config")
        async def create_alert_config(config: AlertConfig):
            """Create new alert configuration"""
            return self.add_alert_config(config)

        @app.post("/api/system/override")
        async def system_override(override: SystemOverride):
            """Apply system override"""
            return await self.apply_system_override(override)

        @app.post("/api/traffic/config")
        async def update_traffic_config(config: TrafficConfig):
            """Update traffic configuration"""
            return await self.update_traffic_configuration(config)

        @app.get("/api/models/history")
        async def get_model_history():
            """Get model version history"""
            return await self.get_model_history()

        @app.post("/api/models/{version_id}/deploy")
        async def deploy_model_version(version_id: str):
            """Deploy specific model version"""
            return await self.deploy_model_version(version_id)

        @app.post("/api/learning/retrain")
        async def trigger_retraining(background_tasks: BackgroundTasks):
            """Trigger model retraining"""
            return await self.trigger_retraining(background_tasks)

        @app.get("/api/ab-tests")
        async def get_ab_test_results():
            """Get A/B testing results"""
            return await self.get_ab_test_data()

    def _add_websocket_routes(self, app: FastAPI):
        """Add WebSocket routes for real-time updates"""

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            client_info = {
                "client_id": str(uuid.uuid4()),
                "connected_at": datetime.now(),
                "subscriptions": ["all"]  # Default subscription
            }

            await self.connection_manager.connect(websocket, client_info)

            try:
                while True:
                    # Wait for messages from client (e.g., subscription updates)
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    if message.get("type") == "subscribe":
                        client_info["subscriptions"] = message.get("topics", ["all"])

                    elif message.get("type") == "ping":
                        await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})

            except WebSocketDisconnect:
                self.connection_manager.disconnect(websocket)

    def _add_static_routes(self, app: FastAPI):
        """Add static file routes for web interface"""

        # Create basic HTML interface
        @app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Serve dashboard home page"""
            return self._generate_dashboard_html()

        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now()}

    def _generate_dashboard_html(self) -> str:
        """Generate basic dashboard HTML"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>RL Agent Selection Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .status-card { border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .status-good { background-color: #d4edda; border-color: #c3e6cb; }
                .status-warning { background-color: #fff3cd; border-color: #ffeaa7; }
                .status-error { background-color: #f8d7da; border-color: #f5c6cb; }
                .metric { margin: 5px 0; }
                #alerts { max-height: 300px; overflow-y: auto; }
                .alert { padding: 8px; margin: 5px 0; border-radius: 3px; }
                .alert-info { background-color: #d1ecf1; }
                .alert-warning { background-color: #fff3cd; }
                .alert-error { background-color: #f8d7da; }
                .alert-critical { background-color: #f5c6cb; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>🤖 RL Agent Selection Monitoring Dashboard</h1>

            <div id="system-status" class="status-card">
                <h2>System Status</h2>
                <div id="status-content">Loading...</div>
            </div>

            <div id="performance-metrics" class="status-card">
                <h2>Performance Metrics</h2>
                <div id="performance-content">Loading...</div>
            </div>

            <div id="safety-status" class="status-card">
                <h2>Safety Status</h2>
                <div id="safety-content">Loading...</div>
            </div>

            <div id="alerts" class="status-card">
                <h2>Active Alerts</h2>
                <div id="alerts-content">Loading...</div>
            </div>

            <script>
                // WebSocket connection for real-time updates
                const ws = new WebSocket('ws://localhost:8000/ws');

                ws.onopen = function(event) {
                    console.log('Connected to dashboard');
                    // Subscribe to all updates
                    ws.send(JSON.stringify({type: 'subscribe', topics: ['all']}));
                };

                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'status_update') {
                        updateDashboard(data.payload);
                    }
                };

                function updateDashboard(data) {
                    // Update system status
                    if (data.system_status) {
                        updateSystemStatus(data.system_status);
                    }

                    // Update performance metrics
                    if (data.performance) {
                        updatePerformanceMetrics(data.performance);
                    }

                    // Update safety status
                    if (data.safety) {
                        updateSafetyStatus(data.safety);
                    }

                    // Update alerts
                    if (data.alerts) {
                        updateAlerts(data.alerts);
                    }
                }

                function updateSystemStatus(status) {
                    const content = document.getElementById('status-content');
                    const isHealthy = status.overall_healthy;

                    content.innerHTML = `
                        <div class="metric">Overall Health: <strong>${isHealthy ? 'Healthy' : 'Degraded'}</strong></div>
                        <div class="metric">Safety Level: <strong>${status.safety_level}</strong></div>
                        <div class="metric">RL Enabled: <strong>${status.rl_enabled ? 'Yes' : 'No'}</strong></div>
                        <div class="metric">Last Update: <strong>${new Date().toLocaleTimeString()}</strong></div>
                    `;

                    const card = document.getElementById('system-status');
                    card.className = `status-card ${isHealthy ? 'status-good' : 'status-warning'}`;
                }

                function updatePerformanceMetrics(perf) {
                    const content = document.getElementById('performance-content');
                    content.innerHTML = `
                        <div class="metric">Success Rate: <strong>${(perf.success_rate * 100).toFixed(1)}%</strong></div>
                        <div class="metric">Avg Latency: <strong>${perf.avg_latency_ms.toFixed(0)}ms</strong></div>
                        <div class="metric">P95 Latency: <strong>${perf.p95_latency_ms.toFixed(0)}ms</strong></div>
                        <div class="metric">Total Selections: <strong>${perf.total_selections}</strong></div>
                    `;
                }

                function updateSafetyStatus(safety) {
                    const content = document.getElementById('safety-content');
                    const isSafe = safety.resource_safe && safety.performance_healthy;

                    content.innerHTML = `
                        <div class="metric">Resource Safe: <strong>${safety.resource_safe ? 'Yes' : 'No'}</strong></div>
                        <div class="metric">Performance Healthy: <strong>${safety.performance_healthy ? 'Yes' : 'No'}</strong></div>
                        <div class="metric">Circuit Breakers: <strong>${Object.keys(safety.circuit_breakers).length}</strong></div>
                        <div class="metric">Recent Incidents: <strong>${safety.total_incidents}</strong></div>
                    `;

                    const card = document.getElementById('safety-status');
                    card.className = `status-card ${isSafe ? 'status-good' : 'status-error'}`;
                }

                function updateAlerts(alerts) {
                    const content = document.getElementById('alerts-content');

                    if (alerts.length === 0) {
                        content.innerHTML = '<div class="metric">No active alerts</div>';
                        return;
                    }

                    content.innerHTML = alerts.map(alert => `
                        <div class="alert alert-${alert.severity}">
                            <strong>${alert.title}</strong><br>
                            ${alert.message}<br>
                            <small>${new Date(alert.timestamp).toLocaleString()}</small>
                        </div>
                    `).join('');
                }

                // Initial data load
                async function loadInitialData() {
                    try {
                        const [statusRes, perfRes, safetyRes, alertsRes] = await Promise.all([
                            fetch('/api/status'),
                            fetch('/api/performance'),
                            fetch('/api/safety'),
                            fetch('/api/alerts')
                        ]);

                        const data = {
                            system_status: await statusRes.json(),
                            performance: await perfRes.json(),
                            safety: await safetyRes.json(),
                            alerts: await alertsRes.json()
                        };

                        updateDashboard(data);
                    } catch (error) {
                        console.error('Failed to load initial data:', error);
                    }
                }

                // Load data immediately and then every 30 seconds
                loadInitialData();
                setInterval(loadInitialData, 30000);
            </script>
        </body>
        </html>
        '''

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "timestamp": datetime.now(),
            "monitoring_active": self.monitoring_active,
            "last_update": self.last_update_time,
            "components": {}
        }

        # Check component status
        if self.hybrid_selector:
            status["components"]["hybrid_selector"] = {
                "available": True,
                "rl_enabled": self.hybrid_selector.rl_selector is not None,
                "ab_testing": getattr(self.hybrid_selector.config, 'enable_ab_testing', False)
            }

        if self.safety_manager:
            safety_status = self.safety_manager.get_safety_status()
            status["components"]["safety"] = {
                "available": True,
                "safety_level": safety_status["current_safety_level"],
                "resource_safe": safety_status["resource_safe"],
                "performance_healthy": safety_status["performance_healthy"]
            }

        if self.learning_pipeline:
            learning_status = self.learning_pipeline.get_learning_status()
            status["components"]["learning"] = {
                "available": True,
                "active": learning_status["learning_active"],
                "total_experiences": learning_status["total_experiences"],
                "current_version": learning_status["current_version"]
            }

        # Overall health assessment
        component_health = [
            comp.get("resource_safe", True) and comp.get("performance_healthy", True)
            for comp in status["components"].values()
        ]
        status["overall_healthy"] = all(component_health) if component_health else True
        status["rl_enabled"] = any(
            comp.get("rl_enabled", False) for comp in status["components"].values()
        )
        status["safety_level"] = status["components"].get("safety", {}).get("safety_level", "normal")

        return status

    async def get_performance_data(self) -> Dict[str, Any]:
        """Get performance metrics data"""
        if not self.performance_monitor:
            return {"status": "not_available"}

        # Get real-time metrics for different strategies
        performance_data = {
            "timestamp": datetime.now(),
            "strategies": {},
            "summary": {}
        }

        # Get metrics for each strategy
        for strategy in ["rl_enabled", "traditional", "hybrid"]:
            metrics = self.performance_monitor.get_real_time_metrics(strategy, "5m")
            if metrics:
                performance_data["strategies"][strategy] = {
                    "success_rate": metrics.success_rate,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "p95_latency_ms": metrics.p95_latency_ms,
                    "avg_quality": metrics.avg_quality_score,
                    "total_selections": metrics.total_selections
                }

        # Calculate summary statistics
        if performance_data["strategies"]:
            all_metrics = list(performance_data["strategies"].values())
            performance_data["summary"] = {
                "success_rate": sum(m["success_rate"] for m in all_metrics) / len(all_metrics),
                "avg_latency_ms": sum(m["avg_latency_ms"] for m in all_metrics) / len(all_metrics),
                "p95_latency_ms": max(m["p95_latency_ms"] for m in all_metrics),
                "total_selections": sum(m["total_selections"] for m in all_metrics)
            }

        return performance_data

    async def get_safety_data(self) -> Dict[str, Any]:
        """Get safety status data"""
        if not self.safety_manager:
            return {"status": "not_available"}

        return self.safety_manager.get_safety_status()

    def get_alerts_data(self) -> List[Dict[str, Any]]:
        """Get current alerts data"""
        return [
            {
                "alert_id": alert.alert_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity,
                "timestamp": alert.timestamp,
                "source": alert.source,
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved
            }
            for alert in self.alerts
            if not alert.resolved
        ]

    def acknowledge_alert(self, alert_id: str) -> Dict[str, Any]:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} acknowledged")
                return {"status": "acknowledged"}

        raise HTTPException(status_code=404, detail="Alert not found")

    def add_alert_config(self, config: AlertConfig) -> Dict[str, Any]:
        """Add new alert configuration"""
        self.alert_configs.append(config)
        logger.info(f"Added alert configuration: {config.name}")
        return {"status": "created", "config": config.dict()}

    async def apply_system_override(self, override: SystemOverride) -> Dict[str, Any]:
        """Apply system override"""
        logger.warning(f"System override applied: {override.action} on {override.component} - {override.reason}")

        if override.component == "safety" and self.safety_manager:
            if override.action == "enter_safe_mode":
                await self.safety_manager.enter_safe_mode(override.reason)
            elif override.action == "restore_normal":
                await self.safety_manager.restore_normal_operation()
            elif override.action == "emergency_shutdown":
                await self.safety_manager.emergency_shutdown(override.reason)

        elif override.component == "learning" and self.learning_pipeline:
            if override.action == "stop_learning":
                await self.learning_pipeline.stop_continuous_learning()
            elif override.action == "start_learning":
                await self.learning_pipeline.start_continuous_learning()

        return {"status": "applied", "override": override.dict()}

    async def update_traffic_configuration(self, config: TrafficConfig) -> Dict[str, Any]:
        """Update traffic configuration"""
        if not self.hybrid_selector:
            raise HTTPException(status_code=400, detail="Hybrid selector not available")

        if config.strategy == "rl_enabled" and hasattr(self.hybrid_selector, 'current_rl_percentage'):
            self.hybrid_selector.current_rl_percentage = config.percentage / 100.0
            logger.info(f"Updated RL traffic to {config.percentage}%")

        return {"status": "updated", "config": config.dict()}

    async def get_model_history(self) -> List[Dict[str, Any]]:
        """Get model version history"""
        if not self.learning_pipeline:
            return []

        return self.learning_pipeline.get_model_history()

    async def deploy_model_version(self, version_id: str) -> Dict[str, Any]:
        """Deploy specific model version"""
        if not self.learning_pipeline:
            raise HTTPException(status_code=400, detail="Learning pipeline not available")

        # In a real implementation, this would deploy the specified version
        logger.info(f"Deploying model version: {version_id}")
        return {"status": "deployed", "version_id": version_id}

    async def trigger_retraining(self, background_tasks: BackgroundTasks) -> Dict[str, Any]:
        """Trigger model retraining"""
        if not self.learning_pipeline:
            raise HTTPException(status_code=400, detail="Learning pipeline not available")

        async def retrain_task():
            try:
                success = await self.learning_pipeline.trigger_periodic_retrain(force=True)
                if success:
                    await self._broadcast_update({"type": "retrain_complete", "success": True})
                else:
                    await self._broadcast_update({"type": "retrain_failed", "success": False})
            except Exception as e:
                logger.error(f"Retraining failed: {e}")
                await self._broadcast_update({"type": "retrain_failed", "error": str(e)})

        background_tasks.add_task(retrain_task)
        return {"status": "triggered"}

    async def get_ab_test_data(self) -> Dict[str, Any]:
        """Get A/B testing results"""
        if not self.performance_monitor:
            return {"status": "not_available"}

        # Analyze A/B test between RL and traditional strategies
        ab_result = self.performance_monitor.analyze_ab_test("rl_enabled", "traditional", "quality_score")

        return {
            "timestamp": datetime.now(),
            "analysis": ab_result,
            "strategies_compared": ["rl_enabled", "traditional"],
            "metric": "quality_score"
        }

    def add_alert(self, title: str, message: str, severity: str, source: str):
        """Add a new alert"""
        alert = DashboardAlert(
            alert_id=str(uuid.uuid4()),
            title=title,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            source=source
        )

        self.alerts.append(alert)
        logger.info(f"Alert created: {title} ({severity})")

        # Broadcast alert to connected clients
        asyncio.create_task(self._broadcast_alert(alert))

    async def _broadcast_alert(self, alert: DashboardAlert):
        """Broadcast new alert to connected clients"""
        await self.connection_manager.broadcast_json({
            "type": "new_alert",
            "alert": {
                "alert_id": alert.alert_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source
            }
        })

    async def _broadcast_update(self, update_data: Dict[str, Any]):
        """Broadcast system update to connected clients"""
        await self.connection_manager.broadcast_json(update_data)

    async def start_monitoring(self):
        """Start background monitoring and real-time updates"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self._shutdown_event.clear()

        # Start monitoring task
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._background_tasks.append(monitoring_task)

        logger.info("Monitoring dashboard started")

    async def stop_monitoring(self):
        """Stop background monitoring"""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        self._shutdown_event.set()

        # Stop background tasks
        for task in self._background_tasks:
            task.cancel()

        try:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error stopping background tasks: {e}")

        self._background_tasks.clear()
        logger.info("Monitoring dashboard stopped")

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Update monitoring data
                await self._update_monitoring_data()

                # Check for alerts
                await self._check_alerts()

                # Broadcast updates to connected clients
                await self._broadcast_status_update()

                self.last_update_time = datetime.now()

                # Wait for next update
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.update_interval_seconds
                )

            except asyncio.TimeoutError:
                # Expected timeout - continue loop
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    async def _update_monitoring_data(self):
        """Update monitoring data from components"""
        # This method would collect fresh data from all components
        # For now, it's a placeholder
        pass

    async def _check_alerts(self):
        """Check for new alerts based on current metrics"""
        # Check performance metrics
        if self.performance_monitor:
            perf_data = await self.get_performance_data()
            summary = perf_data.get("summary", {})

            # Check success rate
            success_rate = summary.get("success_rate", 1.0)
            if success_rate < 0.8:
                self.add_alert(
                    "Low Success Rate",
                    f"System success rate dropped to {success_rate:.1%}",
                    "warning" if success_rate > 0.7 else "error",
                    "performance_monitor"
                )

            # Check latency
            p95_latency = summary.get("p95_latency_ms", 0)
            if p95_latency > 3000:
                self.add_alert(
                    "High Latency",
                    f"P95 latency is {p95_latency:.0f}ms",
                    "warning" if p95_latency < 5000 else "error",
                    "performance_monitor"
                )

        # Check safety status
        if self.safety_manager:
            safety_data = await self.get_safety_data()
            safety_level = safety_data.get("current_safety_level", "normal")

            if safety_level != "normal":
                self.add_alert(
                    "Safety Mode Active",
                    f"System is in {safety_level} mode",
                    "warning" if safety_level == "degraded" else "critical",
                    "safety_manager"
                )

    async def _broadcast_status_update(self):
        """Broadcast status update to all connected clients"""
        try:
            status_data = {
                "type": "status_update",
                "payload": {
                    "system_status": await self.get_system_status(),
                    "performance": await self.get_performance_data(),
                    "safety": await self.get_safety_data(),
                    "alerts": self.get_alerts_data()
                }
            }

            await self.connection_manager.broadcast_json(status_data)

        except Exception as e:
            logger.error(f"Failed to broadcast status update: {e}")


# Factory function for easy initialization
def create_monitoring_dashboard(
    hybrid_selector: Optional[HybridAgentSelector] = None,
    performance_monitor: Optional[RLPerformanceMonitor] = None,
    safety_manager: Optional[SafetyManager] = None,
    learning_pipeline: Optional[ContinuousLearningPipeline] = None
) -> MonitoringDashboard:
    """Create a monitoring dashboard with provided components"""

    dashboard = MonitoringDashboard(
        hybrid_selector=hybrid_selector,
        performance_monitor=performance_monitor,
        safety_manager=safety_manager,
        learning_pipeline=learning_pipeline
    )

    return dashboard


# Example usage and demonstration
async def demo_monitoring_dashboard():
    """Demonstrate monitoring dashboard functionality"""
    print("Monitoring Dashboard Demo")
    print("=" * 50)

    # Create mock components
    class MockHybridSelector:
        def __init__(self):
            self.rl_selector = True
            self.config = type('Config', (), {'enable_ab_testing': True})()

    class MockPerformanceMonitor:
        def get_real_time_metrics(self, strategy, window):
            import random
            return type('Metrics', (), {
                'success_rate': 0.85 + random.uniform(-0.1, 0.1),
                'avg_latency_ms': 800 + random.uniform(-100, 200),
                'p95_latency_ms': 1200 + random.uniform(-200, 300),
                'avg_quality_score': 0.8 + random.uniform(-0.1, 0.1),
                'total_selections': random.randint(50, 200)
            })()

        def analyze_ab_test(self, strategy_a, strategy_b, metric):
            return {
                "status": "success",
                "strategy_a": strategy_a,
                "strategy_b": strategy_b,
                "improvement_percentage": 12.5,
                "is_statistically_significant": True
            }

    class MockSafetyManager:
        def get_safety_status(self):
            return {
                "current_safety_level": "normal",
                "resource_safe": True,
                "performance_healthy": True,
                "circuit_breakers": {"agent_selection": {"state": "closed"}},
                "total_incidents": 2
            }

    # Create dashboard
    dashboard = create_monitoring_dashboard(
        hybrid_selector=MockHybridSelector(),
        performance_monitor=MockPerformanceMonitor(),
        safety_manager=MockSafetyManager()
    )

    print("Dashboard created")

    # Test API endpoints
    print("\n1. System Status:")
    status = await dashboard.get_system_status()
    print(f"  Overall Health: {status['overall_healthy']}")
    print(f"  RL Enabled: {status['rl_enabled']}")
    print(f"  Safety Level: {status['safety_level']}")

    print("\n2. Performance Data:")
    perf_data = await dashboard.get_performance_data()
    summary = perf_data.get("summary", {})
    print(f"  Success Rate: {summary.get('success_rate', 0):.1%}")
    print(f"  Avg Latency: {summary.get('avg_latency_ms', 0):.0f}ms")

    print("\n3. Safety Data:")
    safety_data = await dashboard.get_safety_data()
    print(f"  Safety Level: {safety_data['current_safety_level']}")
    print(f"  Resource Safe: {safety_data['resource_safe']}")
    print(f"  Performance Healthy: {safety_data['performance_healthy']}")

    # Test alerts
    print("\n4. Alert System:")
    dashboard.add_alert("Test Alert", "This is a test warning", "warning", "demo")
    dashboard.add_alert("Critical Issue", "This is a critical alert", "critical", "demo")

    alerts = dashboard.get_alerts_data()
    print(f"  Active Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"    [{alert['severity']}] {alert['title']}: {alert['message']}")

    print("\nDashboard demo completed")

    if FASTAPI_AVAILABLE:
        print(f"\nTo run the web dashboard:")
        print(f"  uvicorn monitoring_dashboard:app --host 0.0.0.0 --port 8000")
        print(f"  Then visit: http://localhost:8000")

    return dashboard


# FastAPI app instance for running the dashboard
if FASTAPI_AVAILABLE and __name__ == "__main__":
    # Create a standalone dashboard for testing
    app_dashboard = create_monitoring_dashboard()
    app = app_dashboard.app

    @app.on_event("startup")
    async def startup_event():
        await app_dashboard.start_monitoring()
        logger.info("Dashboard server started")

    @app.on_event("shutdown")
    async def shutdown_event():
        await app_dashboard.stop_monitoring()
        logger.info("Dashboard server stopped")

else:
    # Run demo if not starting as web server
    if __name__ == "__main__":
        asyncio.run(demo_monitoring_dashboard())