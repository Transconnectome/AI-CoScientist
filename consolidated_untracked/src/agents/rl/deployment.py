"""
Production Deployment Configuration for RL Agent Selection

This module provides comprehensive deployment configuration and utilities
for productionizing the RL-enhanced agent selection system, including:

- Environment configuration management
- Database initialization and migrations
- Service health checks and readiness probes
- Graceful startup and shutdown procedures
- Resource management and scaling configuration
- Integration with existing AI-CoScientist infrastructure

Designed for containerized deployment with Docker/Kubernetes support.
"""

import asyncio
import logging
import os
import sys
import signal
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import yaml
from datetime import datetime
import psutil

try:
    from .hybrid_agent_selector import create_hybrid_selector, HybridConfig
    from .performance_monitor import create_performance_monitor
    from .safety_mechanisms import create_safety_manager
    from .continuous_learning import create_continuous_learning_pipeline
    from .monitoring_dashboard import create_monitoring_dashboard
    from .traffic_migration import create_migration_controller
    from .rl_integration import enhance_agent_pool_with_rl
    RL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    RL_COMPONENTS_AVAILABLE = False
    logging.error(f"RL components not available: {e}")

logger = logging.getLogger(__name__)


class RLSystemConfig:
    """Centralized configuration for RL system deployment"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_data = self._load_config(config_path)
        self.validate_config()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        config = {
            # Core RL settings
            "rl_enabled": os.getenv("RL_ENABLED", "true").lower() == "true",
            "rl_model_path": os.getenv("RL_MODEL_PATH", "/app/models/rl_agent_selection"),
            "rl_training_enabled": os.getenv("RL_TRAINING_ENABLED", "true").lower() == "true",

            # A/B Testing
            "ab_testing_enabled": os.getenv("AB_TESTING_ENABLED", "true").lower() == "true",
            "initial_rl_traffic_pct": float(os.getenv("INITIAL_RL_TRAFFIC_PCT", "10")),
            "max_rl_traffic_pct": float(os.getenv("MAX_RL_TRAFFIC_PCT", "90")),

            # Performance thresholds
            "performance_thresholds": {
                "success_rate_warning": float(os.getenv("SUCCESS_RATE_WARNING", "0.85")),
                "success_rate_critical": float(os.getenv("SUCCESS_RATE_CRITICAL", "0.75")),
                "latency_p95_warning_ms": float(os.getenv("LATENCY_P95_WARNING_MS", "2000")),
                "latency_p95_critical_ms": float(os.getenv("LATENCY_P95_CRITICAL_MS", "5000")),
            },

            # Safety settings
            "safety_enabled": os.getenv("SAFETY_ENABLED", "true").lower() == "true",
            "circuit_breaker_enabled": os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true",
            "rate_limiting_enabled": os.getenv("RATE_LIMITING_ENABLED", "true").lower() == "true",

            # Monitoring
            "monitoring_enabled": os.getenv("MONITORING_ENABLED", "true").lower() == "true",
            "prometheus_enabled": os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true",
            "dashboard_enabled": os.getenv("DASHBOARD_ENABLED", "true").lower() == "true",
            "dashboard_port": int(os.getenv("DASHBOARD_PORT", "8001")),

            # Continuous learning
            "continuous_learning_enabled": os.getenv("CONTINUOUS_LEARNING_ENABLED", "true").lower() == "true",
            "learning_mode": os.getenv("LEARNING_MODE", "hybrid"),  # online_only, periodic_retrain, hybrid
            "retrain_interval_hours": int(os.getenv("RETRAIN_INTERVAL_HOURS", "24")),

            # Storage and persistence
            "model_storage_path": os.getenv("MODEL_STORAGE_PATH", "/app/storage/models"),
            "metrics_storage_path": os.getenv("METRICS_STORAGE_PATH", "/app/storage/metrics"),
            "backup_enabled": os.getenv("BACKUP_ENABLED", "true").lower() == "true",

            # Resource limits
            "max_memory_usage_mb": int(os.getenv("MAX_MEMORY_USAGE_MB", "2048")),
            "max_cpu_usage_percent": float(os.getenv("MAX_CPU_USAGE_PERCENT", "80")),
            "max_concurrent_requests": int(os.getenv("MAX_CONCURRENT_REQUESTS", "100")),

            # Integration settings
            "agent_pool_integration": os.getenv("AGENT_POOL_INTEGRATION", "enhanced").lower(),  # enhanced, replacement
            "existing_agent_pool_backup": os.getenv("AGENT_POOL_BACKUP", "true").lower() == "true",
        }

        # Load from file if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
                config.update(file_config)

        return config

    def validate_config(self):
        """Validate configuration values"""
        errors = []

        # Check percentages are valid
        if not 0 <= self.config_data["initial_rl_traffic_pct"] <= 100:
            errors.append("initial_rl_traffic_pct must be between 0 and 100")

        if not 0 <= self.config_data["max_rl_traffic_pct"] <= 100:
            errors.append("max_rl_traffic_pct must be between 0 and 100")

        # Check paths are writable
        model_path = Path(self.config_data["model_storage_path"])
        try:
            model_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create model storage path: {e}")

        metrics_path = Path(self.config_data["metrics_storage_path"])
        try:
            metrics_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create metrics storage path: {e}")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config_data.get(key, default)

    def get_nested(self, *keys) -> Any:
        """Get nested configuration value"""
        value = self.config_data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value


class RLSystemManager:
    """
    Comprehensive RL system manager for production deployment

    Handles initialization, startup, shutdown, and health monitoring
    of all RL system components.
    """

    def __init__(self, config: RLSystemConfig, agent_pool):
        self.config = config
        self.agent_pool = agent_pool

        # Component instances
        self.enhanced_agent_pool = None
        self.hybrid_selector = None
        self.performance_monitor = None
        self.safety_manager = None
        self.learning_pipeline = None
        self.monitoring_dashboard = None
        self.migration_controller = None

        # System state
        self.initialized = False
        self.running = False
        self.health_status = "unknown"
        self.startup_time = None
        self.shutdown_handlers = []

        logger.info("RL System Manager initialized")

    async def initialize(self) -> bool:
        """Initialize all RL system components"""
        try:
            logger.info("Initializing RL system components...")

            # 1. Create performance monitor
            if self.config.get("monitoring_enabled"):
                self.performance_monitor = create_performance_monitor(
                    enable_prometheus=self.config.get("prometheus_enabled")
                )
                logger.info("Performance monitor initialized")

            # 2. Create safety manager
            if self.config.get("safety_enabled"):
                safety_config = {
                    "max_cpu_usage_percent": self.config.get("max_cpu_usage_percent"),
                    "max_memory_usage_percent": 90.0,  # Leave some headroom
                    "max_concurrent_requests": self.config.get("max_concurrent_requests"),
                }
                self.safety_manager = create_safety_manager(
                    enable_circuit_breakers=self.config.get("circuit_breaker_enabled"),
                    enable_rate_limiting=self.config.get("rate_limiting_enabled"),
                    custom_config=safety_config
                )
                await self.safety_manager.start_monitoring()
                logger.info("Safety manager initialized")

            # 3. Create hybrid selector
            if self.config.get("rl_enabled"):
                hybrid_config = HybridConfig(
                    enable_rl=True,
                    enable_ab_testing=self.config.get("ab_testing_enabled"),
                    rl_traffic_percentage=self.config.get("initial_rl_traffic_pct") / 100.0,
                    rl_model_path=self.config.get("rl_model_path"),
                    enable_performance_monitoring=True,
                )
                self.hybrid_selector = create_hybrid_selector(
                    self.agent_pool,
                    enable_rl=True,
                    enable_ab_testing=self.config.get("ab_testing_enabled")
                )
                logger.info("Hybrid selector initialized")

            # 4. Enhance agent pool
            if self.config.get("agent_pool_integration") == "enhanced":
                self.enhanced_agent_pool = enhance_agent_pool_with_rl(
                    self.agent_pool,
                    {
                        "enable_rl": self.config.get("rl_enabled"),
                        "enable_ab_testing": self.config.get("ab_testing_enabled"),
                        "rl_traffic_percentage": self.config.get("initial_rl_traffic_pct") / 100.0,
                    }
                )
                logger.info("Agent pool enhanced with RL")

            # 5. Create continuous learning pipeline
            if (self.config.get("continuous_learning_enabled") and
                self.hybrid_selector and
                hasattr(self.hybrid_selector, 'rl_selector')):

                self.learning_pipeline = create_continuous_learning_pipeline(
                    self.agent_pool,
                    self.hybrid_selector.rl_selector,
                    learning_mode=self.config.get("learning_mode", "hybrid"),
                    enable_human_feedback=True
                )
                logger.info("Continuous learning pipeline initialized")

            # 6. Create migration controller
            if self.hybrid_selector and self.performance_monitor:
                self.migration_controller = create_migration_controller(
                    self.hybrid_selector,
                    self.performance_monitor,
                    self.safety_manager
                )
                logger.info("Migration controller initialized")

            # 7. Create monitoring dashboard
            if self.config.get("dashboard_enabled"):
                self.monitoring_dashboard = create_monitoring_dashboard(
                    hybrid_selector=self.hybrid_selector,
                    performance_monitor=self.performance_monitor,
                    safety_manager=self.safety_manager,
                    learning_pipeline=self.learning_pipeline
                )
                logger.info("Monitoring dashboard initialized")

            self.initialized = True
            logger.info("RL system initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"RL system initialization failed: {e}")
            await self.cleanup()
            return False

    async def start(self) -> bool:
        """Start all RL system services"""
        if not self.initialized:
            logger.error("Cannot start uninitialized system")
            return False

        try:
            logger.info("Starting RL system services...")
            self.startup_time = datetime.now()

            # Start background monitoring
            if self.performance_monitor:
                await self.performance_monitor.start_background_monitoring()

            if self.learning_pipeline:
                await self.learning_pipeline.start_continuous_learning()

            if self.monitoring_dashboard:
                await self.monitoring_dashboard.start_monitoring()

            # Set up shutdown handlers
            self._setup_shutdown_handlers()

            self.running = True
            self.health_status = "healthy"

            logger.info("RL system started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start RL system: {e}")
            self.health_status = "unhealthy"
            return False

    def _setup_shutdown_handlers(self):
        """Set up graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    async def shutdown(self) -> bool:
        """Gracefully shutdown all RL system services"""
        logger.info("Shutting down RL system...")

        try:
            # Stop background services
            if self.learning_pipeline:
                await self.learning_pipeline.stop_continuous_learning()

            if self.performance_monitor:
                await self.performance_monitor.stop_background_monitoring()

            if self.safety_manager:
                await self.safety_manager.stop_monitoring()

            if self.monitoring_dashboard:
                await self.monitoring_dashboard.stop_monitoring()

            # Save models and state
            await self._save_system_state()

            self.running = False
            self.health_status = "shutdown"

            logger.info("RL system shutdown completed")
            return True

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            return False

    async def _save_system_state(self):
        """Save system state before shutdown"""
        try:
            if self.hybrid_selector and hasattr(self.hybrid_selector, 'save_rl_model'):
                model_path = self.config.get("model_storage_path") + "/final_model"
                self.hybrid_selector.save_rl_model(model_path)
                logger.info("Saved RL model state")

            # Save performance metrics
            if self.performance_monitor:
                metrics_path = Path(self.config.get("metrics_storage_path")) / "final_metrics.json"
                dashboard_data = self.performance_monitor.get_performance_dashboard_data()
                with open(metrics_path, 'w') as f:
                    json.dump(dashboard_data, f, default=str, indent=2)
                logger.info("Saved performance metrics")

        except Exception as e:
            logger.error(f"Failed to save system state: {e}")

    async def cleanup(self):
        """Clean up resources on failure"""
        if self.running:
            await self.shutdown()

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        status = {
            "status": self.health_status,
            "initialized": self.initialized,
            "running": self.running,
            "startup_time": self.startup_time.isoformat() if self.startup_time else None,
            "uptime_seconds": (
                (datetime.now() - self.startup_time).total_seconds()
                if self.startup_time else 0
            ),
            "components": {}
        }

        # Check component health
        if self.hybrid_selector:
            status["components"]["hybrid_selector"] = {
                "status": "healthy" if self.hybrid_selector else "unhealthy",
                "rl_enabled": hasattr(self.hybrid_selector, 'rl_selector') and self.hybrid_selector.rl_selector is not None
            }

        if self.performance_monitor:
            status["components"]["performance_monitor"] = {"status": "healthy"}

        if self.safety_manager:
            safety_status = self.safety_manager.get_safety_status()
            status["components"]["safety_manager"] = {
                "status": "healthy" if safety_status["current_safety_level"] == "normal" else "degraded",
                "safety_level": safety_status["current_safety_level"]
            }

        if self.learning_pipeline:
            learning_status = self.learning_pipeline.get_learning_status()
            status["components"]["learning_pipeline"] = {
                "status": "healthy" if learning_status["learning_active"] else "inactive",
                "total_experiences": learning_status["total_experiences"]
            }

        # System resources
        try:
            status["resources"] = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
        except:
            status["resources"] = {"status": "unavailable"}

        return status

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for monitoring"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "system_health": self.get_health_status(),
            "performance": {},
            "traffic": {},
            "learning": {}
        }

        # Performance metrics
        if self.performance_monitor:
            dashboard_data = self.performance_monitor.get_performance_dashboard_data()
            summary["performance"] = {
                "strategies_active": len(dashboard_data.get("real_time_metrics", {}).get("5m", {})),
                "total_selections_1h": dashboard_data["system_health"]["total_selections_1h"],
                "error_rate_1h": dashboard_data["system_health"]["error_rate_1h"]
            }

        # Traffic distribution
        if self.hybrid_selector:
            perf_report = self.hybrid_selector.get_performance_report()
            summary["traffic"] = {
                "rl_enabled": perf_report["configuration"]["rl_enabled"],
                "current_rl_traffic": perf_report["configuration"].get("current_rl_traffic", 0),
                "ab_testing": perf_report["configuration"].get("ab_testing_enabled", False)
            }

        # Learning status
        if self.learning_pipeline:
            learning_status = self.learning_pipeline.get_learning_status()
            summary["learning"] = {
                "active": learning_status["learning_active"],
                "total_experiences": learning_status["total_experiences"],
                "current_model_version": learning_status["current_version"]
            }

        return summary

    async def validate_deployment(self) -> Dict[str, Any]:
        """Validate deployment readiness"""
        validation_results = {
            "overall_status": "pass",
            "checks": {},
            "recommendations": []
        }

        # Check configuration
        try:
            self.config.validate_config()
            validation_results["checks"]["configuration"] = "pass"
        except Exception as e:
            validation_results["checks"]["configuration"] = f"fail: {e}"
            validation_results["overall_status"] = "fail"

        # Check resource availability
        try:
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory_percent = psutil.virtual_memory().percent

            if cpu_percent > 80:
                validation_results["recommendations"].append("High CPU usage detected")
            if memory_percent > 85:
                validation_results["recommendations"].append("High memory usage detected")

            validation_results["checks"]["resources"] = "pass"
        except:
            validation_results["checks"]["resources"] = "fail: cannot check system resources"

        # Check component availability
        if not RL_COMPONENTS_AVAILABLE:
            validation_results["checks"]["rl_components"] = "fail: RL components not available"
            validation_results["overall_status"] = "fail"
        else:
            validation_results["checks"]["rl_components"] = "pass"

        # Check storage paths
        model_path = Path(self.config.get("model_storage_path"))
        metrics_path = Path(self.config.get("metrics_storage_path"))

        if not model_path.exists() or not os.access(model_path, os.W_OK):
            validation_results["checks"]["model_storage"] = "fail: cannot write to model storage path"
            validation_results["overall_status"] = "fail"
        else:
            validation_results["checks"]["model_storage"] = "pass"

        if not metrics_path.exists() or not os.access(metrics_path, os.W_OK):
            validation_results["checks"]["metrics_storage"] = "fail: cannot write to metrics storage path"
            validation_results["overall_status"] = "fail"
        else:
            validation_results["checks"]["metrics_storage"] = "pass"

        return validation_results


# Factory function for easy deployment
def create_rl_system_manager(agent_pool, config_path: Optional[str] = None) -> RLSystemManager:
    """Create RL system manager with configuration"""
    config = RLSystemConfig(config_path)
    return RLSystemManager(config, agent_pool)


# CLI interface for system management
async def main():
    """Main entry point for RL system deployment"""
    import argparse

    parser = argparse.ArgumentParser(description="RL Agent Selection System Deployment")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--validate", action="store_true", help="Validate deployment only")
    parser.add_argument("--health", action="store_true", help="Check system health")
    parser.add_argument("--metrics", action="store_true", help="Show metrics summary")

    args = parser.parse_args()

    # Mock agent pool for standalone deployment
    class MockAgentPool:
        def __init__(self):
            self.agents = {}

    agent_pool = MockAgentPool()
    manager = create_rl_system_manager(agent_pool, args.config)

    if args.validate:
        print("Validating deployment...")
        validation = await manager.validate_deployment()
        print(f"Validation status: {validation['overall_status']}")
        for check, status in validation['checks'].items():
            print(f"  {check}: {status}")
        if validation['recommendations']:
            print("Recommendations:")
            for rec in validation['recommendations']:
                print(f"  - {rec}")
        return

    if args.health:
        if not manager.initialized:
            await manager.initialize()
        health = manager.get_health_status()
        print(f"System health: {health['status']}")
        print(f"Running: {health['running']}")
        print(f"Uptime: {health['uptime_seconds']:.0f}s")
        return

    if args.metrics:
        if not manager.initialized:
            await manager.initialize()
        metrics = manager.get_metrics_summary()
        print(json.dumps(metrics, indent=2))
        return

    # Full deployment
    print("Initializing RL system...")
    if await manager.initialize():
        print("Starting RL system...")
        if await manager.start():
            print("RL system started successfully")
            print(f"Health status: {manager.get_health_status()['status']}")

            try:
                # Keep running until interrupted
                while manager.running:
                    await asyncio.sleep(10)
                    health = manager.get_health_status()
                    if health["status"] != "healthy":
                        print(f"Health status changed: {health['status']}")

            except KeyboardInterrupt:
                print("Received interrupt signal")
            finally:
                await manager.shutdown()
        else:
            print("Failed to start RL system")
            sys.exit(1)
    else:
        print("Failed to initialize RL system")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())