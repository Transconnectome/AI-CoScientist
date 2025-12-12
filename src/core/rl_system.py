"""
RL System Integration Module

Integrates the RL-enhanced agent selection system with the main AI-CoScientist
application, providing seamless integration with the existing architecture.

This module handles:
- RL system initialization and configuration
- Integration with existing agent pools
- Health monitoring and metrics exposure
- Graceful startup and shutdown procedures
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from src.core.config import settings

logger = logging.getLogger(__name__)

# Global RL system manager instance
_rl_manager: Optional['RLSystemManager'] = None
_rl_enabled = False

try:
    from src.agents.rl.deployment import create_rl_system_manager, RLSystemConfig
    from src.agents.rl_integration import enhance_agent_pool_with_rl
    RL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RL system not available: {e}")
    RL_AVAILABLE = False


async def initialize_rl_system(agent_pool=None) -> bool:
    """
    Initialize the RL system with the main application

    Args:
        agent_pool: Existing agent pool to enhance

    Returns:
        True if RL system was successfully initialized
    """
    global _rl_manager, _rl_enabled

    if not RL_AVAILABLE:
        logger.info("RL system not available, skipping initialization")
        return False

    if not getattr(settings, 'rl_enabled', True):
        logger.info("RL system disabled in configuration")
        return False

    try:
        logger.info("Initializing RL system...")

        # Create RL system manager
        config_path = getattr(settings, 'rl_config_path', None)
        _rl_manager = create_rl_system_manager(agent_pool, config_path)

        # Initialize and start RL system
        if await _rl_manager.initialize():
            if await _rl_manager.start():
                _rl_enabled = True
                logger.info("RL system initialized successfully")
                return True
            else:
                logger.error("Failed to start RL system")
                return False
        else:
            logger.error("Failed to initialize RL system")
            return False

    except Exception as e:
        logger.error(f"RL system initialization failed: {e}")
        _rl_enabled = False
        return False


async def shutdown_rl_system():
    """Shutdown the RL system gracefully"""
    global _rl_manager, _rl_enabled

    if _rl_manager and _rl_enabled:
        logger.info("Shutting down RL system...")
        try:
            await _rl_manager.shutdown()
            _rl_enabled = False
            logger.info("RL system shutdown completed")
        except Exception as e:
            logger.error(f"Error during RL system shutdown: {e}")


def get_rl_system_manager():
    """Get the global RL system manager instance"""
    return _rl_manager if _rl_enabled else None


def is_rl_enabled() -> bool:
    """Check if RL system is enabled and running"""
    return _rl_enabled and _rl_manager is not None


def get_rl_health_status() -> Dict[str, Any]:
    """Get RL system health status"""
    if not is_rl_enabled():
        return {
            "status": "disabled",
            "available": RL_AVAILABLE,
            "enabled": False
        }

    try:
        return _rl_manager.get_health_status()
    except Exception as e:
        logger.error(f"Failed to get RL health status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def get_rl_metrics() -> Dict[str, Any]:
    """Get RL system metrics"""
    if not is_rl_enabled():
        return {
            "status": "disabled",
            "metrics": {}
        }

    try:
        return _rl_manager.get_metrics_summary()
    except Exception as e:
        logger.error(f"Failed to get RL metrics: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def enhance_agent_pool_if_available(agent_pool):
    """
    Enhance agent pool with RL capabilities if available

    Args:
        agent_pool: Original agent pool

    Returns:
        Enhanced agent pool or original if RL not available
    """
    if not RL_AVAILABLE or not is_rl_enabled():
        logger.info("RL system not available, using original agent pool")
        return agent_pool

    try:
        # Get RL configuration from manager
        rl_config = {
            "enable_rl": True,
            "enable_ab_testing": getattr(settings, 'rl_ab_testing_enabled', True),
            "rl_traffic_percentage": getattr(settings, 'rl_initial_traffic_pct', 10) / 100.0,
        }

        enhanced_pool = enhance_agent_pool_with_rl(agent_pool, rl_config)
        logger.info("Agent pool enhanced with RL capabilities")
        return enhanced_pool

    except Exception as e:
        logger.error(f"Failed to enhance agent pool with RL: {e}")
        return agent_pool


@asynccontextmanager
async def rl_lifespan_manager(app, agent_pool=None):
    """
    Context manager for RL system lifespan management

    Use this in FastAPI lifespan to properly manage RL system startup/shutdown
    """
    # Startup
    await initialize_rl_system(agent_pool)

    try:
        yield
    finally:
        # Shutdown
        await shutdown_rl_system()


# Health check endpoint decorator
def add_rl_health_endpoints(app):
    """Add RL system health endpoints to FastAPI app"""

    @app.get("/health/rl")
    async def rl_health_check():
        """RL system health check endpoint"""
        return get_rl_health_status()

    @app.get("/metrics/rl")
    async def rl_metrics_endpoint():
        """RL system metrics endpoint"""
        return get_rl_metrics()

    @app.get("/rl/status")
    async def rl_status_endpoint():
        """Comprehensive RL system status"""
        return {
            "available": RL_AVAILABLE,
            "enabled": is_rl_enabled(),
            "health": get_rl_health_status(),
            "metrics": get_rl_metrics()
        }


# Configuration validation
def validate_rl_config() -> Dict[str, Any]:
    """Validate RL system configuration"""
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "config": {}
    }

    if not RL_AVAILABLE:
        validation_result["warnings"].append("RL components not installed")
        validation_result["config"]["available"] = False
        return validation_result

    # Check configuration values
    config_checks = [
        ("rl_enabled", bool, True),
        ("rl_ab_testing_enabled", bool, True),
        ("rl_initial_traffic_pct", (int, float), 10),
        ("rl_max_traffic_pct", (int, float), 90),
        ("rl_model_path", str, "/app/models/rl_agent_selection"),
        ("rl_config_path", str, None),
    ]

    for attr_name, expected_type, default_value in config_checks:
        value = getattr(settings, attr_name, default_value)
        validation_result["config"][attr_name] = value

        if value is not None and not isinstance(value, expected_type):
            validation_result["errors"].append(
                f"{attr_name} should be {expected_type.__name__}, got {type(value).__name__}"
            )
            validation_result["valid"] = False

    # Check percentage values are in valid range
    initial_pct = validation_result["config"].get("rl_initial_traffic_pct", 10)
    max_pct = validation_result["config"].get("rl_max_traffic_pct", 90)

    if not (0 <= initial_pct <= 100):
        validation_result["errors"].append("rl_initial_traffic_pct must be between 0 and 100")
        validation_result["valid"] = False

    if not (0 <= max_pct <= 100):
        validation_result["errors"].append("rl_max_traffic_pct must be between 0 and 100")
        validation_result["valid"] = False

    if initial_pct > max_pct:
        validation_result["warnings"].append("rl_initial_traffic_pct is higher than rl_max_traffic_pct")

    return validation_result


# Utility functions for integration
async def migrate_to_rl(target_percentage: float = 100.0) -> Dict[str, Any]:
    """
    Start traffic migration to RL system

    Args:
        target_percentage: Target percentage of traffic for RL

    Returns:
        Migration status
    """
    if not is_rl_enabled():
        return {"status": "error", "message": "RL system not enabled"}

    try:
        manager = get_rl_system_manager()
        if not manager or not manager.migration_controller:
            return {"status": "error", "message": "Migration controller not available"}

        # Plan and start migration
        migration_id = manager.migration_controller.plan_migration(
            target_strategy="rl_enabled"
        )

        success = await manager.migration_controller.start_migration(migration_id)

        return {
            "status": "success" if success else "error",
            "migration_id": migration_id,
            "target_percentage": target_percentage
        }

    except Exception as e:
        logger.error(f"Failed to start RL migration: {e}")
        return {"status": "error", "message": str(e)}


async def rollback_from_rl() -> Dict[str, Any]:
    """
    Rollback from RL to traditional agent selection

    Returns:
        Rollback status
    """
    if not is_rl_enabled():
        return {"status": "error", "message": "RL system not enabled"}

    try:
        manager = get_rl_system_manager()
        if not manager or not manager.migration_controller:
            return {"status": "error", "message": "Migration controller not available"}

        from src.agents.rl.traffic_migration import RollbackReason
        success = await manager.migration_controller.rollback_migration(
            RollbackReason.MANUAL_OVERRIDE, manual=True
        )

        return {"status": "success" if success else "error"}

    except Exception as e:
        logger.error(f"Failed to rollback from RL: {e}")
        return {"status": "error", "message": str(e)}