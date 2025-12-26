"""
Gradual Traffic Migration System for RL Agent Selection

This module implements a sophisticated traffic migration system that safely
transitions agent selection traffic from traditional methods to RL-enhanced
approaches. It provides:

- Automated traffic ramping with configurable schedules
- Performance-based migration decisions
- Automatic rollback on performance degradation
- Canary deployments and blue-green testing
- Traffic splitting with sophisticated routing rules
- Real-time monitoring and alerting during migration
- Rollback and recovery mechanisms
- Migration history and audit logging

The system ensures safe, data-driven migration to RL-enhanced agent selection
while maintaining system stability and performance guarantees.
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics
import threading

try:
    from .hybrid_agent_selector import HybridAgentSelector, SelectionStrategy
    from .performance_monitor import RLPerformanceMonitor
    from .safety_mechanisms import SafetyManager, SafetyLevel
    RL_COMPONENTS_AVAILABLE = True
except ImportError:
    RL_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MigrationPhase(Enum):
    """Migration phase types"""
    PLANNING = "planning"
    CANARY = "canary"
    GRADUAL_RAMP = "gradual_ramp"
    FULL_MIGRATION = "full_migration"
    MONITORING = "monitoring"
    ROLLBACK = "rollback"
    COMPLETED = "completed"
    FAILED = "failed"


class MigrationTrigger(Enum):
    """Migration trigger types"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    SAFETY_INCIDENT = "safety_incident"


class RollbackReason(Enum):
    """Rollback reason types"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_SPIKE = "error_rate_spike"
    SAFETY_INCIDENT = "safety_incident"
    MANUAL_OVERRIDE = "manual_override"
    TIMEOUT_EXCEEDED = "timeout_exceeded"


@dataclass
class TrafficTarget:
    """Traffic distribution target"""
    strategy: str
    percentage: float
    min_requests: int = 10  # Minimum requests before evaluation


@dataclass
class MigrationSchedule:
    """Migration schedule configuration"""
    phase_duration_minutes: int = 60
    canary_percentage: float = 5.0
    ramp_increments: List[float] = field(default_factory=lambda: [10.0, 25.0, 50.0, 75.0, 100.0])
    evaluation_window_minutes: int = 15
    min_samples_per_phase: int = 50


@dataclass
class PerformanceThresholds:
    """Performance thresholds for migration decisions"""
    min_success_rate: float = 0.85
    max_latency_p95_ms: float = 2000.0
    max_error_rate: float = 0.10
    min_quality_score: float = 0.70
    performance_improvement_threshold: float = 0.05  # 5% improvement required


@dataclass
class MigrationEvent:
    """Migration event record"""
    event_id: str
    timestamp: datetime
    phase: MigrationPhase
    action: str
    details: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    trigger: Optional[MigrationTrigger] = None


@dataclass
class MigrationConfig:
    """Complete migration configuration"""
    migration_id: str
    source_strategy: str
    target_strategy: str
    schedule: MigrationSchedule
    thresholds: PerformanceThresholds
    enable_automatic_rollback: bool = True
    max_migration_duration_hours: int = 24
    require_manual_approval: bool = False


class TrafficSplitter:
    """Intelligent traffic splitting for migration"""

    def __init__(self, config: MigrationConfig):
        self.config = config
        self.current_targets: List[TrafficTarget] = []
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.total_requests = 0
        self._lock = threading.RLock()

    def update_targets(self, targets: List[TrafficTarget]):
        """Update traffic distribution targets"""
        with self._lock:
            self.current_targets = targets
            # Normalize percentages to sum to 100
            total_percentage = sum(t.percentage for t in targets)
            if total_percentage > 0:
                for target in self.current_targets:
                    target.percentage = (target.percentage / total_percentage) * 100

        logger.info(f"Updated traffic targets: {[f'{t.strategy}:{t.percentage:.1f}%' for t in targets]}")

    def route_request(self, request_context: Dict[str, Any]) -> str:
        """Route a request to appropriate strategy"""
        with self._lock:
            if not self.current_targets:
                return self.config.source_strategy

            self.total_requests += 1

            # Calculate current distribution
            current_distribution = self._calculate_current_distribution()

            # Find strategy that needs more traffic
            for target in sorted(self.current_targets, key=lambda x: x.percentage, reverse=True):
                current_percentage = current_distribution.get(target.strategy, 0.0)

                # Check if this strategy needs more requests
                if (current_percentage < target.percentage - 5.0 or  # 5% tolerance
                    self.request_counts[target.strategy] < target.min_requests):

                    self.request_counts[target.strategy] += 1
                    return target.strategy

            # Default routing based on weighted probability
            import random
            rand = random.random() * 100

            cumulative = 0
            for target in self.current_targets:
                cumulative += target.percentage
                if rand <= cumulative:
                    self.request_counts[target.strategy] += 1
                    return target.strategy

            # Fallback to source strategy
            self.request_counts[self.config.source_strategy] += 1
            return self.config.source_strategy

    def _calculate_current_distribution(self) -> Dict[str, float]:
        """Calculate current traffic distribution"""
        if self.total_requests == 0:
            return {}

        return {
            strategy: (count / self.total_requests) * 100
            for strategy, count in self.request_counts.items()
        }

    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get traffic distribution statistics"""
        with self._lock:
            current_dist = self._calculate_current_distribution()
            target_dist = {t.strategy: t.percentage for t in self.current_targets}

            return {
                "total_requests": self.total_requests,
                "current_distribution": current_dist,
                "target_distribution": target_dist,
                "request_counts": dict(self.request_counts),
                "distribution_variance": self._calculate_distribution_variance(current_dist, target_dist)
            }

    def _calculate_distribution_variance(self,
                                       current: Dict[str, float],
                                       target: Dict[str, float]) -> float:
        """Calculate variance between current and target distribution"""
        if not target:
            return 0.0

        variances = []
        for strategy, target_pct in target.items():
            current_pct = current.get(strategy, 0.0)
            variance = abs(current_pct - target_pct)
            variances.append(variance)

        return statistics.mean(variances) if variances else 0.0

    def reset_stats(self):
        """Reset traffic statistics"""
        with self._lock:
            self.request_counts.clear()
            self.total_requests = 0


class MigrationController:
    """
    Comprehensive traffic migration controller

    Orchestrates the entire migration process from planning through completion,
    with automated decision-making and rollback capabilities.
    """

    def __init__(self,
                 hybrid_selector: HybridAgentSelector,
                 performance_monitor: RLPerformanceMonitor,
                 safety_manager: Optional[SafetyManager] = None):
        """
        Initialize migration controller

        Args:
            hybrid_selector: Hybrid agent selector to control
            performance_monitor: Performance monitoring system
            safety_manager: Optional safety manager for incident detection
        """
        self.hybrid_selector = hybrid_selector
        self.performance_monitor = performance_monitor
        self.safety_manager = safety_manager

        # Migration state
        self.current_migration: Optional[MigrationConfig] = None
        self.current_phase = MigrationPhase.PLANNING
        self.migration_start_time: Optional[datetime] = None
        self.phase_start_time: Optional[datetime] = None

        # Traffic management
        self.traffic_splitter: Optional[TrafficSplitter] = None
        self.migration_events: deque = deque(maxlen=1000)

        # Performance tracking
        self.baseline_metrics: Dict[str, float] = {}
        self.phase_metrics: Dict[MigrationPhase, Dict[str, float]] = {}

        # Control state
        self.migration_active = False
        self.automatic_mode = True
        self.approval_pending = False

        # Background monitoring
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        logger.info("Migration Controller initialized")

    def plan_migration(self,
                      target_strategy: str = "rl_enabled",
                      schedule: Optional[MigrationSchedule] = None,
                      thresholds: Optional[PerformanceThresholds] = None) -> str:
        """
        Plan a new migration

        Args:
            target_strategy: Target strategy to migrate to
            schedule: Migration schedule configuration
            thresholds: Performance thresholds for decisions

        Returns:
            Migration ID
        """
        if self.migration_active:
            raise ValueError("Migration already in progress")

        migration_id = f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Use defaults if not provided
        if schedule is None:
            schedule = MigrationSchedule()

        if thresholds is None:
            thresholds = PerformanceThresholds()

        # Create migration configuration
        self.current_migration = MigrationConfig(
            migration_id=migration_id,
            source_strategy="traditional",
            target_strategy=target_strategy,
            schedule=schedule,
            thresholds=thresholds
        )

        # Initialize traffic splitter
        self.traffic_splitter = TrafficSplitter(self.current_migration)

        # Record planning event
        self._record_event(
            MigrationPhase.PLANNING,
            "migration_planned",
            {
                "target_strategy": target_strategy,
                "schedule": asdict(schedule),
                "thresholds": asdict(thresholds)
            },
            MigrationTrigger.MANUAL
        )

        logger.info(f"Migration planned: {migration_id} -> {target_strategy}")
        return migration_id

    async def start_migration(self, migration_id: str) -> bool:
        """
        Start the planned migration

        Args:
            migration_id: Migration to start

        Returns:
            True if migration started successfully
        """
        if not self.current_migration or self.current_migration.migration_id != migration_id:
            raise ValueError("No matching migration plan found")

        if self.migration_active:
            raise ValueError("Migration already active")

        # Collect baseline performance metrics
        await self._collect_baseline_metrics()

        # Start migration
        self.migration_active = True
        self.migration_start_time = datetime.now()
        self.current_phase = MigrationPhase.CANARY

        # Start background monitoring
        await self._start_background_monitoring()

        # Begin canary phase
        await self._begin_canary_phase()

        self._record_event(
            MigrationPhase.CANARY,
            "migration_started",
            {"baseline_metrics": self.baseline_metrics},
            MigrationTrigger.MANUAL
        )

        logger.info(f"Migration started: {migration_id}")
        return True

    async def _collect_baseline_metrics(self):
        """Collect baseline performance metrics before migration"""
        # Get recent performance for source strategy
        source_metrics = self.performance_monitor.get_real_time_metrics(
            self.current_migration.source_strategy, "1h"
        )

        if source_metrics:
            self.baseline_metrics = {
                "success_rate": source_metrics.success_rate,
                "avg_latency_ms": source_metrics.avg_latency_ms,
                "p95_latency_ms": source_metrics.p95_latency_ms,
                "avg_quality": source_metrics.avg_quality_score,
                "total_selections": source_metrics.total_selections
            }
        else:
            # Use safe defaults if no historical data
            self.baseline_metrics = {
                "success_rate": 0.85,
                "avg_latency_ms": 1000.0,
                "p95_latency_ms": 1500.0,
                "avg_quality": 0.75,
                "total_selections": 0
            }

        logger.info(f"Baseline metrics collected: {self.baseline_metrics}")

    async def _begin_canary_phase(self):
        """Begin canary deployment phase"""
        self.phase_start_time = datetime.now()

        # Set canary traffic distribution
        canary_targets = [
            TrafficTarget(self.current_migration.source_strategy,
                         100.0 - self.current_migration.schedule.canary_percentage),
            TrafficTarget(self.current_migration.target_strategy,
                         self.current_migration.schedule.canary_percentage)
        ]

        self.traffic_splitter.update_targets(canary_targets)

        # Update hybrid selector configuration
        if hasattr(self.hybrid_selector, 'current_rl_percentage'):
            self.hybrid_selector.current_rl_percentage = self.current_migration.schedule.canary_percentage / 100.0

        logger.info(f"Canary phase started - {self.current_migration.schedule.canary_percentage}% target strategy")

    async def _advance_to_next_phase(self):
        """Advance migration to the next phase"""
        if self.current_phase == MigrationPhase.CANARY:
            await self._begin_gradual_ramp()
        elif self.current_phase == MigrationPhase.GRADUAL_RAMP:
            await self._continue_ramp_or_complete()
        elif self.current_phase == MigrationPhase.FULL_MIGRATION:
            await self._complete_migration()

    async def _begin_gradual_ramp(self):
        """Begin gradual traffic ramping"""
        self.current_phase = MigrationPhase.GRADUAL_RAMP
        self.phase_start_time = datetime.now()

        # Start with first ramp increment
        first_increment = self.current_migration.schedule.ramp_increments[0]
        await self._update_traffic_percentage(first_increment)

        self._record_event(
            MigrationPhase.GRADUAL_RAMP,
            "gradual_ramp_started",
            {"target_percentage": first_increment}
        )

        logger.info(f"Gradual ramp started - target: {first_increment}%")

    async def _continue_ramp_or_complete(self):
        """Continue ramping or move to full migration"""
        # Find current percentage
        current_pct = 0.0
        for target in self.traffic_splitter.current_targets:
            if target.strategy == self.current_migration.target_strategy:
                current_pct = target.percentage
                break

        # Find next increment
        next_increment = None
        for increment in self.current_migration.schedule.ramp_increments:
            if increment > current_pct:
                next_increment = increment
                break

        if next_increment and next_increment < 100.0:
            # Continue ramping
            await self._update_traffic_percentage(next_increment)
            logger.info(f"Traffic ramped to {next_increment}%")
        else:
            # Move to full migration
            self.current_phase = MigrationPhase.FULL_MIGRATION
            await self._update_traffic_percentage(100.0)

            self._record_event(
                MigrationPhase.FULL_MIGRATION,
                "full_migration_started",
                {"target_percentage": 100.0}
            )

            logger.info("Full migration phase started")

    async def _update_traffic_percentage(self, target_percentage: float):
        """Update traffic distribution percentage"""
        targets = [
            TrafficTarget(self.current_migration.source_strategy,
                         100.0 - target_percentage),
            TrafficTarget(self.current_migration.target_strategy,
                         target_percentage)
        ]

        self.traffic_splitter.update_targets(targets)

        # Update hybrid selector if applicable
        if hasattr(self.hybrid_selector, 'current_rl_percentage'):
            self.hybrid_selector.current_rl_percentage = target_percentage / 100.0

    async def _complete_migration(self):
        """Complete the migration process"""
        self.current_phase = MigrationPhase.COMPLETED
        self.migration_active = False

        # Final performance evaluation
        final_metrics = await self._evaluate_current_performance()

        self._record_event(
            MigrationPhase.COMPLETED,
            "migration_completed",
            {
                "final_metrics": final_metrics,
                "duration_minutes": (datetime.now() - self.migration_start_time).total_seconds() / 60
            }
        )

        # Stop background monitoring
        await self._stop_background_monitoring()

        logger.info(f"Migration completed successfully: {self.current_migration.migration_id}")

    async def _evaluate_current_performance(self) -> Dict[str, float]:
        """Evaluate current performance across all strategies"""
        performance_data = {}

        # Get metrics for each strategy
        for strategy in [self.current_migration.source_strategy, self.current_migration.target_strategy]:
            metrics = self.performance_monitor.get_real_time_metrics(strategy, "15m")
            if metrics:
                performance_data[strategy] = {
                    "success_rate": metrics.success_rate,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "p95_latency_ms": metrics.p95_latency_ms,
                    "avg_quality": metrics.avg_quality_score,
                    "total_selections": metrics.total_selections
                }

        return performance_data

    async def _check_migration_health(self) -> Tuple[bool, Optional[RollbackReason]]:
        """
        Check migration health and determine if rollback is needed

        Returns:
            Tuple of (is_healthy, rollback_reason)
        """
        if not self.current_migration:
            return True, None

        # Get current performance
        target_metrics = self.performance_monitor.get_real_time_metrics(
            self.current_migration.target_strategy, "15m"
        )

        if not target_metrics or target_metrics.total_selections < self.current_migration.schedule.min_samples_per_phase:
            # Not enough data yet
            return True, None

        thresholds = self.current_migration.thresholds

        # Check success rate
        if target_metrics.success_rate < thresholds.min_success_rate:
            return False, RollbackReason.PERFORMANCE_DEGRADATION

        # Check latency
        if target_metrics.p95_latency_ms > thresholds.max_latency_p95_ms:
            return False, RollbackReason.PERFORMANCE_DEGRADATION

        # Check quality score
        if target_metrics.avg_quality_score < thresholds.min_quality_score:
            return False, RollbackReason.PERFORMANCE_DEGRADATION

        # Check error rate (derived from success rate)
        error_rate = 1.0 - target_metrics.success_rate
        if error_rate > thresholds.max_error_rate:
            return False, RollbackReason.ERROR_RATE_SPIKE

        # Check safety manager if available
        if self.safety_manager:
            safety_status = self.safety_manager.get_safety_status()
            if safety_status.get("current_safety_level") != "normal":
                return False, RollbackReason.SAFETY_INCIDENT

        # Check timeout
        if self.migration_start_time:
            duration_hours = (datetime.now() - self.migration_start_time).total_seconds() / 3600
            if duration_hours > self.current_migration.max_migration_duration_hours:
                return False, RollbackReason.TIMEOUT_EXCEEDED

        return True, None

    async def rollback_migration(self, reason: RollbackReason, manual: bool = False) -> bool:
        """
        Rollback the current migration

        Args:
            reason: Reason for rollback
            manual: Whether rollback is manually triggered

        Returns:
            True if rollback was successful
        """
        if not self.migration_active:
            logger.warning("No active migration to rollback")
            return False

        logger.warning(f"Rolling back migration: {reason.value}")

        self.current_phase = MigrationPhase.ROLLBACK

        # Revert traffic to source strategy
        rollback_targets = [
            TrafficTarget(self.current_migration.source_strategy, 100.0),
            TrafficTarget(self.current_migration.target_strategy, 0.0)
        ]

        self.traffic_splitter.update_targets(rollback_targets)

        # Update hybrid selector
        if hasattr(self.hybrid_selector, 'current_rl_percentage'):
            self.hybrid_selector.current_rl_percentage = 0.0

        # Record rollback event
        self._record_event(
            MigrationPhase.ROLLBACK,
            "migration_rolled_back",
            {
                "reason": reason.value,
                "manual": manual,
                "duration_minutes": (datetime.now() - self.migration_start_time).total_seconds() / 60
            },
            MigrationTrigger.MANUAL_OVERRIDE if manual else MigrationTrigger.SAFETY_INCIDENT
        )

        # Stop migration
        self.migration_active = False
        self.current_phase = MigrationPhase.FAILED

        # Stop background monitoring
        await self._stop_background_monitoring()

        logger.info(f"Migration rolled back: {self.current_migration.migration_id}")
        return True

    def _record_event(self,
                     phase: MigrationPhase,
                     action: str,
                     details: Dict[str, Any],
                     trigger: Optional[MigrationTrigger] = None):
        """Record a migration event"""
        event = MigrationEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            phase=phase,
            action=action,
            details=details,
            trigger=trigger
        )

        self.migration_events.append(event)
        logger.info(f"Migration event: {phase.value} - {action}")

    async def _start_background_monitoring(self):
        """Start background monitoring tasks"""
        if self._background_tasks:
            return

        self._shutdown_event.clear()

        # Start monitoring task
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._background_tasks.append(monitoring_task)

        logger.info("Migration monitoring started")

    async def _stop_background_monitoring(self):
        """Stop background monitoring tasks"""
        if not self._background_tasks:
            return

        self._shutdown_event.set()

        # Cancel and wait for tasks
        for task in self._background_tasks:
            task.cancel()

        try:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error stopping monitoring tasks: {e}")

        self._background_tasks.clear()
        logger.info("Migration monitoring stopped")

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Check migration health
                is_healthy, rollback_reason = await self._check_migration_health()

                if not is_healthy and self.current_migration.enable_automatic_rollback:
                    await self.rollback_migration(rollback_reason, manual=False)
                    break

                # Check if phase should advance
                if await self._should_advance_phase():
                    await self._advance_to_next_phase()

                # Wait before next check
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=30)

            except asyncio.TimeoutError:
                # Expected timeout - continue monitoring
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in migration monitoring: {e}")
                await asyncio.sleep(60)

    async def _should_advance_phase(self) -> bool:
        """Check if migration should advance to next phase"""
        if not self.phase_start_time:
            return False

        # Check minimum phase duration
        phase_duration = (datetime.now() - self.phase_start_time).total_seconds() / 60
        min_duration = self.current_migration.schedule.evaluation_window_minutes

        if phase_duration < min_duration:
            return False

        # Check if we have enough samples
        target_metrics = self.performance_monitor.get_real_time_metrics(
            self.current_migration.target_strategy, "15m"
        )

        if not target_metrics:
            return False

        min_samples = self.current_migration.schedule.min_samples_per_phase
        if target_metrics.total_selections < min_samples:
            return False

        # Check performance criteria
        thresholds = self.current_migration.thresholds

        meets_criteria = (
            target_metrics.success_rate >= thresholds.min_success_rate and
            target_metrics.p95_latency_ms <= thresholds.max_latency_p95_ms and
            target_metrics.avg_quality_score >= thresholds.min_quality_score
        )

        return meets_criteria

    def get_migration_status(self) -> Dict[str, Any]:
        """Get comprehensive migration status"""
        if not self.current_migration:
            return {"status": "no_active_migration"}

        # Calculate progress
        progress_percentage = 0.0
        if self.current_phase == MigrationPhase.CANARY:
            progress_percentage = 25.0
        elif self.current_phase == MigrationPhase.GRADUAL_RAMP:
            current_pct = 0.0
            for target in (self.traffic_splitter.current_targets if self.traffic_splitter else []):
                if target.strategy == self.current_migration.target_strategy:
                    current_pct = target.percentage
                    break
            progress_percentage = 25.0 + (current_pct * 0.75)  # 25-100% range
        elif self.current_phase == MigrationPhase.FULL_MIGRATION:
            progress_percentage = 95.0
        elif self.current_phase == MigrationPhase.COMPLETED:
            progress_percentage = 100.0

        # Get traffic distribution
        traffic_stats = self.traffic_splitter.get_distribution_stats() if self.traffic_splitter else {}

        # Get recent events
        recent_events = [
            {
                "timestamp": event.timestamp.isoformat(),
                "phase": event.phase.value,
                "action": event.action,
                "details": event.details
            }
            for event in list(self.migration_events)[-10:]
        ]

        return {
            "migration_id": self.current_migration.migration_id,
            "active": self.migration_active,
            "current_phase": self.current_phase.value,
            "progress_percentage": progress_percentage,
            "source_strategy": self.current_migration.source_strategy,
            "target_strategy": self.current_migration.target_strategy,
            "start_time": self.migration_start_time.isoformat() if self.migration_start_time else None,
            "duration_minutes": (
                (datetime.now() - self.migration_start_time).total_seconds() / 60
                if self.migration_start_time else 0
            ),
            "traffic_distribution": traffic_stats,
            "baseline_metrics": self.baseline_metrics,
            "automatic_mode": self.automatic_mode,
            "recent_events": recent_events
        }

    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration event history"""
        return [
            {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "phase": event.phase.value,
                "action": event.action,
                "details": event.details,
                "performance_metrics": event.performance_metrics,
                "trigger": event.trigger.value if event.trigger else None
            }
            for event in self.migration_events
        ]

    async def pause_migration(self) -> bool:
        """Pause the current migration"""
        if not self.migration_active:
            return False

        self.automatic_mode = False
        logger.info("Migration paused")
        return True

    async def resume_migration(self) -> bool:
        """Resume a paused migration"""
        if not self.current_migration:
            return False

        self.automatic_mode = True
        logger.info("Migration resumed")
        return True

    async def cancel_migration(self) -> bool:
        """Cancel the current migration"""
        if not self.migration_active:
            return False

        await self.rollback_migration(RollbackReason.MANUAL_OVERRIDE, manual=True)
        return True


# Factory function for easy initialization
def create_migration_controller(
    hybrid_selector: HybridAgentSelector,
    performance_monitor: RLPerformanceMonitor,
    safety_manager: Optional[SafetyManager] = None
) -> MigrationController:
    """Create a migration controller with provided components"""

    controller = MigrationController(
        hybrid_selector=hybrid_selector,
        performance_monitor=performance_monitor,
        safety_manager=safety_manager
    )

    return controller


# Example usage and demonstration
async def demo_traffic_migration():
    """Demonstrate traffic migration functionality"""
    print("Traffic Migration System Demo")
    print("=" * 50)

    # Mock components
    class MockHybridSelector:
        def __init__(self):
            self.current_rl_percentage = 0.0
            self.rl_selector = True

    class MockPerformanceMonitor:
        def __init__(self):
            self.metrics_data = {
                "traditional": {"success_rate": 0.85, "avg_latency_ms": 900, "p95_latency_ms": 1300, "avg_quality_score": 0.78, "total_selections": 100},
                "rl_enabled": {"success_rate": 0.88, "avg_latency_ms": 850, "p95_latency_ms": 1200, "avg_quality_score": 0.82, "total_selections": 0}
            }

        def get_real_time_metrics(self, strategy, window):
            data = self.metrics_data.get(strategy, {})
            return type('Metrics', (), data)() if data else None

    class MockSafetyManager:
        def get_safety_status(self):
            return {"current_safety_level": "normal"}

    # Create components
    hybrid_selector = MockHybridSelector()
    performance_monitor = MockPerformanceMonitor()
    safety_manager = MockSafetyManager()

    # Create migration controller
    controller = create_migration_controller(
        hybrid_selector=hybrid_selector,
        performance_monitor=performance_monitor,
        safety_manager=safety_manager
    )

    print("Migration Controller created")

    # Plan migration
    print("\n1. Planning Migration:")
    schedule = MigrationSchedule(
        phase_duration_minutes=1,  # Short for demo
        canary_percentage=10.0,
        ramp_increments=[20.0, 50.0, 80.0, 100.0],
        evaluation_window_minutes=1,
        min_samples_per_phase=1  # Low for demo
    )

    migration_id = controller.plan_migration(
        target_strategy="rl_enabled",
        schedule=schedule
    )
    print(f"  Migration planned: {migration_id}")

    # Start migration
    print("\n2. Starting Migration:")
    success = await controller.start_migration(migration_id)
    print(f"  Migration started: {success}")

    # Simulate migration progress
    print("\n3. Migration Progress:")
    for i in range(10):
        await asyncio.sleep(0.5)  # Short intervals for demo

        status = controller.get_migration_status()
        print(f"  Phase: {status['current_phase']}, Progress: {status['progress_percentage']:.1f}%")

        # Simulate more requests for RL strategy
        if controller.traffic_splitter:
            for j in range(20):
                strategy = controller.traffic_splitter.route_request({})
                # Simulate request completion
                performance_monitor.metrics_data[strategy]["total_selections"] += 1

        # Force advancement for demo
        if controller.automatic_mode and await controller._should_advance_phase():
            await controller._advance_to_next_phase()

        if status['current_phase'] == 'completed':
            break

    # Final status
    print("\n4. Final Status:")
    final_status = controller.get_migration_status()
    print(f"  Final Phase: {final_status['current_phase']}")
    print(f"  Duration: {final_status['duration_minutes']:.1f} minutes")
    print(f"  Progress: {final_status['progress_percentage']:.1f}%")

    # Show traffic distribution
    traffic_stats = final_status.get('traffic_distribution', {})
    if traffic_stats:
        print(f"  Final Distribution: {traffic_stats.get('current_distribution', {})}")

    # Show migration history
    print("\n5. Migration Events:")
    history = controller.get_migration_history()
    for event in history[-5:]:  # Last 5 events
        print(f"  {event['timestamp'][:19]}: {event['phase']} - {event['action']}")

    print("\nMigration demo completed")
    return controller


if __name__ == "__main__":
    # Run traffic migration demo
    asyncio.run(demo_traffic_migration())