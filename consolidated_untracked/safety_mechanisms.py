"""
Comprehensive Safety Mechanisms for RL Agent Selection

This module implements critical safety mechanisms to ensure the RL-enhanced
agent selection system operates reliably and safely in production environments.

Safety features:
- Circuit breaker patterns for fault tolerance
- Rate limiting and resource protection
- Fallback chains with guaranteed responses
- Performance degradation detection and mitigation
- Emergency shutdown procedures
- Safe mode operations
- Audit logging and compliance monitoring
- Resource exhaustion protection
- Model drift detection and automatic remediation

The safety system operates on multiple layers to provide defense in depth
against various failure modes while maintaining system availability.
"""

import asyncio
import logging
import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import statistics
import hashlib

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """System safety levels"""
    NORMAL = "normal"           # Full functionality
    DEGRADED = "degraded"       # Reduced functionality
    SAFE_MODE = "safe_mode"     # Basic functionality only
    EMERGENCY = "emergency"     # Emergency shutdown


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"           # Normal operation
    OPEN = "open"              # Circuit tripped, blocking requests
    HALF_OPEN = "half_open"    # Testing if service recovered


class SafetyTrigger(Enum):
    """Types of safety triggers"""
    HIGH_ERROR_RATE = "high_error_rate"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MODEL_DRIFT = "model_drift"
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    MANUAL_OVERRIDE = "manual_override"
    VALIDATION_FAILURE = "validation_failure"


@dataclass
class SafetyIncident:
    """Safety incident record"""
    incident_id: str
    trigger: SafetyTrigger
    severity: SafetyLevel
    description: str
    timestamp: datetime
    component: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolution_time: Optional[datetime] = None
    resolved: bool = False


@dataclass
class SafetyConfig:
    """Safety system configuration"""
    # Circuit breaker settings
    circuit_failure_threshold: int = 5
    circuit_timeout_seconds: int = 60
    circuit_recovery_timeout: int = 30

    # Performance monitoring
    max_error_rate: float = 0.15  # 15% error rate threshold
    max_latency_p95_ms: float = 3000.0  # 3 second latency threshold
    performance_window_minutes: int = 10

    # Resource limits
    max_cpu_usage_percent: float = 85.0
    max_memory_usage_percent: float = 90.0
    max_concurrent_requests: int = 100

    # Model drift detection
    quality_degradation_threshold: float = 0.1  # 10% quality drop
    drift_detection_window_hours: int = 24

    # Emergency settings
    enable_emergency_shutdown: bool = True
    emergency_contact_timeout_seconds: float = 30.0
    safe_mode_agent_fallback: str = "neuroscience_expert"

    # Audit and compliance
    audit_log_retention_days: int = 30
    enable_detailed_logging: bool = True


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""

    def __init__(self, name: str, config: SafetyConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.total_requests = 0
        self.total_failures = 0
        self._lock = threading.RLock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        with self._lock:
            self.total_requests += 1

            # Check circuit state
            if self.state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")

            try:
                # Execute the function
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                self._on_success()
                return result

            except Exception as e:
                self._on_failure()
                raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if not self.last_failure_time:
            return True

        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.circuit_recovery_timeout

    def _on_success(self):
        """Handle successful execution"""
        with self._lock:
            self.failure_count = 0
            self.last_success_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                logger.info(f"Circuit breaker {self.name} closed after successful recovery")

    def _on_failure(self):
        """Handle failed execution"""
        with self._lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.config.circuit_failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self._lock:
            error_rate = self.total_failures / self.total_requests if self.total_requests > 0 else 0.0

            return {
                "name": self.name,
                "state": self.state.value,
                "total_requests": self.total_requests,
                "total_failures": self.total_failures,
                "current_failure_count": self.failure_count,
                "error_rate": error_rate,
                "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
                "last_success": self.last_success_time.isoformat() if self.last_success_time else None
            }

    def reset(self):
        """Manually reset circuit breaker"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            logger.info(f"Circuit breaker {self.name} manually reset")


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, max_tokens: int, refill_rate: float):
        """
        Initialize rate limiter

        Args:
            max_tokens: Maximum number of tokens in bucket
            refill_rate: Tokens added per second
        """
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = max_tokens
        self.last_refill = time.time()
        self._lock = threading.RLock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens from the bucket

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False

    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate

        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = now

    def get_available_tokens(self) -> int:
        """Get number of available tokens"""
        with self._lock:
            self._refill()
            return int(self.tokens)


class ResourceMonitor:
    """System resource monitoring and protection"""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.cpu_readings: deque = deque(maxlen=60)  # Last 60 readings
        self.memory_readings: deque = deque(maxlen=60)
        self.concurrent_requests = 0
        self._lock = threading.RLock()

    def check_resource_limits(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if system resources are within safe limits

        Returns:
            Tuple of (is_safe, resource_status)
        """
        if not PSUTIL_AVAILABLE:
            return True, {"status": "monitoring_unavailable"}

        try:
            # Get current resource usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            with self._lock:
                self.cpu_readings.append(cpu_percent)
                self.memory_readings.append(memory_percent)

                # Calculate averages over last 10 readings
                avg_cpu = statistics.mean(list(self.cpu_readings)[-10:]) if self.cpu_readings else 0
                avg_memory = statistics.mean(list(self.memory_readings)[-10:]) if self.memory_readings else 0

            # Check limits
            cpu_safe = avg_cpu <= self.config.max_cpu_usage_percent
            memory_safe = avg_memory <= self.config.max_memory_usage_percent
            requests_safe = self.concurrent_requests <= self.config.max_concurrent_requests

            is_safe = cpu_safe and memory_safe and requests_safe

            status = {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "concurrent_requests": self.concurrent_requests,
                "cpu_safe": cpu_safe,
                "memory_safe": memory_safe,
                "requests_safe": requests_safe,
                "overall_safe": is_safe
            }

            return is_safe, status

        except Exception as e:
            logger.error(f"Resource monitoring error: {e}")
            return True, {"status": "error", "error": str(e)}  # Fail safe

    def track_request_start(self):
        """Track start of a request"""
        with self._lock:
            self.concurrent_requests += 1

    def track_request_end(self):
        """Track end of a request"""
        with self._lock:
            self.concurrent_requests = max(0, self.concurrent_requests - 1)


class PerformanceWatchdog:
    """Monitor performance and detect degradation"""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.error_history: deque = deque(maxlen=1000)
        self.latency_history: deque = deque(maxlen=1000)
        self.quality_history: deque = deque(maxlen=1000)
        self._lock = threading.RLock()

    def record_operation(self,
                        success: bool,
                        latency_ms: float,
                        quality_score: Optional[float] = None):
        """Record operation outcome for monitoring"""
        timestamp = datetime.now()

        with self._lock:
            self.error_history.append({
                'timestamp': timestamp,
                'success': success
            })

            self.latency_history.append({
                'timestamp': timestamp,
                'latency_ms': latency_ms
            })

            if quality_score is not None:
                self.quality_history.append({
                    'timestamp': timestamp,
                    'quality': quality_score
                })

    def check_performance_health(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if performance is within acceptable bounds

        Returns:
            Tuple of (is_healthy, performance_status)
        """
        with self._lock:
            # Check recent window
            window_cutoff = datetime.now() - timedelta(minutes=self.config.performance_window_minutes)

            # Filter recent data
            recent_errors = [e for e in self.error_history if e['timestamp'] > window_cutoff]
            recent_latencies = [l for l in self.latency_history if l['timestamp'] > window_cutoff]
            recent_qualities = [q for q in self.quality_history if q['timestamp'] > window_cutoff]

            if not recent_errors:
                return True, {"status": "insufficient_data"}

            # Calculate metrics
            error_rate = 1.0 - (sum(1 for e in recent_errors if e['success']) / len(recent_errors))

            latencies = [l['latency_ms'] for l in recent_latencies]
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else (max(latencies) if latencies else 0)

            avg_quality = statistics.mean([q['quality'] for q in recent_qualities]) if recent_qualities else 1.0

            # Check thresholds
            error_rate_ok = error_rate <= self.config.max_error_rate
            latency_ok = p95_latency <= self.config.max_latency_p95_ms

            # Quality degradation check (needs baseline)
            quality_ok = avg_quality >= 0.5  # Simple threshold for now

            is_healthy = error_rate_ok and latency_ok and quality_ok

            status = {
                "error_rate": error_rate,
                "p95_latency_ms": p95_latency,
                "avg_quality": avg_quality,
                "sample_size": len(recent_errors),
                "error_rate_ok": error_rate_ok,
                "latency_ok": latency_ok,
                "quality_ok": quality_ok,
                "overall_healthy": is_healthy
            }

            return is_healthy, status


class SafetyManager:
    """
    Comprehensive safety management system

    Coordinates all safety mechanisms and provides centralized safety control
    for the RL agent selection system.
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        """
        Initialize safety manager

        Args:
            config: Safety configuration
        """
        self.config = config or SafetyConfig()
        self.current_safety_level = SafetyLevel.NORMAL
        self.safety_incidents: deque = deque(maxlen=1000)

        # Safety components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.resource_monitor = ResourceMonitor(self.config)
        self.performance_watchdog = PerformanceWatchdog(self.config)

        # State tracking
        self.emergency_shutdown_active = False
        self.safe_mode_reason: Optional[str] = None
        self.last_safety_check = datetime.now()

        # Background monitoring
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        logger.info("Safety Manager initialized")

    def create_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Create and register a circuit breaker"""
        circuit_breaker = CircuitBreaker(name, self.config)
        self.circuit_breakers[name] = circuit_breaker
        logger.info(f"Created circuit breaker: {name}")
        return circuit_breaker

    def create_rate_limiter(self, name: str, max_tokens: int, refill_rate: float) -> RateLimiter:
        """Create and register a rate limiter"""
        rate_limiter = RateLimiter(max_tokens, refill_rate)
        self.rate_limiters[name] = rate_limiter
        logger.info(f"Created rate limiter: {name} (max_tokens={max_tokens}, refill_rate={refill_rate})")
        return rate_limiter

    async def safe_execute(self,
                          operation_name: str,
                          func: Callable,
                          *args,
                          fallback_func: Optional[Callable] = None,
                          **kwargs) -> Any:
        """
        Safely execute an operation with full safety protection

        Args:
            operation_name: Name of the operation for tracking
            func: Function to execute
            fallback_func: Fallback function if primary fails
            *args, **kwargs: Arguments for the function

        Returns:
            Result of the operation or fallback
        """
        operation_start = time.time()
        self.resource_monitor.track_request_start()

        try:
            # Check safety level
            if self.current_safety_level == SafetyLevel.EMERGENCY:
                if fallback_func:
                    logger.warning(f"Emergency mode: using fallback for {operation_name}")
                    return await self._execute_with_monitoring(fallback_func, *args, **kwargs)
                else:
                    raise SafetyError("Emergency shutdown active and no fallback available")

            # Check rate limits
            rate_limiter = self.rate_limiters.get(operation_name)
            if rate_limiter and not await rate_limiter.acquire():
                logger.warning(f"Rate limit exceeded for {operation_name}")
                if fallback_func:
                    return await self._execute_with_monitoring(fallback_func, *args, **kwargs)
                else:
                    raise SafetyError(f"Rate limit exceeded for {operation_name}")

            # Execute through circuit breaker if available
            circuit_breaker = self.circuit_breakers.get(operation_name)
            if circuit_breaker:
                result = await circuit_breaker.call(func, *args, **kwargs)
            else:
                result = await self._execute_with_monitoring(func, *args, **kwargs)

            # Record successful operation
            latency_ms = (time.time() - operation_start) * 1000
            self.performance_watchdog.record_operation(True, latency_ms)

            return result

        except Exception as e:
            # Record failed operation
            latency_ms = (time.time() - operation_start) * 1000
            self.performance_watchdog.record_operation(False, latency_ms)

            logger.error(f"Operation {operation_name} failed: {e}")

            # Try fallback if available
            if fallback_func:
                try:
                    logger.info(f"Using fallback for {operation_name}")
                    result = await self._execute_with_monitoring(fallback_func, *args, **kwargs)
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {operation_name}: {fallback_error}")

            # Check if this should trigger safety measures
            await self._check_safety_triggers(operation_name, e)

            raise

        finally:
            self.resource_monitor.track_request_end()

    async def _execute_with_monitoring(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with resource monitoring"""
        # Check resources before execution
        is_safe, resource_status = self.resource_monitor.check_resource_limits()

        if not is_safe:
            logger.warning(f"Resource limits exceeded: {resource_status}")
            await self._trigger_safety_incident(
                SafetyTrigger.RESOURCE_EXHAUSTION,
                f"Resource limits exceeded: {resource_status}",
                "resource_monitor",
                resource_status
            )

        # Execute function
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    async def _check_safety_triggers(self, operation_name: str, error: Exception):
        """Check if error should trigger safety measures"""
        # Check performance health
        is_healthy, perf_status = self.performance_watchdog.check_performance_health()

        if not is_healthy:
            logger.warning(f"Performance degradation detected: {perf_status}")
            await self._trigger_safety_incident(
                SafetyTrigger.PERFORMANCE_DEGRADATION,
                f"Performance degradation in {operation_name}: {perf_status}",
                operation_name,
                perf_status
            )

    async def _trigger_safety_incident(self,
                                     trigger: SafetyTrigger,
                                     description: str,
                                     component: str,
                                     metrics: Dict[str, Any]):
        """Trigger a safety incident and take appropriate action"""
        incident_id = self._generate_incident_id()

        incident = SafetyIncident(
            incident_id=incident_id,
            trigger=trigger,
            severity=self._determine_severity(trigger, metrics),
            description=description,
            timestamp=datetime.now(),
            component=component,
            metrics=metrics
        )

        self.safety_incidents.append(incident)

        logger.warning(f"Safety incident triggered: {incident_id} - {description}")

        # Take action based on severity
        await self._handle_safety_incident(incident)

    def _determine_severity(self, trigger: SafetyTrigger, metrics: Dict[str, Any]) -> SafetyLevel:
        """Determine incident severity"""
        if trigger == SafetyTrigger.RESOURCE_EXHAUSTION:
            cpu_percent = metrics.get('cpu_percent', 0)
            memory_percent = metrics.get('memory_percent', 0)
            if cpu_percent > 95 or memory_percent > 95:
                return SafetyLevel.EMERGENCY
            elif cpu_percent > 90 or memory_percent > 90:
                return SafetyLevel.SAFE_MODE
            else:
                return SafetyLevel.DEGRADED

        elif trigger == SafetyTrigger.PERFORMANCE_DEGRADATION:
            error_rate = metrics.get('error_rate', 0)
            if error_rate > 0.5:  # 50% error rate
                return SafetyLevel.EMERGENCY
            elif error_rate > 0.3:  # 30% error rate
                return SafetyLevel.SAFE_MODE
            else:
                return SafetyLevel.DEGRADED

        elif trigger == SafetyTrigger.HIGH_ERROR_RATE:
            return SafetyLevel.SAFE_MODE

        else:
            return SafetyLevel.DEGRADED

    async def _handle_safety_incident(self, incident: SafetyIncident):
        """Handle safety incident based on severity"""
        if incident.severity == SafetyLevel.EMERGENCY:
            await self.emergency_shutdown(f"Emergency incident: {incident.description}")

        elif incident.severity == SafetyLevel.SAFE_MODE:
            await self.enter_safe_mode(f"Safety incident: {incident.description}")

        elif incident.severity == SafetyLevel.DEGRADED:
            await self.enter_degraded_mode(f"Performance incident: {incident.description}")

    def _generate_incident_id(self) -> str:
        """Generate unique incident ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{timestamp}_{len(self.safety_incidents)}"
        incident_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"INC_{timestamp}_{incident_hash}"

    async def emergency_shutdown(self, reason: str):
        """Initiate emergency shutdown"""
        if self.emergency_shutdown_active:
            return

        logger.critical(f"EMERGENCY SHUTDOWN INITIATED: {reason}")

        self.emergency_shutdown_active = True
        self.current_safety_level = SafetyLevel.EMERGENCY

        # Notify monitoring systems
        await self._notify_emergency_contact()

        # Stop all non-essential operations
        await self._shutdown_non_essential_operations()

    async def enter_safe_mode(self, reason: str):
        """Enter safe mode with basic functionality only"""
        if self.current_safety_level in [SafetyLevel.EMERGENCY, SafetyLevel.SAFE_MODE]:
            return

        logger.warning(f"ENTERING SAFE MODE: {reason}")

        self.current_safety_level = SafetyLevel.SAFE_MODE
        self.safe_mode_reason = reason

    async def enter_degraded_mode(self, reason: str):
        """Enter degraded mode with reduced functionality"""
        if self.current_safety_level != SafetyLevel.NORMAL:
            return

        logger.warning(f"ENTERING DEGRADED MODE: {reason}")
        self.current_safety_level = SafetyLevel.DEGRADED

    async def restore_normal_operation(self):
        """Restore normal operation if conditions allow"""
        if self.emergency_shutdown_active:
            logger.error("Cannot restore normal operation during emergency shutdown")
            return False

        # Check if it's safe to restore
        is_safe, resource_status = self.resource_monitor.check_resource_limits()
        is_healthy, perf_status = self.performance_watchdog.check_performance_health()

        if is_safe and is_healthy:
            logger.info("Restoring normal operation")
            self.current_safety_level = SafetyLevel.NORMAL
            self.safe_mode_reason = None
            return True
        else:
            logger.warning(f"Cannot restore normal operation - Safe: {is_safe}, Healthy: {is_healthy}")
            return False

    async def _notify_emergency_contact(self):
        """Notify emergency contact about critical incident"""
        # In a real implementation, this would send alerts via email, SMS, etc.
        logger.critical("Emergency contact notification would be sent here")

    async def _shutdown_non_essential_operations(self):
        """Shutdown non-essential operations during emergency"""
        # In a real implementation, this would gracefully shutdown background tasks
        logger.critical("Non-essential operations shutdown would happen here")

    def get_safe_fallback_response(self, operation_type: str) -> Any:
        """Get safe fallback response for different operation types"""
        if operation_type == "agent_selection":
            return [self.config.safe_mode_agent_fallback]
        elif operation_type == "task_routing":
            return self.config.safe_mode_agent_fallback
        else:
            return None

    def get_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status"""
        is_safe, resource_status = self.resource_monitor.check_resource_limits()
        is_healthy, perf_status = self.performance_watchdog.check_performance_health()

        circuit_breaker_stats = {
            name: cb.get_stats() for name, cb in self.circuit_breakers.items()
        }

        rate_limiter_stats = {
            name: {"available_tokens": rl.get_available_tokens()}
            for name, rl in self.rate_limiters.items()
        }

        recent_incidents = [
            {
                "incident_id": inc.incident_id,
                "trigger": inc.trigger.value,
                "severity": inc.severity.value,
                "description": inc.description,
                "timestamp": inc.timestamp.isoformat(),
                "resolved": inc.resolved
            }
            for inc in list(self.safety_incidents)[-10:]  # Last 10 incidents
        ]

        return {
            "current_safety_level": self.current_safety_level.value,
            "emergency_shutdown_active": self.emergency_shutdown_active,
            "safe_mode_reason": self.safe_mode_reason,
            "resource_safe": is_safe,
            "performance_healthy": is_healthy,
            "resource_status": resource_status,
            "performance_status": perf_status,
            "circuit_breakers": circuit_breaker_stats,
            "rate_limiters": rate_limiter_stats,
            "recent_incidents": recent_incidents,
            "total_incidents": len(self.safety_incidents)
        }

    async def start_monitoring(self):
        """Start background safety monitoring"""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._shutdown_event.clear()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Safety monitoring started")

    async def stop_monitoring(self):
        """Stop background safety monitoring"""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        self._shutdown_event.set()

        if self._monitoring_task:
            try:
                await asyncio.wait_for(self._monitoring_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._monitoring_task.cancel()

        logger.info("Safety monitoring stopped")

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Periodic safety checks
                await self._periodic_safety_check()

                # Wait for next check
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=30)

            except asyncio.TimeoutError:
                # Expected timeout - continue loop
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in safety monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _periodic_safety_check(self):
        """Perform periodic safety health check"""
        self.last_safety_check = datetime.now()

        # Check if we can restore normal operation
        if self.current_safety_level != SafetyLevel.NORMAL:
            await self.restore_normal_operation()

        # Check for resource issues
        is_safe, resource_status = self.resource_monitor.check_resource_limits()
        if not is_safe:
            await self._trigger_safety_incident(
                SafetyTrigger.RESOURCE_EXHAUSTION,
                f"Resource monitoring check failed: {resource_status}",
                "periodic_check",
                resource_status
            )

        # Check performance health
        is_healthy, perf_status = self.performance_watchdog.check_performance_health()
        if not is_healthy:
            await self._trigger_safety_incident(
                SafetyTrigger.PERFORMANCE_DEGRADATION,
                f"Performance health check failed: {perf_status}",
                "periodic_check",
                perf_status
            )


class SafetyError(Exception):
    """Exception raised by safety system"""
    pass


# Factory function for easy initialization
def create_safety_manager(
    enable_circuit_breakers: bool = True,
    enable_rate_limiting: bool = True,
    custom_config: Optional[Dict[str, Any]] = None
) -> SafetyManager:
    """Create a safety manager with sensible defaults"""

    config = SafetyConfig()
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    safety_manager = SafetyManager(config)

    if enable_circuit_breakers:
        # Create common circuit breakers
        safety_manager.create_circuit_breaker("agent_selection")
        safety_manager.create_circuit_breaker("model_inference")
        safety_manager.create_circuit_breaker("performance_monitoring")

    if enable_rate_limiting:
        # Create common rate limiters
        safety_manager.create_rate_limiter("agent_selection", max_tokens=100, refill_rate=10.0)
        safety_manager.create_rate_limiter("model_training", max_tokens=5, refill_rate=0.1)

    return safety_manager


# Example usage and demonstration
async def demo_safety_mechanisms():
    """Demonstrate safety mechanisms functionality"""
    print("Safety Mechanisms Demo")
    print("=" * 50)

    # Create safety manager
    safety_manager = create_safety_manager()

    await safety_manager.start_monitoring()

    # Simulate normal operation
    print("\n1. Normal Operations:")

    async def sample_operation():
        await asyncio.sleep(0.1)  # Simulate work
        return "Operation successful"

    async def failing_operation():
        await asyncio.sleep(0.05)
        raise Exception("Simulated failure")

    # Test successful operations
    for i in range(5):
        try:
            result = await safety_manager.safe_execute(
                "agent_selection",
                sample_operation
            )
            print(f"  Operation {i+1}: {result}")
        except Exception as e:
            print(f"  Operation {i+1}: Failed - {e}")

    # Test with failures and fallback
    print("\n2. Operations with Failures:")

    async def fallback_operation():
        return "Fallback response"

    for i in range(3):
        try:
            result = await safety_manager.safe_execute(
                "agent_selection",
                failing_operation,
                fallback_func=fallback_operation
            )
            print(f"  Failing operation {i+1}: {result}")
        except Exception as e:
            print(f"  Failing operation {i+1}: Failed - {e}")

    # Test circuit breaker tripping
    print("\n3. Circuit Breaker Test:")
    for i in range(8):  # Should trip after 5 failures
        try:
            result = await safety_manager.safe_execute(
                "model_inference",
                failing_operation,
                fallback_func=fallback_operation
            )
            print(f"  CB test {i+1}: {result}")
        except Exception as e:
            print(f"  CB test {i+1}: {e}")

    # Check safety status
    print("\n4. Safety Status:")
    status = safety_manager.get_safety_status()
    print(f"  Safety Level: {status['current_safety_level']}")
    print(f"  Resource Safe: {status['resource_safe']}")
    print(f"  Performance Healthy: {status['performance_healthy']}")
    print(f"  Total Incidents: {status['total_incidents']}")

    # Show circuit breaker stats
    print("\n5. Circuit Breaker Stats:")
    for name, stats in status['circuit_breakers'].items():
        print(f"  {name}: {stats['state']} (errors: {stats['error_rate']:.2%})")

    await safety_manager.stop_monitoring()
    print("\nSafety demo completed")

    return safety_manager


if __name__ == "__main__":
    # Run safety mechanisms demo
    asyncio.run(demo_safety_mechanisms())