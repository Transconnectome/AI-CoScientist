"""Performance monitoring and optimization utilities."""

import functools
import time
from typing import Any, Callable, Optional
from contextlib import asynccontextmanager

import structlog
from prometheus_client import Counter, Histogram, Gauge

logger = structlog.get_logger(__name__)

# Prometheus metrics
request_count = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

request_duration = Histogram(
    "api_request_duration_seconds",
    "API request duration",
    ["method", "endpoint"]
)

active_requests = Gauge(
    "api_active_requests",
    "Active API requests",
    ["endpoint"]
)

db_query_count = Counter(
    "db_queries_total",
    "Total database queries",
    ["operation", "model"]
)

db_query_duration = Histogram(
    "db_query_duration_seconds",
    "Database query duration",
    ["operation", "model"]
)

cache_hit_count = Counter(
    "cache_hits_total",
    "Cache hits",
    ["cache_type"]
)

cache_miss_count = Counter(
    "cache_misses_total",
    "Cache misses",
    ["cache_type"]
)


def track_time(operation: str, model: Optional[str] = None):
    """Decorator to track operation execution time.

    Args:
        operation: Operation name (e.g., 'query', 'insert', 'update')
        model: Model name (optional)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                if model:
                    db_query_duration.labels(
                        operation=operation,
                        model=model
                    ).observe(duration)
                    db_query_count.labels(
                        operation=operation,
                        model=model
                    ).inc()

                logger.debug(
                    "operation_completed",
                    operation=operation,
                    model=model,
                    duration=duration
                )

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                if model:
                    db_query_duration.labels(
                        operation=operation,
                        model=model
                    ).observe(duration)
                    db_query_count.labels(
                        operation=operation,
                        model=model
                    ).inc()

                logger.debug(
                    "operation_completed",
                    operation=operation,
                    model=model,
                    duration=duration
                )

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@asynccontextmanager
async def track_request(method: str, endpoint: str):
    """Context manager to track HTTP request metrics.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
    """
    active_requests.labels(endpoint=endpoint).inc()
    start = time.time()
    status = "success"

    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.time() - start
        active_requests.labels(endpoint=endpoint).dec()
        request_count.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
        request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)


def track_cache(cache_type: str = "redis"):
    """Decorator to track cache hits and misses.

    Args:
        cache_type: Type of cache (redis, memory, etc.)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await func(*args, **kwargs)

            # Assume result tuple: (value, hit: bool)
            if isinstance(result, tuple) and len(result) == 2:
                value, hit = result
                if hit:
                    cache_hit_count.labels(cache_type=cache_type).inc()
                else:
                    cache_miss_count.labels(cache_type=cache_type).inc()
                return value

            return result

        return wrapper
    return decorator
