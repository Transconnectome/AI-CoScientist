"""Optimized connection pooling configuration."""

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
from src.core.config import settings


def create_optimized_engine():
    """Create database engine with optimized connection pooling.

    Returns:
        Async SQLAlchemy engine
    """
    # Connection pool configuration
    pool_config = {
        "poolclass": QueuePool,
        "pool_size": settings.database_pool_size,
        "max_overflow": settings.database_max_overflow,
        "pool_pre_ping": True,  # Verify connections before using
        "pool_recycle": 3600,  # Recycle connections after 1 hour
        "pool_timeout": 30,  # Wait up to 30s for connection
        "echo": settings.database_echo,
        "echo_pool": False,
        "connect_args": {
            "server_settings": {
                "application_name": settings.app_name,
                "jit": "off",  # Disable JIT for faster simple queries
            },
            "command_timeout": 60,
            "timeout": 10,
        },
    }

    # Use NullPool for testing to avoid connection issues
    if settings.environment == "testing":
        pool_config["poolclass"] = NullPool
        del pool_config["pool_size"]
        del pool_config["max_overflow"]
        del pool_config["pool_timeout"]

    engine = create_async_engine(
        settings.database_url,
        **pool_config
    )

    return engine


def create_optimized_session_maker(engine):
    """Create session maker with optimized settings.

    Args:
        engine: SQLAlchemy engine

    Returns:
        Async session maker
    """
    return async_sessionmaker(
        engine,
        expire_on_commit=False,  # Don't expire objects after commit
        autoflush=False,  # Manual flush control for performance
        autocommit=False,
    )
