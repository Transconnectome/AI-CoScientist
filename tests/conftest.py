import os
import pytest

# Set mock environment variables for testing at module level
# This ensures they are set before any application code is imported
os.environ["SECRET_KEY"] = "test_secret_key_at_least_32_chars_long_value"
os.environ["DATABASE_URL"] = "postgresql+asyncpg://user:pass@localhost/db"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["RABBITMQ_URL"] = "amqp://guest:guest@localhost:5672/"
os.environ["CELERY_BROKER_URL"] = "redis://localhost:6379/0"
os.environ["CELERY_RESULT_BACKEND"] = "redis://localhost:6379/0"
os.environ["OPENAI_API_KEY"] = "sk-test"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
os.environ["GEMINI_API_KEY"] = "test-gemini"
os.environ["DEEPSEEK_API_KEY"] = "test-deepseek"

@pytest.fixture(autouse=True)
def mock_env_vars():
    """Fixture to ensure env vars are set (redundant but safe)."""
    pass
