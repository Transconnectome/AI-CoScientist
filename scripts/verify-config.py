#!/usr/bin/env python3
"""Verify configuration and API keys."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_env_file():
    """Check if .env file exists."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print("❌ .env file not found!")
        return False
    print("✅ .env file exists")
    return True

def load_config():
    """Load configuration and check critical settings."""
    try:
        from dotenv import load_dotenv
        load_dotenv()

        from src.core.config import settings

        print("\n📋 Configuration Status:")
        print("=" * 50)

        # Application
        print(f"\n🏗️  Application:")
        print(f"   Name: {settings.app_name}")
        print(f"   Version: {settings.app_version}")
        print(f"   Environment: {settings.environment}")

        # API Keys
        print(f"\n🔑 API Keys:")
        api_keys = {
            "OpenAI": settings.openai_api_key,
            "Anthropic": settings.anthropic_api_key,
        }

        for name, key in api_keys.items():
            if key and len(key) > 20:
                masked_key = f"{key[:10]}...{key[-10:]}"
                print(f"   ✅ {name}: {masked_key}")
            else:
                print(f"   ❌ {name}: Not configured or invalid")

        # LLM Configuration
        print(f"\n🤖 LLM Configuration:")
        print(f"   Primary Provider: {settings.llm_primary_provider}")
        print(f"   Fallback Provider: {settings.llm_fallback_provider}")
        print(f"   OpenAI Model: {settings.openai_model}")
        print(f"   Anthropic Model: {settings.anthropic_model}")
        print(f"   Cache Enabled: {settings.llm_cache_enabled}")

        # Database
        print(f"\n🗄️  Database:")
        db_url = settings.database_url
        if "postgresql" in db_url:
            # Mask password in URL
            if "@" in db_url:
                parts = db_url.split("@")
                masked_url = f"{parts[0].split(':')[0]}://***:***@{parts[1]}"
                print(f"   URL: {masked_url}")
            else:
                print(f"   URL: {db_url}")
            print(f"   ✅ PostgreSQL configured")
        else:
            print(f"   ❌ Invalid database URL")

        # Redis
        print(f"\n⚡ Redis:")
        print(f"   URL: {settings.redis_url}")
        print(f"   Cache TTL: {settings.redis_cache_ttl}s")

        # Celery
        print(f"\n📋 Celery:")
        print(f"   Broker: {settings.celery_broker_url}")
        print(f"   Backend: {settings.celery_result_backend}")

        # Vector Database
        print(f"\n🔍 Vector Database:")
        print(f"   Host: {settings.chromadb_host}:{settings.chromadb_port}")
        print(f"   Collection: {settings.chromadb_collection}")
        print(f"   Embedding Model: {settings.embedding_model}")

        # CORS
        print(f"\n🌐 CORS:")
        print(f"   Origins: {settings.cors_origins}")

        print("\n" + "=" * 50)
        print("✅ Configuration loaded successfully!")
        print("\nNext steps:")
        print("  1. Start Docker services: docker-compose up -d")
        print("  2. Run migrations: poetry run alembic upgrade head")
        print("  3. Start API: poetry run python -m src.main")

        return True

    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nRun: poetry install")
        return False
    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 AI-CoScientist Configuration Verification")
    print("=" * 50)

    if not check_env_file():
        print("\nPlease run: ./scripts/setup.sh")
        sys.exit(1)

    if not load_config():
        sys.exit(1)

    print("\n✅ All checks passed!")

if __name__ == "__main__":
    main()
