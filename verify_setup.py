#!/usr/bin/env python3
"""Verify AI-CoScientist system configuration."""

import os
import sys
from pathlib import Path

def check_env_vars():
    """Check if required environment variables are set."""
    print("="*80)
    print("ENVIRONMENT VARIABLES CHECK")
    print("="*80)

    required_vars = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
        'GOOGLE_API_KEY',
        'DATABASE_URL',
    ]

    optional_vars = [
        'REDIS_URL',
        'CHROMADB_HOST',
        'LLM_PRIMARY_PROVIDER',
        'LLM_FALLBACK_PROVIDER',
    ]

    missing_required = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            masked = value[:10] + "..." if len(value) > 10 else "***"
            print(f"  ✅ {var}: {masked}")
        else:
            print(f"  ❌ {var}: NOT SET")
            missing_required.append(var)

    print(f"\nOptional variables:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {value}")
        else:
            print(f"  ⚠️  {var}: not set")

    if missing_required:
        print(f"\n❌ Missing required variables: {', '.join(missing_required)}")
        return False
    else:
        print(f"\n✅ All required environment variables are set")
        return True


def check_database():
    """Check database connection and tables."""
    print("\n" + "="*80)
    print("DATABASE CHECK")
    print("="*80)

    try:
        import psycopg2
        from dotenv import load_dotenv

        load_dotenv()
        db_url = os.getenv('DATABASE_URL')

        # Convert async URL to sync for testing
        if 'asyncpg' in db_url:
            db_url = db_url.replace('postgresql+asyncpg://', 'postgresql://')

        # Parse connection string
        from urllib.parse import urlparse
        parsed = urlparse(db_url)

        conn = psycopg2.connect(
            host=parsed.hostname or 'localhost',
            port=parsed.port or 5432,
            database=parsed.path[1:] if parsed.path else 'postgres',
            user=parsed.username or os.getenv('USER'),
            password=parsed.password or ''
        )

        cursor = conn.cursor()

        # Check tables
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)

        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = ['projects', 'hypotheses', 'experiments', 'papers', 'paper_sections']
        missing_tables = [t for t in expected_tables if t not in tables]

        print(f"  Database: {parsed.path[1:]}")
        print(f"  Connection: ✅ Connected")
        print(f"\n  Tables found ({len(tables)}):")
        for table in tables:
            status = "✅" if table in expected_tables or table == 'alembic_version' else "⚠️"
            print(f"    {status} {table}")

        if missing_tables:
            print(f"\n  ❌ Missing expected tables: {', '.join(missing_tables)}")
            cursor.close()
            conn.close()
            return False
        else:
            print(f"\n  ✅ All expected tables exist")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
        return False


def check_api_keys():
    """Test API key validity (basic format check)."""
    print("\n" + "="*80)
    print("API KEYS VALIDATION")
    print("="*80)

    from dotenv import load_dotenv
    load_dotenv()

    api_keys = {
        'OpenAI': os.getenv('OPENAI_API_KEY'),
        'Anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'Google': os.getenv('GOOGLE_API_KEY'),
    }

    all_valid = True
    for provider, key in api_keys.items():
        if not key:
            print(f"  ❌ {provider}: No API key set")
            all_valid = False
        elif len(key) < 20:
            print(f"  ⚠️  {provider}: API key seems too short (possible configuration error)")
            all_valid = False
        else:
            print(f"  ✅ {provider}: API key configured (length: {len(key)} chars)")

    return all_valid


def check_dependencies():
    """Check if required Python packages are installed."""
    print("\n" + "="*80)
    print("DEPENDENCIES CHECK")
    print("="*80)

    required_packages = [
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'alembic',
        'asyncpg',
        'openai',
        'anthropic',
        'python-dotenv',
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}: NOT INSTALLED")
            missing.append(package)

    if missing:
        print(f"\n  ❌ Missing packages: {', '.join(missing)}")
        print(f"  Install with: poetry install")
        return False
    else:
        print(f"\n  ✅ All required packages are installed")
        return True


def check_project_structure():
    """Check if required directories and files exist."""
    print("\n" + "="*80)
    print("PROJECT STRUCTURE CHECK")
    print("="*80)

    required_paths = [
        'src/models/project.py',
        'src/services/paper/__init__.py',
        'src/services/paper/parser.py',
        'src/services/paper/analyzer.py',
        'src/services/paper/improver.py',
        'src/services/paper/generator.py',
        'src/api/v1/papers.py',
        'src/schemas/paper.py',
        'alembic/env.py',
        'alembic.ini',
        '.env',
    ]

    missing = []
    for path in required_paths:
        full_path = Path(path)
        if full_path.exists():
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path}: NOT FOUND")
            missing.append(path)

    if missing:
        print(f"\n  ❌ Missing files: {', '.join(missing)}")
        return False
    else:
        print(f"\n  ✅ All required files exist")
        return True


def main():
    """Run all verification checks."""
    print("\n" + "="*80)
    print("AI-CoScientist System Configuration Verification")
    print("="*80)

    from dotenv import load_dotenv
    load_dotenv()

    checks = [
        ("Project Structure", check_project_structure),
        ("Environment Variables", check_env_vars),
        ("Dependencies", check_dependencies),
        ("API Keys", check_api_keys),
        ("Database", check_database),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error during {name} check: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "="*80)
        print("✅ SYSTEM READY")
        print("="*80)
        print("""
All checks passed! Your AI-CoScientist system is configured and ready to use.

Next steps:
1. Run the paper analysis demo:
   python3 demo_paper_analysis.py

2. Or start the API server:
   poetry run uvicorn src.main:app --reload

3. Access API documentation:
   http://localhost:8000/docs
""")
        return 0
    else:
        print("\n" + "="*80)
        print("❌ CONFIGURATION INCOMPLETE")
        print("="*80)
        print("\nPlease fix the failed checks above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
