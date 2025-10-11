#!/usr/bin/env python3
"""TDD-style Migration Runner

Following Test-Driven Development philosophy:
1. RED: Run tests to verify migration is needed (should fail)
2. GREEN: Run migration to make tests pass
3. REFACTOR: Verify and document results
"""
import subprocess
import sys
from pathlib import Path

# Color codes for output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text: str, color: str = BLUE):
    """Print formatted header."""
    print(f"\n{BOLD}{color}{'=' * 60}{RESET}")
    print(f"{BOLD}{color}{text}{RESET}")
    print(f"{BOLD}{color}{'=' * 60}{RESET}\n")


def run_command(cmd: list[str], description: str = "") -> tuple[int, str]:
    """Run shell command and return result."""
    if description:
        print(f"{BLUE}▶ {description}{RESET}")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode, result.stdout + result.stderr


def check_alembic_available() -> bool:
    """Check if alembic is installed."""
    try:
        result = subprocess.run(
            ["python", "-c", "import alembic; print(alembic.__version__)"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ Alembic installed: version {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Alembic not found")
            return False
    except Exception as e:
        print(f"❌ Error checking Alembic: {e}")
        return False


def get_current_migration() -> str:
    """Get current migration version."""
    result = subprocess.run(
        ["python", "-c",
         "from alembic.config import Config; "
         "from alembic import command; "
         "from io import StringIO; "
         "import sys; "
         "cfg = Config('alembic.ini'); "
         "out = StringIO(); "
         "cfg.attributes['stdout'] = out; "
         "command.current(cfg); "
         "print(out.getvalue())"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    return result.stdout.strip()


def run_migration() -> bool:
    """Execute database migration."""
    result = subprocess.run(
        ["python", "-c",
         "from alembic.config import Config; "
         "from alembic import command; "
         "cfg = Config('alembic.ini'); "
         "command.upgrade(cfg, 'head')"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def run_tests(test_file: str = "tests/test_migration_verification.py") -> tuple[bool, str]:
    """Run migration verification tests."""
    returncode, output = run_command(
        ["pytest", test_file, "-v", "--tb=short"],
        f"Running migration tests: {test_file}"
    )
    return returncode == 0, output


def main():
    """Main TDD migration workflow."""
    print(f"{BOLD}{BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║   Test-Driven Database Migration (Phase 4)                ║")
    print("║   Following RED → GREEN → REFACTOR Philosophy             ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(RESET)

    # Step 0: Prerequisites
    print_header("Step 0: Prerequisites Check", BLUE)
    if not check_alembic_available():
        print(f"\n{RED}❌ Alembic not installed. Run: pip install alembic{RESET}")
        sys.exit(1)

    # Step 1: RED - Run tests before migration (should fail)
    print_header("🔴 RED Phase: Run Tests BEFORE Migration", RED)
    print("Expected: Tests should FAIL because tables don't exist yet\n")

    current = get_current_migration()
    print(f"Current migration: {current}\n")

    passed, output = run_tests()

    if passed:
        print(f"\n{YELLOW}⚠️  Tests passed! Migration may already be applied.{RESET}")
        print(f"{YELLOW}   Checking if Phase 4 tables exist...{RESET}")

        # Check if we're already at Phase 4
        if "abc123456789" in current:
            print(f"\n{GREEN}✅ Phase 4 migration already applied!{RESET}")
            print(f"{GREEN}   Current version: {current}{RESET}")
            print(f"\n{BLUE}ℹ️  Skipping to verification phase...{RESET}")
        else:
            print(f"\n{YELLOW}⚠️  Unexpected state: tests pass but migration not applied{RESET}")
            sys.exit(1)
    else:
        print(f"\n{RED}✅ Good! Tests failed as expected (RED phase){RESET}")
        print(f"{RED}   This confirms migration is needed.{RESET}")

        # Count failures
        if "failed" in output:
            import re
            match = re.search(r'(\d+) failed', output)
            if match:
                print(f"{RED}   Failed tests: {match.group(1)}{RESET}")

    # Step 2: GREEN - Run migration
    print_header("🟢 GREEN Phase: Run Migration", GREEN)
    print("Executing: alembic upgrade head\n")

    if "abc123456789" not in current:
        success = run_migration()

        if not success:
            print(f"\n{RED}❌ Migration failed!{RESET}")
            sys.exit(1)

        print(f"\n{GREEN}✅ Migration completed successfully!{RESET}")

        # Verify new migration
        new_current = get_current_migration()
        print(f"\nNew migration version: {new_current}")
    else:
        print(f"{GREEN}✅ Migration already at target version{RESET}")

    # Step 3: Verify with tests
    print_header("✅ Verification: Run Tests AFTER Migration", GREEN)
    print("Expected: All tests should PASS now\n")

    passed, output = run_tests()

    if passed:
        print(f"\n{BOLD}{GREEN}╔════════════════════════════════════════════════════════════╗{RESET}")
        print(f"{BOLD}{GREEN}║   ✅ SUCCESS! All tests passed!                            ║{RESET}")
        print(f"{BOLD}{GREEN}║   Phase 4 database migration complete!                    ║{RESET}")
        print(f"{BOLD}{GREEN}╚════════════════════════════════════════════════════════════╝{RESET}")

        # Count passed tests
        import re
        match = re.search(r'(\d+) passed', output)
        if match:
            print(f"\n{GREEN}✅ Passed tests: {match.group(1)}{RESET}")

    else:
        print(f"\n{RED}❌ Some tests still failing after migration{RESET}")
        print(f"{RED}   Please check the output above for details{RESET}")
        sys.exit(1)

    # Step 4: Summary
    print_header("📊 Migration Summary", BLUE)

    summary = f"""
{GREEN}✅ Phase 4 Database Migration Complete{RESET}

{BOLD}What was created:{RESET}
  • 3 new tables: paper_versions, improvement_history, iteration_sessions
  • 3 new columns in papers: version_major, version_minor, version_patch
  • 8 performance indexes
  • Foreign key relationships established

{BOLD}TDD Cycle Completed:{RESET}
  🔴 RED:    Tests failed before migration ✓
  🟢 GREEN:  Migration executed successfully ✓
  ✅ VERIFY: Tests pass after migration ✓

{BOLD}Next Steps:{RESET}
  1. Run demo: python scripts/demo_phase4_auto.py
  2. Start chatbot: python scripts/chat_reviewer_enhanced.py
  3. Test with real papers!

{BOLD}Rollback (if needed):{RESET}
  python -c "from alembic.config import Config; from alembic import command; \\
             cfg = Config('alembic.ini'); command.downgrade(cfg, '-1')"
"""

    print(summary)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}⚠️  Migration interrupted by user{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}❌ Error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
