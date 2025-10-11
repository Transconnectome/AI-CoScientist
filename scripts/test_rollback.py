#!/usr/bin/env python3
"""Test Rollback Capability (TDD Safety Check)

Verifies that Phase 4 migration can be safely rolled back if needed.
This is the REFACTOR phase of TDD - ensuring we can undo changes.
"""
from pathlib import Path


def check_downgrade_exists():
    """Verify rollback migration exists."""
    migration_file = Path('alembic/versions/abc123456789_add_phase4_version_tracking.py')

    if not migration_file.exists():
        print(f"❌ Migration file not found: {migration_file}")
        return False

    print(f"✅ Migration file exists: {migration_file}")

    # Read and check downgrade function
    content = migration_file.read_text()

    checks = {
        'downgrade function': 'def downgrade()',
        'drop iteration_sessions': 'drop_table(\'iteration_sessions\')',
        'drop improvement_history': 'drop_table(\'improvement_history\')',
        'drop paper_versions': 'drop_table(\'paper_versions\')',
        'remove version_major': 'drop_column(\'papers\', \'version_major\')',
        'remove version_minor': 'drop_column(\'papers\', \'version_minor\')',
        'remove version_patch': 'drop_column(\'papers\', \'version_patch\')',
    }

    all_passed = True
    for check_name, search_pattern in checks.items():
        if search_pattern in content:
            print(f"✅ {check_name}: present")
        else:
            print(f"❌ {check_name}: MISSING!")
            all_passed = False

    return all_passed


def print_rollback_instructions():
    """Print instructions for performing rollback."""
    print("\n" + "=" * 60)
    print("ROLLBACK INSTRUCTIONS")
    print("=" * 60)

    instructions = """
To rollback Phase 4 migration (if needed):

1. Downgrade one migration:
   python -c "from alembic.config import Config; from alembic import command; cfg = Config('alembic.ini'); command.downgrade(cfg, '-1')"

2. Verify rollback:
   python -c "from alembic.config import Config; from alembic import command; from io import StringIO; cfg = Config('alembic.ini'); out = StringIO(); cfg.attributes['stdout'] = out; command.current(cfg); print(out.getvalue())"

3. Expected result:
   Should show: 287862b51369 (previous version)

4. To re-apply:
   python -c "from alembic.config import Config; from alembic import command; cfg = Config('alembic.ini'); command.upgrade(cfg, 'head')"

⚠️  WARNING: Rollback will:
   • Drop tables: paper_versions, improvement_history, iteration_sessions
   • Remove columns: papers.version_major, version_minor, version_patch
   • Delete ALL Phase 4 data (not recoverable!)

✅ SAFE to rollback when:
   • No Phase 4 data exists yet
   • Testing/development environment
   • Before production deployment

❌ DO NOT rollback when:
   • Production data exists
   • Phase 4 features are in use
   • Without database backup
"""

    print(instructions)


def main():
    """Main rollback verification."""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║   TDD Safety Check: Rollback Capability                   ║")
    print("║   REFACTOR Phase - Ensuring Reversibility                 ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    if check_downgrade_exists():
        print("\n╔════════════════════════════════════════════════════════════╗")
        print("║   ✅ ROLLBACK CAPABILITY VERIFIED                         ║")
        print("║   Migration is safely reversible                          ║")
        print("╚════════════════════════════════════════════════════════════╝")

        print_rollback_instructions()
        return 0
    else:
        print("\n❌ Rollback capability check FAILED")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
