#!/usr/bin/env python3
"""Simple Migration Verification Script

Verifies Phase 4 database migration completed successfully.
Uses simple SQL queries without async complications.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psycopg2
from tabulate import tabulate


# Database connection (adjust if needed)
DB_CONFIG = {
    'dbname': 'ai_coscientist',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': 5432
}


def print_header(text):
    """Print formatted header."""
    print(f"\n{'=' * 60}")
    print(f"{text}")
    print('=' * 60)


def check_table_exists(cursor, table_name):
    """Check if a table exists."""
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = %s
        );
    """, (table_name,))
    return cursor.fetchone()[0]


def get_table_columns(cursor, table_name):
    """Get columns for a table."""
    cursor.execute("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position;
    """, (table_name,))
    return cursor.fetchall()


def get_table_indexes(cursor, table_name):
    """Get indexes for a table."""
    cursor.execute("""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = %s
        ORDER BY indexname;
    """, (table_name,))
    return cursor.fetchall()


def get_foreign_keys(cursor, table_name):
    """Get foreign keys for a table."""
    cursor.execute("""
        SELECT
            tc.constraint_name,
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_name = %s;
    """, (table_name,))
    return cursor.fetchall()


def main():
    """Main verification logic."""
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║   Phase 4 Migration Verification                          ║")
    print("║   Database Schema Check                                   ║")
    print("╚════════════════════════════════════════════════════════════╝")

    try:
        # Connect to database
        print("\n📊 Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✅ Connected successfully!")

        # Track verification results
        total_checks = 0
        passed_checks = 0

        # 1. Check papers table has version columns
        print_header("1. Checking papers table version columns")
        total_checks += 1

        columns = [col[0] for col in get_table_columns(cursor, 'papers')]
        version_columns = ['version_major', 'version_minor', 'version_patch']

        missing = [col for col in version_columns if col not in columns]

        if not missing:
            print(f"✅ All version columns present: {', '.join(version_columns)}")
            passed_checks += 1
        else:
            print(f"❌ Missing columns: {', '.join(missing)}")

        # 2. Check new tables
        print_header("2. Checking Phase 4 tables")

        new_tables = ['paper_versions', 'improvement_history', 'iteration_sessions']

        for table in new_tables:
            total_checks += 1
            exists = check_table_exists(cursor, table)

            if exists:
                print(f"✅ Table '{table}' exists")
                passed_checks += 1

                # Show column count
                columns = get_table_columns(cursor, table)
                print(f"   └─ {len(columns)} columns")
            else:
                print(f"❌ Table '{table}' missing!")

        # 3. Check indexes
        print_header("3. Checking performance indexes")

        expected_indexes = {
            'paper_versions': ['idx_paper_versions_paper_id', 'idx_paper_versions_version'],
            'improvement_history': [
                'idx_improvement_history_paper_id',
                'idx_improvement_history_status',
                'idx_improvement_history_type'
            ],
            'iteration_sessions': [
                'idx_iteration_sessions_paper_id',
                'idx_iteration_sessions_is_complete'
            ]
        }

        for table, expected_idx in expected_indexes.items():
            indexes = [idx[0] for idx in get_table_indexes(cursor, table)]

            for idx_name in expected_idx:
                total_checks += 1
                if idx_name in indexes:
                    print(f"✅ Index '{idx_name}' exists")
                    passed_checks += 1
                else:
                    print(f"❌ Index '{idx_name}' missing!")

        # 4. Check foreign keys
        print_header("4. Checking foreign key relationships")

        fk_checks = {
            'paper_versions': ['papers'],
            'improvement_history': ['papers', 'paper_versions'],
            'iteration_sessions': ['papers', 'paper_versions']
        }

        for table, expected_refs in fk_checks.items():
            fks = get_foreign_keys(cursor, table)
            ref_tables = [fk[2] for fk in fks]

            for ref_table in expected_refs:
                total_checks += 1
                if ref_table in ref_tables:
                    print(f"✅ {table} → {ref_table} FK exists")
                    passed_checks += 1
                else:
                    print(f"❌ {table} → {ref_table} FK missing!")

        # 5. Check data migration
        print_header("5. Checking existing data migration")

        cursor.execute("""
            SELECT COUNT(*) FROM papers
            WHERE version_major IS NULL OR version_minor IS NULL OR version_patch IS NULL
        """)
        null_count = cursor.fetchone()[0]

        total_checks += 1
        if null_count == 0:
            print(f"✅ All existing papers have version fields initialized")
            passed_checks += 1
        else:
            print(f"⚠️  {null_count} papers have NULL version fields")

        # Show a few sample papers
        cursor.execute("""
            SELECT id, version_major, version_minor, version_patch
            FROM papers
            LIMIT 3
        """)
        samples = cursor.fetchall()

        if samples:
            print(f"\n   Sample papers:")
            for paper in samples:
                print(f"   • {paper[0]}: v{paper[1]}.{paper[2]}.{paper[3]}")
        else:
            print("   (No papers in database yet)")

        # Summary
        print_header("VERIFICATION SUMMARY")

        success_rate = (passed_checks / total_checks) * 100
        print(f"Total checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Success rate: {success_rate:.1f}%")

        if passed_checks == total_checks:
            print("\n╔════════════════════════════════════════════════════════════╗")
            print("║   ✅ ALL CHECKS PASSED!                                    ║")
            print("║   Phase 4 migration completed successfully!               ║")
            print("╚════════════════════════════════════════════════════════════╝")
            return 0
        else:
            print("\n⚠️  Some checks failed. Please review the output above.")
            return 1

    except psycopg2.Error as e:
        print(f"\n❌ Database error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


if __name__ == '__main__':
    sys.exit(main())
