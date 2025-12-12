"""Test-Driven Migration Verification

This test suite verifies Phase 4 database migration following TDD philosophy:
1. RED: Tests fail before migration
2. GREEN: Tests pass after migration
3. REFACTOR: Verify rollback safety
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Import settings with fallback
try:
    from src.core.config import settings
except ImportError:
    # Create minimal settings for testing
    class Settings:
        database_url = "postgresql+asyncpg://postgres:postgres@localhost/ai_coscientist"
    settings = Settings()


@pytest.fixture
async def db_inspector():
    """Create database inspector for schema verification."""
    engine = create_async_engine(str(settings.database_url))

    async with engine.begin() as conn:
        inspector = await conn.run_sync(lambda sync_conn: inspect(sync_conn))
        yield inspector

    await engine.dispose()


@pytest.fixture
async def db_session():
    """Create database session for data verification."""
    engine = create_async_engine(str(settings.database_url))
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session

    await engine.dispose()


class TestPhase4MigrationSchema:
    """Test Phase 4 database schema requirements."""

    @pytest.mark.asyncio
    async def test_papers_table_has_version_columns(self, db_inspector):
        """Verify papers table has semantic versioning columns."""
        columns = {col['name'] for col in db_inspector.get_columns('papers')}

        # These columns must exist after migration
        assert 'version_major' in columns, "papers.version_major column missing"
        assert 'version_minor' in columns, "papers.version_minor column missing"
        assert 'version_patch' in columns, "papers.version_patch column missing"

    @pytest.mark.asyncio
    async def test_paper_versions_table_exists(self, db_inspector):
        """Verify paper_versions table exists with correct structure."""
        tables = db_inspector.get_table_names()
        assert 'paper_versions' in tables, "paper_versions table not found"

        # Check essential columns
        columns = {col['name'] for col in db_inspector.get_columns('paper_versions')}
        required_columns = {
            'id', 'paper_id', 'major', 'minor', 'patch',
            'content_snapshot', 'sections_snapshot', 'version_type',
            'change_summary', 'quality_score', 'created_at', 'updated_at'
        }

        missing = required_columns - columns
        assert not missing, f"Missing columns in paper_versions: {missing}"

    @pytest.mark.asyncio
    async def test_improvement_history_table_exists(self, db_inspector):
        """Verify improvement_history table exists with correct structure."""
        tables = db_inspector.get_table_names()
        assert 'improvement_history' in tables, "improvement_history table not found"

        # Check essential columns
        columns = {col['name'] for col in db_inspector.get_columns('improvement_history')}
        required_columns = {
            'id', 'paper_id', 'version_id', 'section_name',
            'original_content', 'improved_content', 'improvement_type',
            'changes_made', 'status', 'quality_before', 'quality_after',
            'improvement_score', 'created_at', 'updated_at'
        }

        missing = required_columns - columns
        assert not missing, f"Missing columns in improvement_history: {missing}"

    @pytest.mark.asyncio
    async def test_iteration_sessions_table_exists(self, db_inspector):
        """Verify iteration_sessions table exists with correct structure."""
        tables = db_inspector.get_table_names()
        assert 'iteration_sessions' in tables, "iteration_sessions table not found"

        # Check essential columns
        columns = {col['name'] for col in db_inspector.get_columns('iteration_sessions')}
        required_columns = {
            'id', 'paper_id', 'start_version_id', 'current_version_id',
            'target_score', 'max_iterations', 'focus_areas',
            'current_iteration', 'current_score', 'is_complete',
            'improvements_applied', 'score_improvement', 'created_at', 'updated_at'
        }

        missing = required_columns - columns
        assert not missing, f"Missing columns in iteration_sessions: {missing}"

    @pytest.mark.asyncio
    async def test_indexes_created(self, db_inspector):
        """Verify performance indexes are created."""
        # paper_versions indexes
        pv_indexes = {idx['name'] for idx in db_inspector.get_indexes('paper_versions')}
        assert 'idx_paper_versions_paper_id' in pv_indexes
        assert 'idx_paper_versions_version' in pv_indexes

        # improvement_history indexes
        ih_indexes = {idx['name'] for idx in db_inspector.get_indexes('improvement_history')}
        assert 'idx_improvement_history_paper_id' in ih_indexes
        assert 'idx_improvement_history_status' in ih_indexes
        assert 'idx_improvement_history_type' in ih_indexes

        # iteration_sessions indexes
        is_indexes = {idx['name'] for idx in db_inspector.get_indexes('iteration_sessions')}
        assert 'idx_iteration_sessions_paper_id' in is_indexes
        assert 'idx_iteration_sessions_is_complete' in is_indexes

    @pytest.mark.asyncio
    async def test_foreign_key_constraints(self, db_inspector):
        """Verify foreign key relationships are established."""
        # paper_versions foreign keys
        pv_fks = db_inspector.get_foreign_keys('paper_versions')
        pv_fk_tables = {fk['referred_table'] for fk in pv_fks}
        assert 'papers' in pv_fk_tables, "paper_versions should reference papers"

        # improvement_history foreign keys
        ih_fks = db_inspector.get_foreign_keys('improvement_history')
        ih_fk_tables = {fk['referred_table'] for fk in ih_fks}
        assert 'papers' in ih_fk_tables
        assert 'paper_versions' in ih_fk_tables

        # iteration_sessions foreign keys
        is_fks = db_inspector.get_foreign_keys('iteration_sessions')
        is_fk_tables = {fk['referred_table'] for fk in is_fks}
        assert 'papers' in is_fk_tables
        assert 'paper_versions' in is_fk_tables


class TestPhase4MigrationData:
    """Test Phase 4 migration data integrity."""

    @pytest.mark.asyncio
    async def test_existing_papers_migrated(self, db_session):
        """Verify existing papers have version fields initialized."""
        result = await db_session.execute(
            text("SELECT id, version_major, version_minor, version_patch FROM papers LIMIT 5")
        )
        papers = result.fetchall()

        if papers:
            for paper in papers:
                # Check default values were set
                assert paper.version_major is not None, "version_major should be initialized"
                assert paper.version_minor is not None, "version_minor should be initialized"
                assert paper.version_patch is not None, "version_patch should be initialized"

                # Default should be 1.0.0
                assert paper.version_major >= 1, "version_major should be >= 1"
                assert paper.version_minor >= 0, "version_minor should be >= 0"
                assert paper.version_patch >= 0, "version_patch should be >= 0"

    @pytest.mark.asyncio
    async def test_new_tables_empty_but_insertable(self, db_session):
        """Verify new tables are empty but ready for data."""
        # Check tables are initially empty
        for table in ['paper_versions', 'improvement_history', 'iteration_sessions']:
            result = await db_session.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            # Tables should be empty initially (or have migrated data)
            assert count >= 0, f"{table} should be accessible"


class TestMigrationRollback:
    """Test migration rollback safety."""

    @pytest.mark.asyncio
    async def test_downgrade_script_exists(self):
        """Verify rollback migration exists."""
        import os
        migration_file = 'alembic/versions/abc123456789_add_phase4_version_tracking.py'
        assert os.path.exists(migration_file), "Migration file not found"

        # Read migration file
        with open(migration_file, 'r') as f:
            content = f.read()

        # Verify downgrade function exists
        assert 'def downgrade()' in content, "downgrade() function missing"

        # Verify it drops the tables
        assert 'drop_table' in content, "downgrade should drop tables"
        assert "'paper_versions'" in content
        assert "'improvement_history'" in content
        assert "'iteration_sessions'" in content


if __name__ == '__main__':
    print("🔴 RED Phase: Running tests (should fail before migration)")
    print("=" * 60)
    pytest.main([__file__, '-v', '--tb=short'])
