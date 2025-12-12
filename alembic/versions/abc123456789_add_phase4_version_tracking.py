"""Add Phase 4 version tracking and improvement history

Revision ID: abc123456789
Revises: 287862b51369
Create Date: 2025-10-10 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'abc123456789'
down_revision: Union[str, Sequence[str], None] = '287862b51369'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema for Phase 4: Version Tracking & Learning."""

    # 1. Add semantic versioning columns to papers table
    op.add_column('papers', sa.Column('version_major', sa.Integer(), nullable=False, server_default='1'))
    op.add_column('papers', sa.Column('version_minor', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('papers', sa.Column('version_patch', sa.Integer(), nullable=False, server_default='0'))

    # Migrate existing papers: copy version to version_major
    op.execute("UPDATE papers SET version_major = version WHERE version_major = 1")

    # 2. Create paper_versions table
    op.create_table(
        'paper_versions',
        sa.Column('paper_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('parent_version_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('major', sa.Integer(), nullable=False),
        sa.Column('minor', sa.Integer(), nullable=False),
        sa.Column('patch', sa.Integer(), nullable=False),
        sa.Column('content_snapshot', sa.Text(), nullable=False),
        sa.Column('sections_snapshot', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('version_type', sa.String(length=20), nullable=False),
        sa.Column('change_summary', sa.Text(), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['parent_version_id'], ['paper_versions.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for paper_versions
    op.create_index('idx_paper_versions_paper_id', 'paper_versions', ['paper_id'], unique=False)
    op.create_index('idx_paper_versions_version', 'paper_versions', ['major', 'minor', 'patch'], unique=False)

    # 3. Create improvement_history table
    op.create_table(
        'improvement_history',
        sa.Column('paper_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('version_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('section_name', sa.String(length=200), nullable=False),
        sa.Column('original_content', sa.Text(), nullable=False),
        sa.Column('improved_content', sa.Text(), nullable=False),
        sa.Column('improvement_type', sa.String(length=100), nullable=False),
        sa.Column('changes_made', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('quality_before', sa.Float(), nullable=False),
        sa.Column('quality_after', sa.Float(), nullable=True),
        sa.Column('improvement_score', sa.Float(), nullable=False),
        sa.Column('user_feedback', sa.Text(), nullable=True),
        sa.Column('success_rating', sa.Integer(), nullable=True),
        sa.Column('pattern_embedding_id', sa.String(length=100), nullable=True),
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['version_id'], ['paper_versions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for improvement_history
    op.create_index('idx_improvement_history_paper_id', 'improvement_history', ['paper_id'], unique=False)
    op.create_index('idx_improvement_history_status', 'improvement_history', ['status'], unique=False)
    op.create_index('idx_improvement_history_type', 'improvement_history', ['improvement_type'], unique=False)

    # 4. Create iteration_sessions table
    op.create_table(
        'iteration_sessions',
        sa.Column('paper_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('start_version_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('current_version_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('target_score', sa.Float(), nullable=False),
        sa.Column('max_iterations', sa.Integer(), nullable=False),
        sa.Column('focus_areas', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('current_iteration', sa.Integer(), nullable=False),
        sa.Column('current_score', sa.Float(), nullable=False),
        sa.Column('is_complete', sa.Boolean(), nullable=False),
        sa.Column('improvements_applied', sa.Integer(), nullable=False),
        sa.Column('score_improvement', sa.Float(), nullable=False),
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['paper_id'], ['papers.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['start_version_id'], ['paper_versions.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['current_version_id'], ['paper_versions.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for iteration_sessions
    op.create_index('idx_iteration_sessions_paper_id', 'iteration_sessions', ['paper_id'], unique=False)
    op.create_index('idx_iteration_sessions_is_complete', 'iteration_sessions', ['is_complete'], unique=False)


def downgrade() -> None:
    """Downgrade schema - remove Phase 4 tables and columns."""

    # Drop iteration_sessions
    op.drop_index('idx_iteration_sessions_is_complete', table_name='iteration_sessions')
    op.drop_index('idx_iteration_sessions_paper_id', table_name='iteration_sessions')
    op.drop_table('iteration_sessions')

    # Drop improvement_history
    op.drop_index('idx_improvement_history_type', table_name='improvement_history')
    op.drop_index('idx_improvement_history_status', table_name='improvement_history')
    op.drop_index('idx_improvement_history_paper_id', table_name='improvement_history')
    op.drop_table('improvement_history')

    # Drop paper_versions
    op.drop_index('idx_paper_versions_version', table_name='paper_versions')
    op.drop_index('idx_paper_versions_paper_id', table_name='paper_versions')
    op.drop_table('paper_versions')

    # Remove semantic versioning columns from papers
    op.drop_column('papers', 'version_patch')
    op.drop_column('papers', 'version_minor')
    op.drop_column('papers', 'version_major')
