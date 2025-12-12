"""Add RAG evaluation tables for Phase 2

Revision ID: def456789012
Revises: abc123456789
Create Date: 2025-10-22 00:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'def456789012'
down_revision: Union[str, Sequence[str], None] = 'abc123456789'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema for Phase 2: RAG Evaluation System."""

    # 1. Create rag_evaluations table
    op.create_table(
        'rag_evaluations',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('dataset_id', sa.String(length=255), nullable=False),
        sa.Column('evaluation_type', sa.String(length=50), nullable=False),
        sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('evaluation_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_rag_evaluations_dataset_id', 'rag_evaluations', ['dataset_id'])
    op.create_index('ix_rag_evaluations_evaluation_type', 'rag_evaluations', ['evaluation_type'])

    # 2. Create rag_performance_metrics table
    op.create_table(
        'rag_performance_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('operation', sa.String(length=100), nullable=False),
        sa.Column('latency', sa.Float(), nullable=False),
        sa.Column('token_usage', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('cost', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_rag_performance_metrics_operation', 'rag_performance_metrics', ['operation'])

    # 3. Create rag_cost_budgets table
    op.create_table(
        'rag_cost_budgets',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('total_budget', sa.Float(), nullable=False),
        sa.Column('spent', sa.Float(), nullable=False, server_default='0'),
        sa.Column('warning_threshold', sa.Float(), nullable=False, server_default='0.8'),
        sa.Column('critical_threshold', sa.Float(), nullable=False, server_default='0.95'),
        sa.Column('expenses', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # 4. Create rag_ab_tests table
    op.create_table(
        'rag_ab_tests',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='active'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_rag_ab_tests_status', 'rag_ab_tests', ['status'])

    # 5. Create rag_ab_test_results table
    op.create_table(
        'rag_ab_test_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('test_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('variant_name', sa.String(length=100), nullable=False),
        sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('cost', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['test_id'], ['rag_ab_tests.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_rag_ab_test_results_test_id', 'rag_ab_test_results', ['test_id'])
    op.create_index('ix_rag_ab_test_results_variant_name', 'rag_ab_test_results', ['variant_name'])


def downgrade() -> None:
    """Downgrade schema - remove RAG evaluation tables."""

    # Drop tables in reverse order (respecting foreign keys)
    op.drop_index('ix_rag_ab_test_results_variant_name', table_name='rag_ab_test_results')
    op.drop_index('ix_rag_ab_test_results_test_id', table_name='rag_ab_test_results')
    op.drop_table('rag_ab_test_results')

    op.drop_index('ix_rag_ab_tests_status', table_name='rag_ab_tests')
    op.drop_table('rag_ab_tests')

    op.drop_table('rag_cost_budgets')

    op.drop_index('ix_rag_performance_metrics_operation', table_name='rag_performance_metrics')
    op.drop_table('rag_performance_metrics')

    op.drop_index('ix_rag_evaluations_evaluation_type', table_name='rag_evaluations')
    op.drop_index('ix_rag_evaluations_dataset_id', table_name='rag_evaluations')
    op.drop_table('rag_evaluations')
