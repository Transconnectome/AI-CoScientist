"""Add metrics tracking tables

Revision ID: 1760450436
Revises: a32a81c0d290
Create Date: 2025-10-14 22:57:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision = '1760450436'
down_revision = 'a32a81c0d290'
branch_labels = None
depends_on = None


def upgrade():
    # Agent execution metrics
    op.create_table(
        'agent_executions',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('agent_id', sa.String(100), nullable=False, index=True),
        sa.Column('task_type', sa.String(50), nullable=False),
        sa.Column('task_id', sa.String(100), nullable=False),
        sa.Column('success', sa.Boolean, nullable=False),
        sa.Column('confidence', sa.Float),
        sa.Column('execution_time_ms', sa.Float),
        sa.Column('tokens_used', sa.Integer),
        sa.Column('quality_score', sa.Float),
        sa.Column('extra_metadata', JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Workflow performance
    op.create_table(
        'workflow_metrics',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('task_type', sa.String(50), nullable=False),
        sa.Column('agents_used', JSONB, nullable=False),
        sa.Column('quality_score', sa.Float, nullable=False),
        sa.Column('execution_time_ms', sa.Float, nullable=False),
        sa.Column('success', sa.Boolean, nullable=False),
        sa.Column('extra_metadata', JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )

    # Indexes for performance
    op.create_index(
        'idx_agent_exec_agent_task',
        'agent_executions',
        ['agent_id', 'task_type']
    )
    op.create_index(
        'idx_workflow_task_quality',
        'workflow_metrics',
        ['task_type', 'quality_score']
    )


def downgrade():
    op.drop_table('workflow_metrics')
    op.drop_table('agent_executions')
