"""add_literature_monitoring_tables

Revision ID: a32a81c0d290
Revises: abc123456789
Create Date: 2025-10-11 23:55:42.963097

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a32a81c0d290'
down_revision: Union[str, Sequence[str], None] = 'abc123456789'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create literature_sources table
    op.create_table(
        'literature_sources',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('source_type', sa.String(length=50), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('query', sa.Text(), nullable=True),
        sa.Column('last_sync_time', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='active'),
        sa.Column('sync_frequency', sa.String(length=20), nullable=False, server_default='daily'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for literature_sources
    op.create_index('idx_literature_sources_type', 'literature_sources', ['source_type'])
    op.create_index('idx_literature_sources_status', 'literature_sources', ['status'])
    op.create_index('idx_literature_sources_sync_time', 'literature_sources', ['last_sync_time'])

    # Create monitoring_alerts table
    op.create_table(
        'monitoring_alerts',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=True),
        sa.Column('topic', sa.String(length=200), nullable=False),
        sa.Column('keywords', sa.ARRAY(sa.Text()), nullable=True),
        sa.Column('frequency', sa.String(length=20), nullable=False, server_default='daily'),
        sa.Column('last_alert_sent', sa.DateTime(), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for monitoring_alerts
    op.create_index('idx_monitoring_alerts_user', 'monitoring_alerts', ['user_id'])
    op.create_index('idx_monitoring_alerts_active', 'monitoring_alerts', ['active'])
    op.create_index('idx_monitoring_alerts_topic', 'monitoring_alerts', ['topic'])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes for monitoring_alerts
    op.drop_index('idx_monitoring_alerts_topic', table_name='monitoring_alerts')
    op.drop_index('idx_monitoring_alerts_active', table_name='monitoring_alerts')
    op.drop_index('idx_monitoring_alerts_user', table_name='monitoring_alerts')

    # Drop monitoring_alerts table
    op.drop_table('monitoring_alerts')

    # Drop indexes for literature_sources
    op.drop_index('idx_literature_sources_sync_time', table_name='literature_sources')
    op.drop_index('idx_literature_sources_status', table_name='literature_sources')
    op.drop_index('idx_literature_sources_type', table_name='literature_sources')

    # Drop literature_sources table
    op.drop_table('literature_sources')
