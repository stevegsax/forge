"""Add lifecycle status fields to ingested_sessions.

Revision ID: 005
Revises: 004
Create Date: 2026-04-28

Adds status, workflow identifiers, error message, and started_at so that
`pbook sessions` can show whether a session is running, completed, or
errored. Existing rows are pre-completion records, so they default to
status='completed'.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("ingested_sessions") as batch:
        batch.add_column(
            sa.Column(
                "status",
                sa.Text,
                nullable=False,
                server_default="completed",
            ),
        )
        batch.add_column(sa.Column("workflow_id", sa.Text, nullable=True))
        batch.add_column(sa.Column("run_id", sa.Text, nullable=True))
        batch.add_column(sa.Column("error_message", sa.Text, nullable=True))
        batch.add_column(sa.Column("started_at", sa.DateTime, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("ingested_sessions") as batch:
        batch.drop_column("started_at")
        batch.drop_column("error_message")
        batch.drop_column("run_id")
        batch.drop_column("workflow_id")
        batch.drop_column("status")
