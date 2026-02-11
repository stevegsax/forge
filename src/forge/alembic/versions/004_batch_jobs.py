"""Add batch_jobs table.

Revision ID: 004
Revises: 003
Create Date: 2026-02-11
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "batch_jobs",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("batch_id", sa.String, nullable=False),
        sa.Column("workflow_id", sa.String, nullable=False),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_batch_jobs_batch_id", "batch_jobs", ["batch_id"])
    op.create_index("ix_batch_jobs_workflow_id", "batch_jobs", ["workflow_id"])


def downgrade() -> None:
    op.drop_table("batch_jobs")
