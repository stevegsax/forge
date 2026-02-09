"""Add playbooks table.

Revision ID: 002
Revises: 001
Create Date: 2026-02-09
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "playbooks",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("title", sa.String, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("tags_json", sa.Text, nullable=False),
        sa.Column("source_task_id", sa.String, nullable=False, index=True),
        sa.Column("source_workflow_id", sa.String, nullable=False),
        sa.Column("extraction_workflow_id", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("playbooks")
