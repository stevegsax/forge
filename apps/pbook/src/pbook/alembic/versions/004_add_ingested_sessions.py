"""Add ingested_sessions table for tracking processed Claude Code transcripts.

Revision ID: 004
Revises: 003
Create Date: 2026-04-10

Tracks which Claude Code session IDs have been processed to avoid
reprocessing on subsequent ingestion runs.
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
        "ingested_sessions",
        sa.Column("session_id", sa.Text, primary_key=True),
        sa.Column("project_name", sa.Text, nullable=False, server_default=""),
        sa.Column(
            "ingested_at",
            sa.DateTime,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("experiences_found", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("entries_created", sa.Integer, nullable=False, server_default=sa.text("0")),
    )


def downgrade() -> None:
    op.drop_table("ingested_sessions")
