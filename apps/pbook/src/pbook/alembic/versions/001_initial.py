"""Create entries table.

Revision ID: 001
Revises:
Create Date: 2026-04-08
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "entries",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("title", sa.String, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("tags_json", sa.Text, nullable=False),
        sa.Column("entry_type", sa.String, nullable=False, server_default="curated"),
        sa.Column("source_project", sa.String, nullable=False, server_default=""),
        sa.Column("source_task_id", sa.String, nullable=False, server_default=""),
        sa.Column("needs_review", sa.Boolean, nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_entries_source_project", "entries", ["source_project"])
    op.create_index("ix_entries_entry_type", "entries", ["entry_type"])


def downgrade() -> None:
    op.drop_table("entries")
