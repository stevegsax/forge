"""Add entry_sources join table for playbook provenance.

Revision ID: 006
Revises: 005
Create Date: 2026-04-28

Records the originating Claude Code session(s) and situations that
caused each playbook to be created. Granularity is per-experience: one
row per (entry, experience), so a single session contributing many
distinct situations to the same playbook yields many rows.
``experience_hash`` is ``sha256(problem + resolution + context)`` and is
nullable so future manual-attribution rows can omit it.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "entry_sources",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "entry_id",
            sa.Integer,
            sa.ForeignKey("entries.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("session_id", sa.Text, nullable=False, server_default=""),
        sa.Column("project_name", sa.Text, nullable=False, server_default=""),
        sa.Column("experience_hash", sa.Text, nullable=True),
        sa.Column("source_context", sa.Text, nullable=False, server_default=""),
        sa.Column("source_context_embedding", sa.LargeBinary, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime,
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "entry_id", "session_id", "experience_hash",
            name="uq_entry_sources_entry_session_hash",
        ),
    )
    op.create_index(
        "ix_entry_sources_session_id", "entry_sources", ["session_id"],
    )
    op.create_index(
        "ix_entry_sources_entry_id", "entry_sources", ["entry_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_entry_sources_entry_id", table_name="entry_sources")
    op.drop_index("ix_entry_sources_session_id", table_name="entry_sources")
    op.drop_table("entry_sources")
