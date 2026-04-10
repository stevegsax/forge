"""Add feedback counters to entries table.

Revision ID: 002
Revises: 001
Create Date: 2026-04-09

Adds helpful_count, harmful_count, and retrieval_count columns
to support ACE-inspired helpfulness tracking.  All columns default
to 0 so existing rows need no backfill.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "entries",
        sa.Column("helpful_count", sa.Integer, nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "entries",
        sa.Column("harmful_count", sa.Integer, nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "entries",
        sa.Column("retrieval_count", sa.Integer, nullable=False, server_default=sa.text("0")),
    )


def downgrade() -> None:
    op.drop_column("entries", "retrieval_count")
    op.drop_column("entries", "harmful_count")
    op.drop_column("entries", "helpful_count")
