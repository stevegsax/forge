"""Add embedding column to entries table.

Revision ID: 003
Revises: 002
Create Date: 2026-04-09

Adds a BLOB column to store vector embeddings for semantic search.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "entries",
        sa.Column("embedding", sa.LargeBinary, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("entries", "embedding")
