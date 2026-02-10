"""Add cache token columns to interactions table.

Revision ID: 003
Revises: 002
Create Date: 2026-02-10
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
        "interactions",
        sa.Column("cache_creation_input_tokens", sa.Integer, nullable=False, server_default="0"),
    )
    op.add_column(
        "interactions",
        sa.Column("cache_read_input_tokens", sa.Integer, nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("interactions", "cache_read_input_tokens")
    op.drop_column("interactions", "cache_creation_input_tokens")
