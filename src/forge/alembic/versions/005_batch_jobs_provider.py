"""Add provider column to batch_jobs table.

Revision ID: 005
Revises: 004
Create Date: 2026-02-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "batch_jobs",
        sa.Column("provider", sa.String, nullable=False, server_default="anthropic"),
    )


def downgrade() -> None:
    op.drop_column("batch_jobs", "provider")
