"""Add file_path column to batch_jobs.

Revision ID: 011
Revises: 010
Create Date: 2026-03-12
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "batch_jobs",
        sa.Column("file_path", sa.String, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("batch_jobs", "file_path")
