"""Create file_content_blobs table.

Revision ID: 007
Revises: 006
Create Date: 2026-03-03
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "file_content_blobs",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("data", sa.LargeBinary, nullable=False),
        sa.Column("mime_type", sa.String, nullable=False),
        sa.Column("file_size_bytes", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("file_content_blobs")
