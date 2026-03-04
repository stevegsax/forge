"""Create ocr_images table for storing extracted document images.

Revision ID: 009
Revises: 008
Create Date: 2026-03-04
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ocr_images",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("document_id", sa.String, nullable=False, server_default=""),
        sa.Column("page_index", sa.Integer, nullable=False),
        sa.Column("original_image_id", sa.String, nullable=False),
        sa.Column("data", sa.LargeBinary, nullable=False),
        sa.Column("mime_type", sa.String, nullable=False),
        sa.Column("file_size_bytes", sa.Integer, nullable=False),
        sa.Column("top_left_x", sa.Integer, nullable=True),
        sa.Column("top_left_y", sa.Integer, nullable=True),
        sa.Column("bottom_right_x", sa.Integer, nullable=True),
        sa.Column("bottom_right_y", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_ocr_images_document_id", "ocr_images", ["document_id"])


def downgrade() -> None:
    op.drop_index("ix_ocr_images_document_id")
    op.drop_table("ocr_images")
