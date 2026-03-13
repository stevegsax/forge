"""Add file_hash column to ocr_results for content-based duplicate detection.

Revision ID: 012
Revises: 011
Create Date: 2026-03-12
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ocr_results",
        sa.Column("file_hash", sa.String, nullable=True),
    )
    op.create_index("ix_ocr_results_file_hash", "ocr_results", ["file_hash"])


def downgrade() -> None:
    op.drop_index("ix_ocr_results_file_hash", table_name="ocr_results")
    op.drop_column("ocr_results", "file_hash")
