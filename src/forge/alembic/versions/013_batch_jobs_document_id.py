"""Add document_id column to batch_jobs.

Gives each OCR submission a stable grouping identifier so that
resubmissions of the same file are not collapsed with their
predecessors by the list_ocr_jobs query.

Historical rows are backfilled with `document_id = id` (treating each
existing row as its own single-chunk submission).

Revision ID: 013
Revises: 012
Create Date: 2026-04-13
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "batch_jobs",
        sa.Column("document_id", sa.String, nullable=True),
    )
    op.create_index(
        "ix_batch_jobs_document_id",
        "batch_jobs",
        ["document_id"],
    )
    # Backfill: treat each historical row as its own submission.
    op.execute("UPDATE batch_jobs SET document_id = id WHERE document_id IS NULL")


def downgrade() -> None:
    op.drop_index("ix_batch_jobs_document_id", table_name="batch_jobs")
    op.drop_column("batch_jobs", "document_id")
