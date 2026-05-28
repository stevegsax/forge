"""Move OCR blobs to S3: add s3_key, drop inline data on blob tables.

Replaces the inline ``data`` (LargeBinary) column on ``file_content_blobs`` and
``ocr_images`` with an ``s3_key`` reference. S3 becomes the only OCR blob store;
there is no inline-in-DB mode. Targets a fresh deploy (empty tables) — no backfill.

Revision ID: 014
Revises: 013
Create Date: 2026-05-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None

_BLOB_TABLES = ("file_content_blobs", "ocr_images")


def upgrade() -> None:
    # batch_alter_table handles SQLite (which can't drop columns natively) by
    # recreating the table; on Postgres it issues native ALTERs. Both tables are
    # empty on a fresh deploy, so adding a NOT NULL column needs no default.
    for table in _BLOB_TABLES:
        with op.batch_alter_table(table) as batch_op:
            batch_op.add_column(sa.Column("s3_key", sa.String, nullable=False))
            batch_op.drop_column("data")


def downgrade() -> None:
    for table in _BLOB_TABLES:
        with op.batch_alter_table(table) as batch_op:
            batch_op.add_column(sa.Column("data", sa.LargeBinary, nullable=False))
            batch_op.drop_column("s3_key")
