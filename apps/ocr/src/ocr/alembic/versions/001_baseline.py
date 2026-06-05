"""OCR app baseline: ocr_results, ocr_images, ocr_file_content_blobs, ocr_job_status.

Squashed baseline (no corpus). Tables are in their final, S3-backed shape: blob
rows carry ``s3_key`` (no inline ``data``); no LargeBinary->s3 transition is
replayed. Coexists with the platform chain via ``version_table=alembic_version_ocr``
and the ``include_object`` filter in env.py.

Revision ID: 001
Revises:
Create Date: 2026-06-04
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ocr_results",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("document_id", sa.String, nullable=False, unique=True, index=True),
        sa.Column("file_path", sa.String, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("page_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("model_name", sa.String, nullable=False),
        sa.Column("input_tokens", sa.Integer, nullable=False),
        sa.Column("output_tokens", sa.Integer, nullable=False),
        sa.Column("batch_id", sa.String, nullable=False),
        sa.Column("workflow_id", sa.String, nullable=False),
        sa.Column("file_hash", sa.String, nullable=True, index=True),
        sa.Column(
            "marked_for_removal",
            sa.Boolean,
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "ocr_file_content_blobs",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("s3_key", sa.String, nullable=False),
        sa.Column("mime_type", sa.String, nullable=False),
        sa.Column("file_size_bytes", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "ocr_images",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("document_id", sa.String, nullable=False, server_default="", index=True),
        sa.Column("page_index", sa.Integer, nullable=False),
        sa.Column("original_image_id", sa.String, nullable=False),
        sa.Column("s3_key", sa.String, nullable=False),
        sa.Column("mime_type", sa.String, nullable=False),
        sa.Column("file_size_bytes", sa.Integer, nullable=False),
        sa.Column("top_left_x", sa.Integer, nullable=True),
        sa.Column("top_left_y", sa.Integer, nullable=True),
        sa.Column("bottom_right_x", sa.Integer, nullable=True),
        sa.Column("bottom_right_y", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "ocr_job_status",
        sa.Column("request_id", sa.String, primary_key=True),
        sa.Column("document_id", sa.String, nullable=False, index=True),
        sa.Column("file_path", sa.String, nullable=False, server_default=""),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("ocr_job_status")
    op.drop_table("ocr_images")
    op.drop_table("ocr_file_content_blobs")
    op.drop_table("ocr_results")
