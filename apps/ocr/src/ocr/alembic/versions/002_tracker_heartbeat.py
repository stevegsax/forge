"""OCR tracker heartbeat: single-row liveness marker for the T4.4 status tracker.

Adds ``ocr_tracker_heartbeat``, a singleton row (``id == 1``) the stateless status
tracker upserts each ~2-minute sweep. Coexists with the platform chain via
``version_table=alembic_version_ocr`` and the ``include_object`` filter in env.py.

Revision ID: 002
Revises: 001
Create Date: 2026-07-22
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ocr_tracker_heartbeat",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("last_run_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("live_jobs", sa.Integer, nullable=False, server_default="0"),
        sa.Column("hints_sent", sa.Integer, nullable=False, server_default="0"),
        sa.Column("cycles_total", sa.Integer, nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_table("ocr_tracker_heartbeat")
