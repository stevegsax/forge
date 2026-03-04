"""Make batch_jobs.batch_id nullable for failed submissions.

Revision ID: 008
Revises: 007
Create Date: 2026-03-03
"""

from __future__ import annotations

from alembic import op

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("batch_jobs") as batch_op:
        batch_op.alter_column("batch_id", nullable=True)


def downgrade() -> None:
    with op.batch_alter_table("batch_jobs") as batch_op:
        batch_op.alter_column("batch_id", nullable=False)
