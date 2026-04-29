"""Add soft-rejection columns to entries.

Revision ID: 007
Revises: 006
Create Date: 2026-04-28

`pbook reject <id>` previously deleted the entry. With these columns
it becomes a soft-mark: the row stays, `rejected=1`, and an optional
`rejection_reason` survives for audit. Default queries hide rejected
rows; consumers can opt in via include_rejected.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("entries") as batch:
        batch.add_column(
            sa.Column(
                "rejected",
                sa.Boolean,
                nullable=False,
                server_default=sa.text("0"),
            ),
        )
        batch.add_column(sa.Column("rejection_reason", sa.Text, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("entries") as batch:
        batch.drop_column("rejection_reason")
        batch.drop_column("rejected")
