"""Add marked_for_removal flag to ocr_results.

Revision ID: 010
Revises: 009
Create Date: 2026-03-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ocr_results",
        sa.Column(
            "marked_for_removal",
            sa.Boolean,
            nullable=False,
            # sa.false() renders dialect-correctly ("0" on SQLite, "false" on
            # Postgres); a literal "0" is an integer and Postgres rejects it for
            # a BOOLEAN column.
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("ocr_results", "marked_for_removal")
