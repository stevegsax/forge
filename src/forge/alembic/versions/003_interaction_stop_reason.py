"""Add nullable stop_reason column to interactions.

Records the LLM response's ``stop_reason`` (e.g. ``end_turn``, ``max_tokens``,
``tool_use``) alongside each interaction row. Adaptive thinking now competes
for tokens inside an unchanged ``max_tokens`` cap (2026-07 Phase 3 code
review), so a truncated response is a real risk worth being able to query for
after the fact — this column is the record-keeping half of that fix; the
other half is the parse-time ``logger.warning`` on ``stop_reason ==
"max_tokens"`` (``activities/batch_parse.py`` and the sync-path activity
wrappers).

Nullable and unbackfilled: existing rows predate stop_reason capture and have
no value to backfill.

Revision ID: 003
Revises: 002
Create Date: 2026-07-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("interactions", sa.Column("stop_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("interactions", "stop_reason")
