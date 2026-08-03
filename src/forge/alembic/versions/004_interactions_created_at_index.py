"""Add a created_at index to interactions.

The interactions table is the authoritative spend record (D97): terminal
results carry surviving-call totals only, so every cost or audit question
— "what ran in the last hour", "what did today cost" — resolves to a
time-windowed read of this table, and T7.3 (honest token accounting)
plans its calibration reads over recent windows of the same table. No
index covers ``created_at``, so each such query is a sequential scan.

Index-only, no table rewrite. Built ``CONCURRENTLY`` (the checklist habit
for indexes on existing tables) and ``IF NOT EXISTS`` (a non-transactional
phase must be resumable — the schema-change process's recovery from a
partial failure is re-run the phase). The table is near-empty today; the
concurrency choice is habit and lint-cleanliness, not present need. This
revision is also forge's maiden schema-change request under the
consumer/operator process (sax-datastores issue #2), deliberately small.

Revision ID: 004
Revises: 003
Create Date: 2026-08-02
"""

from __future__ import annotations

from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.create_index(
            "ix_interactions_created_at",
            "interactions",
            ["created_at"],
            unique=False,
            if_not_exists=True,
            postgresql_concurrently=True,
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index(
            "ix_interactions_created_at",
            table_name="interactions",
            if_exists=True,
            postgresql_concurrently=True,
        )
