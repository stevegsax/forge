"""Add idempotency_key to interactions and playbooks for survivable writes.

A nullable, UNIQUE ``idempotency_key`` lets the survivable ``persist_to_store``
activity dedupe re-applied writes on retry. Nullable so legacy/direct inserts that
don't supply a key still work (multiple NULLs are allowed by a UNIQUE index in both
SQLite and Postgres); the workflow always supplies a deterministic key.

Revision ID: 015
Revises: 014
Create Date: 2026-05-28
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None

_TABLES = ("interactions", "playbooks")


def upgrade() -> None:
    for table in _TABLES:
        with op.batch_alter_table(table) as batch_op:
            batch_op.add_column(sa.Column("idempotency_key", sa.String, nullable=True))
            batch_op.create_unique_constraint(f"uq_{table}_idempotency_key", ["idempotency_key"])


def downgrade() -> None:
    for table in _TABLES:
        with op.batch_alter_table(table) as batch_op:
            batch_op.drop_constraint(f"uq_{table}_idempotency_key", type_="unique")
            batch_op.drop_column("idempotency_key")
