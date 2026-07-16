"""Enable Row Level Security on the pbk_ tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-02

Supabase flags any table without RLS as exposed to the ``anon`` /
``authenticated`` roles via PostgREST. pbook connects as the ``postgres``
role over a direct/pooler Postgres connection, which bypasses RLS, so
enabling RLS with NO policies is safe for pbook (its access is
unaffected) while closing the anon-key exposure: with RLS on and no
policy, anon/authenticated get zero rows.
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

SCHEMA = "pbook"
_TABLES = (
    "pbk_entries",
    "pbk_entry_tags",
    "pbk_entry_sources",
    "pbk_ingested_sessions",
    "pbk_alembic_version",
)


def upgrade() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE {SCHEMA}.{table} ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE {SCHEMA}.{table} DISABLE ROW LEVEL SECURITY")
