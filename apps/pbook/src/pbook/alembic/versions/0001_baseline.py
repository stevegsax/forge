"""Postgres-clean baseline for the pbook schema.

Revision ID: 0001
Revises:
Create Date: 2026-06-01

Single squashed baseline (no SQLite history). Creates the ``pbook``
schema's pbk_-prefixed tables with pgvector embeddings, TIMESTAMPTZ
timestamps, boolean defaults, and IDENTITY primary keys. The schema and
the ``vector`` extension are ensured in ``env.py`` before this runs so
the (schema-qualified) version table can be created.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

SCHEMA = "pbook"
EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.create_table(
        "pbk_entries",
        sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("entry_type", sa.Text, nullable=False, server_default="curated"),
        sa.Column("source_project", sa.Text, nullable=False, server_default=""),
        sa.Column("source_task_id", sa.Text, nullable=False, server_default=""),
        sa.Column("needs_review", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("helpful_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("harmful_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("retrieval_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("rejected", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("rejection_reason", sa.Text, nullable=True),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_pbk_entries_source_project",
        "pbk_entries",
        ["source_project"],
        schema=SCHEMA,
    )
    op.create_index("ix_pbk_entries_entry_type", "pbk_entries", ["entry_type"], schema=SCHEMA)
    # Approximate-nearest-neighbour index for cosine similarity search.
    op.execute(
        f"CREATE INDEX ix_pbk_entries_embedding ON {SCHEMA}.pbk_entries "
        f"USING hnsw (embedding vector_cosine_ops)",
    )

    op.create_table(
        "pbk_entry_tags",
        sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column(
            "entry_id",
            sa.BigInteger,
            sa.ForeignKey(f"{SCHEMA}.pbk_entries.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tag", sa.Text, nullable=False),
        sa.UniqueConstraint("entry_id", "tag", name="uq_pbk_entry_tags_entry_tag"),
        schema=SCHEMA,
    )
    op.create_index("ix_pbk_entry_tags_tag", "pbk_entry_tags", ["tag"], schema=SCHEMA)
    op.create_index(
        "ix_pbk_entry_tags_entry_id",
        "pbk_entry_tags",
        ["entry_id"],
        schema=SCHEMA,
    )

    op.create_table(
        "pbk_entry_sources",
        sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True),
        sa.Column(
            "entry_id",
            sa.BigInteger,
            sa.ForeignKey(f"{SCHEMA}.pbk_entries.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("session_id", sa.Text, nullable=False, server_default=""),
        sa.Column("project_name", sa.Text, nullable=False, server_default=""),
        sa.Column("experience_hash", sa.Text, nullable=True),
        sa.Column("source_context", sa.Text, nullable=False, server_default=""),
        sa.Column("source_context_embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "entry_id",
            "session_id",
            "experience_hash",
            name="uq_pbk_entry_sources_entry_session_hash",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_pbk_entry_sources_session_id",
        "pbk_entry_sources",
        ["session_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_pbk_entry_sources_entry_id",
        "pbk_entry_sources",
        ["entry_id"],
        schema=SCHEMA,
    )

    op.create_table(
        "pbk_ingested_sessions",
        sa.Column("session_id", sa.Text, primary_key=True),
        sa.Column("project_name", sa.Text, nullable=False, server_default=""),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("experiences_found", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("entries_created", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("status", sa.Text, nullable=False, server_default="completed"),
        sa.Column("workflow_id", sa.Text, nullable=True),
        sa.Column("run_id", sa.Text, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("pbk_ingested_sessions", schema=SCHEMA)
    op.drop_table("pbk_entry_sources", schema=SCHEMA)
    op.drop_table("pbk_entry_tags", schema=SCHEMA)
    op.drop_index("ix_pbk_entries_embedding", table_name="pbk_entries", schema=SCHEMA)
    op.drop_table("pbk_entries", schema=SCHEMA)
