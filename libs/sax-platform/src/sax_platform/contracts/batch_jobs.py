"""Read-only ``batch_jobs`` schema for consumer apps.

The platform owns the ``batch_jobs`` table (writes via ``forge.store``). A consumer
app needs to *read* it — e.g. to join provider-batch status onto its own status
table — without importing ``forge``. This module exposes a standalone SQLAlchemy
``Table`` mirror on its own ``MetaData`` (so it is never managed by the consumer's
Alembic chain). It is read-only by convention: consumers SELECT it and never write.

The columns mirror the platform's slim, generic ``batch_jobs`` (no domain fields).
"""

from __future__ import annotations

import sqlalchemy as sa

from .types import UTCDateTime

metadata = sa.MetaData()

batch_jobs = sa.Table(
    "batch_jobs",
    metadata,
    sa.Column("id", sa.String, primary_key=True),
    sa.Column("batch_id", sa.String, nullable=True),
    sa.Column("workflow_id", sa.String, nullable=False),
    sa.Column("status", sa.String, nullable=False),
    sa.Column("provider", sa.String, nullable=False),
    sa.Column("error_message", sa.Text, nullable=True),
    sa.Column("created_at", UTCDateTime),
    sa.Column("updated_at", UTCDateTime),
)
