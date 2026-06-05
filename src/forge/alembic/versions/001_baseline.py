"""Forge platform baseline: interactions, runs, batch_jobs, playbooks.

Squashed baseline for the platform/consumer split (no corpus to preserve). Creates
the platform tables in their final shape; ``batch_jobs`` is generic provider-batch
coordination with no domain columns. OCR tables live in the OCR app's own chain.

Revision ID: 001
Revises:
Create Date: 2026-06-04
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "interactions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("idempotency_key", sa.String, nullable=True, unique=True),
        sa.Column("task_id", sa.String, nullable=False, index=True),
        sa.Column("step_id", sa.String, nullable=True),
        sa.Column("sub_task_id", sa.String, nullable=True),
        sa.Column("role", sa.String, nullable=False),
        sa.Column("system_prompt", sa.Text, nullable=False),
        sa.Column("user_prompt", sa.Text, nullable=False),
        sa.Column("model_name", sa.String, nullable=False),
        sa.Column("input_tokens", sa.Integer, nullable=False),
        sa.Column("output_tokens", sa.Integer, nullable=False),
        sa.Column("latency_ms", sa.Float, nullable=False),
        sa.Column("explanation", sa.Text, server_default=""),
        sa.Column("context_stats_json", sa.Text, nullable=True),
        sa.Column("cache_creation_input_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("cache_read_input_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String, nullable=False, index=True),
        sa.Column("workflow_id", sa.String, nullable=False, unique=True),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("result_json", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "batch_jobs",
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("batch_id", sa.String, nullable=True, index=True),
        sa.Column("workflow_id", sa.String, nullable=False, index=True),
        sa.Column("status", sa.String, nullable=False),
        sa.Column("provider", sa.String, nullable=False, server_default="anthropic"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "playbooks",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("idempotency_key", sa.String, nullable=True, unique=True),
        sa.Column("title", sa.String, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("tags_json", sa.Text, nullable=False),
        sa.Column("source_task_id", sa.String, nullable=False, index=True),
        sa.Column("source_workflow_id", sa.String, nullable=False),
        sa.Column("extraction_workflow_id", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("playbooks")
    op.drop_table("batch_jobs")
    op.drop_table("runs")
    op.drop_table("interactions")
