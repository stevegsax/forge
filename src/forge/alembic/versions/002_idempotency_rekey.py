"""Idempotency rekey: runs (workflow_id -> workflow_id, run_id) + interaction keys.

Reruns of a task reuse the deterministic ``forge-task-{id}`` workflow_id, so the
old ``unique(workflow_id)`` runs key and positional ``{wf}:{role}:{seq}``
interaction keys made ``insert_or_ignore`` silently swallow every rerun row
(T1.6a). This migration:

- adds ``runs.run_id`` and rekeys the table to a composite unique
  ``(workflow_id, run_id)``;
- reshapes existing interaction idempotency keys to the new
  ``{wf}:{run_id}:{role}:{occurrence}`` shape.

Existing rows are preserved. A sentinel ``run_id`` of ``"legacy"`` is backfilled
for pre-migration rows (each old workflow_id was unique, so ``(workflow_id,
"legacy")`` stays unique); the old positional seq is reused verbatim as the
occurrence (injective, so no transient uniqueness violation).

Dialect-branched on purpose: SQLite goes through ``batch_alter_table`` (move and
copy), while Postgres uses direct ALTERs so it never recreates the table and
resets the ``runs.id`` SERIAL sequence.

Revision ID: 002
Revises: 001
Create Date: 2026-07-15
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from forge.persist_models import (
    reshape_legacy_interaction_key,
    restore_legacy_interaction_key,
)

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None

# Sentinel run_id backfilled onto pre-migration rows/keys (they carry no real
# Temporal run_id). Downgrade only reverses keys still bearing this sentinel.
_LEGACY_RUN_ID = "legacy"
_RUNS_UQ = "uq_runs_workflow_id_run_id"
_RUNS_WF_UQ = "uq_runs_workflow_id"


def _runs_table(*, with_run_id: bool, with_composite_uq: bool) -> sa.Table:
    """Describe the runs table for SQLite ``copy_from`` recreation.

    Mirrors the baseline columns (plus ``run_id`` when present). Constraints are
    included/omitted per the flags so the recreate drops exactly what we want.
    """
    columns: list[sa.Column] = [
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String, nullable=False),
        sa.Column("workflow_id", sa.String, nullable=False),
    ]
    if with_run_id:
        columns.append(sa.Column("run_id", sa.String, nullable=True))
    columns += [
        sa.Column("status", sa.String, nullable=False),
        sa.Column("result_json", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    ]
    args: list[object] = [sa.Index("ix_runs_task_id", "task_id")]
    if with_composite_uq:
        args.append(sa.UniqueConstraint("workflow_id", "run_id", name=_RUNS_UQ))
    return sa.Table("runs", sa.MetaData(), *columns, *args)


def _rekey_interaction_keys(bind: sa.engine.Connection, *, forward: bool) -> None:
    """Rewrite interaction idempotency keys in place (pure transform per row)."""
    interactions = sa.table(
        "interactions",
        sa.column("id", sa.Integer),
        sa.column("idempotency_key", sa.String),
    )
    rows = bind.execute(
        sa.select(interactions.c.id, interactions.c.idempotency_key).where(
            interactions.c.idempotency_key.is_not(None)
        )
    ).all()
    transform = reshape_legacy_interaction_key if forward else restore_legacy_interaction_key
    for row_id, key in rows:
        new_key = transform(key, run_id=_LEGACY_RUN_ID)
        if new_key != key:
            bind.execute(
                interactions.update()
                .where(interactions.c.id == row_id)
                .values(idempotency_key=new_key)
            )


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    # --- runs: add run_id, backfill sentinel ---
    op.add_column("runs", sa.Column("run_id", sa.String(), nullable=True))
    runs = sa.table("runs", sa.column("run_id", sa.String))
    op.execute(runs.update().where(runs.c.run_id.is_(None)).values(run_id=_LEGACY_RUN_ID))

    # --- runs: swap unique(workflow_id) -> unique(workflow_id, run_id) ---
    if dialect == "postgresql":
        op.alter_column("runs", "run_id", existing_type=sa.String(), nullable=False)
        insp = sa.inspect(bind)
        for uc in insp.get_unique_constraints("runs"):
            if uc["column_names"] == ["workflow_id"] and uc["name"]:
                op.drop_constraint(uc["name"], "runs", type_="unique")
        for ix in insp.get_indexes("runs"):
            if ix.get("unique") and ix["column_names"] == ["workflow_id"] and ix["name"]:
                op.drop_index(ix["name"], table_name="runs")
        op.create_unique_constraint(_RUNS_UQ, "runs", ["workflow_id", "run_id"])
    else:
        with op.batch_alter_table(
            "runs",
            copy_from=_runs_table(with_run_id=True, with_composite_uq=False),
            recreate="always",
        ) as batch_op:
            batch_op.alter_column("run_id", existing_type=sa.String(), nullable=False)
            batch_op.create_unique_constraint(_RUNS_UQ, ["workflow_id", "run_id"])

    # --- interactions: reshape legacy keys to include run_id ---
    _rekey_interaction_keys(bind, forward=True)


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    # --- interactions: strip the sentinel run_id back out ---
    _rekey_interaction_keys(bind, forward=False)

    # --- runs: restore unique(workflow_id); drop run_id ---
    # Note: restoring the single-column unique fails if reruns recorded multiple
    # rows per workflow_id after upgrade — inherent to reversing the rekey.
    if dialect == "postgresql":
        op.drop_constraint(_RUNS_UQ, "runs", type_="unique")
        op.create_unique_constraint(_RUNS_WF_UQ, "runs", ["workflow_id"])
        op.drop_column("runs", "run_id")
    else:
        with op.batch_alter_table(
            "runs",
            copy_from=_runs_table(with_run_id=True, with_composite_uq=False),
            recreate="always",
        ) as batch_op:
            batch_op.create_unique_constraint(_RUNS_WF_UQ, ["workflow_id"])
            batch_op.drop_column("run_id")
