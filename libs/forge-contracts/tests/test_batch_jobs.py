"""Tests for the read-only ``batch_jobs`` mirror table.

This module exposes a standalone ``Table`` on its own ``MetaData`` so a
consumer app can SELECT the platform's ``batch_jobs`` rows without pulling in
the platform's Alembic chain. The tests pin the schema shape (columns,
nullability) that consumers rely on, and prove the table is independently
creatable/queryable — i.e. that it really is on its own ``MetaData``, not
accidentally shared with anything else.
"""

from __future__ import annotations

import sqlalchemy as sa

from forge_contracts.batch_jobs import batch_jobs, metadata


class TestSchemaShape:
    def test_table_name(self) -> None:
        assert batch_jobs.name == "batch_jobs"

    def test_table_is_on_its_own_metadata(self) -> None:
        assert batch_jobs.metadata is metadata

    def test_id_is_primary_key(self) -> None:
        assert [c.name for c in batch_jobs.primary_key.columns] == ["id"]

    def test_nullability(self) -> None:
        cols = {c.name: c.nullable for c in batch_jobs.columns}
        assert cols == {
            "id": False,
            "batch_id": True,
            "workflow_id": False,
            "status": False,
            "provider": False,
            "error_message": True,
            "created_at": True,
            "updated_at": True,
        }


class TestStandaloneUsability:
    def test_creatable_and_queryable_without_platform_metadata(self) -> None:
        engine = sa.create_engine("sqlite://")
        metadata.create_all(engine)

        with engine.begin() as conn:
            conn.execute(
                batch_jobs.insert().values(
                    id="req-1",
                    batch_id="b-1",
                    workflow_id="wf-1",
                    status="submitted",
                    provider="anthropic",
                )
            )

        with engine.connect() as conn:
            row = conn.execute(sa.select(batch_jobs).where(batch_jobs.c.id == "req-1")).one()

        assert row.status == "submitted"
        assert row.batch_id == "b-1"
        assert row.error_message is None
