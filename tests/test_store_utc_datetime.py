"""Tests for the UTCDateTime TypeDecorator in forge.store.

The decorator normalizes all DateTime columns to tz-aware UTC on read,
and converts tz-aware inputs to UTC on write. Naive inputs are assumed
to already be UTC per project convention.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import sqlalchemy as sa

from forge.store import (
    BatchJob,
    get_batch_job,
    get_engine,
    record_batch_submission,
    run_migrations,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.engine import Engine


def _setup_db(tmp_path: Path):
    db_path = tmp_path / "test.db"
    run_migrations(db_path)
    return get_engine(db_path), db_path


def _read_created_at(engine: Engine, request_id: str) -> datetime:
    """Read back the created_at column via the ORM mapping."""
    t = BatchJob.__table__
    with engine.connect() as conn:
        row = conn.execute(
            t.select().where(t.c.id == request_id)
        ).mappings().first()
    assert row is not None
    return row["created_at"]


class TestUTCDateTime:
    """Round-trip semantics for the UTCDateTime TypeDecorator."""

    def test_write_aware_utc_read_back_aware_utc(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        record_batch_submission(
            engine,
            request_id="req-aware",
            batch_id="b-1",
            workflow_id="wf-1",
        )

        read_back = _read_created_at(engine, "req-aware")
        assert read_back.tzinfo is not None
        assert read_back.tzinfo == UTC

    def test_write_non_utc_aware_normalized_to_utc(self, tmp_path: Path) -> None:
        """Writing a tz-aware non-UTC datetime stores the equivalent UTC instant."""
        engine, _ = _setup_db(tmp_path)

        la_tz = ZoneInfo("America/Los_Angeles")
        # 2026-04-13 10:00 PDT = 2026-04-13 17:00 UTC
        la_dt = datetime(2026, 4, 13, 10, 0, 0, tzinfo=la_tz)

        with engine.begin() as conn:
            conn.execute(
                sa.insert(BatchJob.__table__).values(
                    id="req-la",
                    batch_id="b-la",
                    workflow_id="wf-la",
                    status="submitted",
                    provider="mistral",
                    file_path="/data/la.pdf",
                    document_id="doc-la",
                    created_at=la_dt,
                    updated_at=la_dt,
                )
            )

        read_back = _read_created_at(engine, "req-la")
        assert read_back.tzinfo == UTC
        # Same instant in time, now expressed in UTC
        assert read_back == la_dt
        # And the wall-clock component really is UTC
        expected_utc = datetime(2026, 4, 13, 17, 0, 0, tzinfo=UTC)
        assert read_back == expected_utc

    def test_write_naive_passthrough_read_back_utc(self, tmp_path: Path) -> None:
        """A naive datetime is assumed UTC and comes back with UTC tzinfo."""
        engine, _ = _setup_db(tmp_path)

        naive = datetime(2026, 4, 13, 17, 0, 0)  # naive; convention is UTC
        with engine.begin() as conn:
            conn.execute(
                sa.insert(BatchJob.__table__).values(
                    id="req-naive",
                    batch_id="b-naive",
                    workflow_id="wf-naive",
                    status="submitted",
                    provider="mistral",
                    file_path="/data/naive.pdf",
                    document_id="doc-naive",
                    created_at=naive,
                    updated_at=naive,
                )
            )

        read_back = _read_created_at(engine, "req-naive")
        assert read_back.tzinfo == UTC
        assert read_back.year == 2026
        assert read_back.month == 4
        assert read_back.day == 13
        assert read_back.hour == 17
        assert read_back.minute == 0

    def test_server_default_read_as_utc(self, tmp_path: Path) -> None:
        """Alembic server_default=sa.func.now() rows still read back UTC-aware.

        Simulate by omitting created_at/updated_at entirely and relying on
        SQLAlchemy's Python default (which uses datetime.now(UTC)). The
        TypeDecorator reattaches UTC on read regardless of source.
        """
        engine, _ = _setup_db(tmp_path)

        with engine.begin() as conn:
            conn.execute(
                sa.insert(BatchJob.__table__).values(
                    id="req-default",
                    batch_id="b-default",
                    workflow_id="wf-default",
                    status="submitted",
                    provider="mistral",
                    file_path="/data/default.pdf",
                    document_id="doc-default",
                )
            )

        read_back = _read_created_at(engine, "req-default")
        assert read_back.tzinfo == UTC
        # Should be close to "now"
        delta = datetime.now(UTC) - read_back
        assert timedelta(0) <= delta < timedelta(seconds=30)

    def test_fixed_offset_aware_normalized(self, tmp_path: Path) -> None:
        """A fixed-offset tz (not UTC, not named) round-trips to UTC."""
        engine, _ = _setup_db(tmp_path)

        # UTC-05:00 (e.g. EST without DST)
        offset_tz = timezone(timedelta(hours=-5))
        local_dt = datetime(2026, 4, 13, 12, 0, 0, tzinfo=offset_tz)
        expected_utc = datetime(2026, 4, 13, 17, 0, 0, tzinfo=UTC)

        with engine.begin() as conn:
            conn.execute(
                sa.insert(BatchJob.__table__).values(
                    id="req-offset",
                    batch_id="b-off",
                    workflow_id="wf-off",
                    status="submitted",
                    provider="mistral",
                    file_path="/data/off.pdf",
                    document_id="doc-off",
                    created_at=local_dt,
                    updated_at=local_dt,
                )
            )

        read_back = _read_created_at(engine, "req-offset")
        assert read_back.tzinfo == UTC
        assert read_back == expected_utc

    def test_all_models_emit_utc(self, tmp_path: Path) -> None:
        """Every model with a datetime column returns tz-aware UTC."""
        from forge.store import (
            FileContentBlob,
            Interaction,
            OcrImage,
            OcrResult,
            Playbook,
            Run,
            save_file_content,
            save_ocr_image,
            save_ocr_result,
        )

        engine, _ = _setup_db(tmp_path)

        # Interaction — insert via ORM table directly (save_interaction
        # takes a composite dict builder; use lower-level insert here).
        with engine.begin() as conn:
            conn.execute(
                sa.insert(Interaction.__table__).values(
                    task_id="t",
                    role="assistant",
                    system_prompt="",
                    user_prompt="",
                    model_name="m",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0.0,
                )
            )
            conn.execute(
                sa.insert(Run.__table__).values(
                    task_id="t",
                    workflow_id="wf-run-1",
                    status="succeeded",
                    result_json="{}",
                )
            )
            conn.execute(
                sa.insert(Playbook.__table__).values(
                    title="t",
                    content="c",
                    tags_json="[]",
                    source_task_id="src",
                    source_workflow_id="wf",
                    extraction_workflow_id="wf",
                )
            )

        save_ocr_result(
            engine,
            document_id="doc-all",
            file_path="/data/all.pdf",
            text="text",
            model_name="m",
            input_tokens=0,
            output_tokens=0,
            batch_id="b",
            workflow_id="wf",
        )
        save_file_content(
            engine,
            content_id="blob-all",
            data=b"bytes",
            mime_type="application/pdf",
            file_size_bytes=5,
        )
        save_ocr_image(
            engine,
            image_id="img-all",
            page_index=0,
            original_image_id="orig",
            data=b"img",
            mime_type="image/png",
            file_size_bytes=3,
        )
        record_batch_submission(
            engine,
            request_id="req-all",
            batch_id="b-all",
            workflow_id="wf-all",
        )

        checks = [
            ("interactions", Interaction.__table__),
            ("runs", Run.__table__),
            ("batch_jobs", BatchJob.__table__),
            ("playbooks", Playbook.__table__),
            ("ocr_results", OcrResult.__table__),
            ("file_content_blobs", FileContentBlob.__table__),
            ("ocr_images", OcrImage.__table__),
        ]
        with engine.connect() as conn:
            for name, table in checks:
                row = conn.execute(
                    table.select().limit(1)
                ).mappings().first()
                assert row is not None, f"no row inserted for {name}"
                assert row["created_at"].tzinfo == UTC, (
                    f"{name}.created_at is not UTC-aware: {row['created_at']!r}"
                )
                if name == "batch_jobs":
                    assert row["updated_at"].tzinfo == UTC, (
                        f"{name}.updated_at is not UTC-aware"
                    )

    def test_get_batch_job_returns_utc_aware(self, tmp_path: Path) -> None:
        """The get_batch_job helper should return rows with UTC-aware datetimes."""
        engine, _ = _setup_db(tmp_path)
        record_batch_submission(
            engine,
            request_id="req-get",
            batch_id="b-get",
            workflow_id="wf-get",
        )

        job = get_batch_job(engine, "req-get")
        assert job is not None
        assert job["created_at"].tzinfo == UTC
        assert job["updated_at"].tzinfo == UTC
