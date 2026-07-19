"""Tests for batch job store functions (Phase 14)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import sqlalchemy as sa

from forge.store import (
    BatchJob,
    get_batch_job,
    get_pending_batch_jobs,
    record_batch_failure,
    record_batch_submission,
    run_migrations,
    update_batch_status,
)

if TYPE_CHECKING:
    from pathlib import Path


def _setup_db(tmp_path: Path):
    """Create a test database with migrations applied."""
    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    run_migrations(url)
    return sa.create_engine(url), db_path


# ---------------------------------------------------------------------------
# record_batch_submission
# ---------------------------------------------------------------------------


class TestRecordBatchSubmission:
    def test_insert_and_verify(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        record_batch_submission(
            engine,
            request_id="req-001",
            batch_id="msgbatch_abc",
            workflow_id="wf-123",
        )

        job = get_batch_job(engine, "req-001")
        assert job is not None
        assert job["id"] == "req-001"
        assert job["batch_id"] == "msgbatch_abc"
        assert job["workflow_id"] == "wf-123"

    def test_status_is_submitted(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        record_batch_submission(
            engine,
            request_id="req-002",
            batch_id="msgbatch_def",
            workflow_id="wf-456",
        )

        job = get_batch_job(engine, "req-002")
        assert job is not None
        assert job["status"] == "submitted"


# ---------------------------------------------------------------------------
# record_batch_failure
# ---------------------------------------------------------------------------


class TestRecordBatchFailure:
    def test_inserts_failed_record(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        record_batch_failure(
            engine,
            request_id="req-fail-001",
            workflow_id="wf-fail-1",
            error_message="400 Bad Request: invalid input",
            provider="mistral",
        )

        job = get_batch_job(engine, "req-fail-001")
        assert job is not None
        assert job["status"] == "failed"
        assert job["batch_id"] is None
        assert job["error_message"] == "400 Bad Request: invalid input"
        assert job["provider"] == "mistral"
        assert job["workflow_id"] == "wf-fail-1"


# ---------------------------------------------------------------------------
# update_batch_status
# ---------------------------------------------------------------------------


class TestUpdateBatchStatus:
    def test_update_status(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        record_batch_submission(
            engine,
            request_id="req-010",
            batch_id="msgbatch_x",
            workflow_id="wf-1",
        )

        update_batch_status(engine, request_id="req-010", status="processing")

        job = get_batch_job(engine, "req-010")
        assert job is not None
        assert job["status"] == "processing"

    def test_update_rejects_unknown_status(self, tmp_path: Path) -> None:
        """Unknown status strings must raise at the boundary — no silent writes."""
        import pytest

        engine, _ = _setup_db(tmp_path)
        record_batch_submission(
            engine,
            request_id="req-validate",
            batch_id="msgbatch_v",
            workflow_id="wf-v",
        )

        with pytest.raises(ValueError):
            update_batch_status(engine, request_id="req-validate", status="not_a_real_status")

    def test_update_with_error_message(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        record_batch_submission(
            engine,
            request_id="req-011",
            batch_id="msgbatch_y",
            workflow_id="wf-2",
        )

        update_batch_status(
            engine,
            request_id="req-011",
            status="failed",
            error_message="Rate limit exceeded",
        )

        job = get_batch_job(engine, "req-011")
        assert job is not None
        assert job["status"] == "failed"
        assert job["error_message"] == "Rate limit exceeded"

    def test_updated_at_changes(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        record_batch_submission(
            engine,
            request_id="req-012",
            batch_id="msgbatch_z",
            workflow_id="wf-3",
        )

        job_before = get_batch_job(engine, "req-012")
        assert job_before is not None

        time.sleep(0.05)
        update_batch_status(engine, request_id="req-012", status="processing")

        job_after = get_batch_job(engine, "req-012")
        assert job_after is not None
        assert job_after["updated_at"] >= job_before["updated_at"]


# ---------------------------------------------------------------------------
# update_batch_status — monotonic guard + error-message preservation (T4.1 ST1)
# ---------------------------------------------------------------------------


class TestUpdateBatchStatusMonotonic:
    def test_submitted_to_ended_applies(self, tmp_path: Path) -> None:
        """SUBMITTED is the sole non-terminal state; the ENDED transition applies."""
        engine, _ = _setup_db(tmp_path)
        record_batch_submission(engine, request_id="req-e", batch_id="b", workflow_id="wf")

        update_batch_status(engine, request_id="req-e", status="ended")

        job = get_batch_job(engine, "req-e")
        assert job is not None
        assert job["status"] == "ended"

    def test_ended_to_failed_is_a_noop(self, tmp_path: Path) -> None:
        """A stale retry against an already-terminal (ENDED) row cannot regress it."""
        engine, _ = _setup_db(tmp_path)
        record_batch_submission(engine, request_id="req-n", batch_id="b", workflow_id="wf")
        update_batch_status(engine, request_id="req-n", status="ended")

        update_batch_status(engine, request_id="req-n", status="failed", error_message="late")

        job = get_batch_job(engine, "req-n")
        assert job is not None
        assert job["status"] == "ended"
        assert job["error_message"] is None

    def test_terminal_status_absorbs_delayed_duplicate(self, tmp_path: Path) -> None:
        """AC interleaving: a row reaches terminal, then a delayed duplicate arrives
        and must not regress the recorded outcome."""
        engine, _ = _setup_db(tmp_path)
        record_batch_submission(engine, request_id="req-d", batch_id="b", workflow_id="wf")

        # First update wins and records the terminal failure with its error.
        update_batch_status(engine, request_id="req-d", status="failed", error_message="boom")
        # A delayed duplicate of an earlier transition arrives after the row is
        # already terminal — it matches zero rows and is silently skipped.
        update_batch_status(engine, request_id="req-d", status="ended")

        job = get_batch_job(engine, "req-d")
        assert job is not None
        assert job["status"] == "failed"
        assert job["error_message"] == "boom"

    def test_error_message_none_does_not_clobber_recorded_error(self, tmp_path: Path) -> None:
        """error_message=None must not overwrite an error already on a matched row."""
        engine, _ = _setup_db(tmp_path)
        record_batch_submission(engine, request_id="req-p", batch_id="b", workflow_id="wf")
        # Seed an error on the still-SUBMITTED row so the next update actually matches
        # the monotonic guard and would clobber the error if it wrote error_message.
        with engine.begin() as conn:
            conn.execute(
                sa.update(BatchJob)
                .where(BatchJob.__table__.c.id == "req-p")
                .values(error_message="recorded earlier")
            )

        update_batch_status(engine, request_id="req-p", status="ended")

        job = get_batch_job(engine, "req-p")
        assert job is not None
        assert job["status"] == "ended"
        assert job["error_message"] == "recorded earlier"


# ---------------------------------------------------------------------------
# get_pending_batch_jobs
# ---------------------------------------------------------------------------


class TestGetPendingBatchJobs:
    def test_returns_only_submitted(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        record_batch_submission(engine, request_id="req-a", batch_id="b1", workflow_id="wf-1")
        record_batch_submission(engine, request_id="req-b", batch_id="b2", workflow_id="wf-2")
        record_batch_submission(engine, request_id="req-c", batch_id="b3", workflow_id="wf-3")

        update_batch_status(engine, request_id="req-b", status="processing")

        pending = get_pending_batch_jobs(engine)
        ids = [j["id"] for j in pending]
        assert "req-a" in ids
        assert "req-c" in ids
        assert "req-b" not in ids

    def test_ordered_by_created_at(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        record_batch_submission(engine, request_id="req-first", batch_id="b1", workflow_id="wf-1")
        record_batch_submission(engine, request_id="req-second", batch_id="b2", workflow_id="wf-2")

        pending = get_pending_batch_jobs(engine)
        assert len(pending) == 2
        assert pending[0]["id"] == "req-first"
        assert pending[1]["id"] == "req-second"


# ---------------------------------------------------------------------------
# get_batch_job
# ---------------------------------------------------------------------------


class TestGetBatchJob:
    def test_returns_by_id(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        record_batch_submission(engine, request_id="req-lookup", batch_id="b1", workflow_id="wf-1")

        job = get_batch_job(engine, "req-lookup")
        assert job is not None
        assert job["id"] == "req-lookup"

    def test_returns_none_for_nonexistent(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        job = get_batch_job(engine, "does-not-exist")
        assert job is None


# ---------------------------------------------------------------------------
# Migration 004
# ---------------------------------------------------------------------------


class TestMigration004:
    def test_creates_table_and_allows_crud(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)

        record_batch_submission(engine, request_id="mig-test", batch_id="b1", workflow_id="wf-1")
        job = get_batch_job(engine, "mig-test")
        assert job is not None
        assert job["status"] == "submitted"
