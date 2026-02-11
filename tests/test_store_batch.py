"""Tests for batch job store functions (Phase 14)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from forge.store import (
    get_batch_job,
    get_engine,
    get_pending_batch_jobs,
    record_batch_submission,
    run_migrations,
    update_batch_status,
)

if TYPE_CHECKING:
    from pathlib import Path


def _setup_db(tmp_path: Path):
    """Create a test database with migrations applied."""
    db_path = tmp_path / "test.db"
    run_migrations(db_path)
    return get_engine(db_path), db_path


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

        update_batch_status(engine, request_id="req-010", status="succeeded")

        job = get_batch_job(engine, "req-010")
        assert job is not None
        assert job["status"] == "succeeded"

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
            status="errored",
            error_message="Rate limit exceeded",
        )

        job = get_batch_job(engine, "req-011")
        assert job is not None
        assert job["status"] == "errored"
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
        update_batch_status(engine, request_id="req-012", status="succeeded")

        job_after = get_batch_job(engine, "req-012")
        assert job_after is not None
        assert job_after["updated_at"] >= job_before["updated_at"]


# ---------------------------------------------------------------------------
# get_pending_batch_jobs
# ---------------------------------------------------------------------------


class TestGetPendingBatchJobs:
    def test_returns_only_submitted(self, tmp_path: Path) -> None:
        engine, _ = _setup_db(tmp_path)
        record_batch_submission(engine, request_id="req-a", batch_id="b1", workflow_id="wf-1")
        record_batch_submission(engine, request_id="req-b", batch_id="b2", workflow_id="wf-2")
        record_batch_submission(engine, request_id="req-c", batch_id="b3", workflow_id="wf-3")

        update_batch_status(engine, request_id="req-b", status="succeeded")

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
