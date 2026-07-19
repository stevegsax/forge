"""Tests for forge.store — observability store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import forge.store as _store
from forge.models import (
    TaskResult,
    TransitionSignal,
)
from forge.store import (
    InteractionRow,
    build_playbook_dict,
    get_interactions,
    get_playbooks_by_tags,
    get_run,
    get_store_engine,
    get_unextracted_runs,
    list_recent_playbooks,
    list_recent_runs,
    run_migrations,
    save_interaction,
    save_playbooks,
    save_run,
    tags_overlap,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy import Engine


def _migrate(db_path: Path) -> Engine:
    """Test helper: migrate a throwaway SQLite store and open an engine.

    Uses ``forge.store``'s ``create_engine`` directly against the throwaway URL.
    """
    url = f"sqlite:///{db_path}"
    run_migrations(url)
    return _store.sa.create_engine(url)


# ---------------------------------------------------------------------------
# get_store_engine (re-exported from sax_platform.db; url is required config)
# ---------------------------------------------------------------------------


class TestGetStoreEngine:
    def test_sqlite_url_enables_wal(self, tmp_path: Path) -> None:
        import sqlalchemy as sa

        db_path = tmp_path / "wal.db"
        engine = get_store_engine(f"sqlite:///{db_path}")
        assert engine.dialect.name == "sqlite"
        with engine.connect() as conn:
            mode = conn.execute(sa.text("PRAGMA journal_mode")).scalar()
        assert mode == "wal"

    def test_postgres_url_builds_pooled_engine_without_wal(self) -> None:
        # Building the engine imports the driver but never connects, so no
        # Postgres server is needed — only the psycopg2 DBAPI module.
        import pytest as _pytest

        _pytest.importorskip("psycopg2")
        engine = get_store_engine("postgresql+psycopg2://user:pw@localhost:5432/forge_test")
        assert engine.dialect.name == "postgresql"


# ---------------------------------------------------------------------------
# interaction-row factory (shared by the roundtrip/migration tests)
# ---------------------------------------------------------------------------


def _interaction_row(**overrides: object) -> InteractionRow:
    """Build an interactions-table row with defaults, overridable per field."""
    fields: dict[str, object] = {
        "task_id": "test-task",
        "role": "llm",
        "system_prompt": "You are a code generator.",
        "user_prompt": "Generate code.",
        "model_name": "test-model",
        "input_tokens": 100,
        "output_tokens": 50,
        "latency_ms": 250.0,
        "explanation": "Created file.",
    }
    fields.update(overrides)
    return InteractionRow(**fields)


# ---------------------------------------------------------------------------
# save_interaction / get_interactions roundtrip
# ---------------------------------------------------------------------------


class TestInteractionRoundtrip:
    def test_save_and_get(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        save_interaction(engine, _interaction_row(step_id="step-1"))

        rows = get_interactions(engine, "test-task")
        assert len(rows) == 1
        assert rows[0]["task_id"] == "test-task"
        assert rows[0]["step_id"] == "step-1"
        assert rows[0]["model_name"] == "test-model"

    def test_filter_by_step(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        for step in ["step-1", "step-2"]:
            save_interaction(engine, _interaction_row(step_id=step))

        rows = get_interactions(engine, "test-task", step_id="step-1")
        assert len(rows) == 1
        assert rows[0]["step_id"] == "step-1"

    def test_empty_result(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        rows = get_interactions(engine, "nonexistent")
        assert rows == []


# ---------------------------------------------------------------------------
# save_run / get_run / list_recent_runs roundtrip
# ---------------------------------------------------------------------------


class TestRunRoundtrip:
    def test_save_and_get(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        task_result = TaskResult(task_id="t1", status=TransitionSignal.SUCCESS)
        save_run(engine, task_result, "wf-123", "run-1")

        run_data = get_run(engine, "wf-123")
        assert run_data is not None
        assert run_data["task_id"] == "t1"
        assert run_data["workflow_id"] == "wf-123"
        assert run_data["run_id"] == "run-1"
        assert run_data["status"] == "success"
        assert run_data["result"]["task_id"] == "t1"

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        assert get_run(engine, "nonexistent") is None

    def test_rerun_records_distinct_run(self, tmp_path: Path) -> None:
        """A rerun (same workflow_id, fresh run_id) is a new row, not swallowed."""
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        task_result = TaskResult(task_id="t1", status=TransitionSignal.SUCCESS)
        assert save_run(engine, task_result, "wf-123", "run-A") is True
        assert save_run(engine, task_result, "wf-123", "run-A") is False  # dup absorbed
        assert save_run(engine, task_result, "wf-123", "run-B") is True  # rerun recorded

        runs = list_recent_runs(engine)
        assert {r["run_id"] for r in runs if r["workflow_id"] == "wf-123"} == {"run-A", "run-B"}

    def test_list_recent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        for i in range(3):
            result = TaskResult(task_id=f"t{i}", status=TransitionSignal.SUCCESS)
            save_run(engine, result, f"wf-{i}", f"run-{i}")

        runs = list_recent_runs(engine, limit=2)
        assert len(runs) == 2

    def test_list_empty(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        runs = list_recent_runs(engine)
        assert runs == []


# ---------------------------------------------------------------------------
# run_migrations creates tables
# ---------------------------------------------------------------------------


class TestRunMigrations:
    def test_creates_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "fresh.db"
        engine = _migrate(db_path)

        # Should be able to insert and query
        save_interaction(engine, _interaction_row(task_id="t"))
        rows = get_interactions(engine, "t")
        assert len(rows) == 1

    def test_idempotent(self, tmp_path: Path) -> None:
        url = f"sqlite:///{tmp_path / 'fresh.db'}"
        run_migrations(url)
        run_migrations(url)  # Should not raise


# ---------------------------------------------------------------------------
# build_playbook_dict
# ---------------------------------------------------------------------------


class TestBuildPlaybookDict:
    def test_basic(self) -> None:
        from forge.models import PlaybookEntry

        entry = PlaybookEntry(
            title="Test lesson",
            content="Always do X.",
            tags=["python", "test-writing"],
            source_task_id="t1",
            source_workflow_id="wf-1",
        )
        result = build_playbook_dict(entry, "extract-wf-1")
        assert result["title"] == "Test lesson"
        assert result["content"] == "Always do X."
        assert result["tags_json"] == '["python", "test-writing"]'
        assert result["source_task_id"] == "t1"
        assert result["source_workflow_id"] == "wf-1"
        assert result["extraction_workflow_id"] == "extract-wf-1"


# ---------------------------------------------------------------------------
# Playbook roundtrip tests (Phase 6)
# ---------------------------------------------------------------------------


class TestPlaybookRoundtrip:
    def _insert_playbook(
        self,
        engine: object,
        *,
        title: str = "Lesson",
        tags: list[str] | None = None,
        source_task_id: str = "t1",
        source_workflow_id: str = "wf-1",
    ) -> None:
        import json

        if tags is None:
            tags = ["python"]
        save_playbooks(
            engine,
            [
                {
                    "title": title,
                    "content": "Content.",
                    "tags_json": json.dumps(tags),
                    "source_task_id": source_task_id,
                    "source_workflow_id": source_workflow_id,
                    "extraction_workflow_id": "extract-1",
                }
            ],
        )

    def test_save_and_get_by_tags(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        self._insert_playbook(engine, tags=["python", "api"])
        self._insert_playbook(engine, title="JS lesson", tags=["javascript"])

        results = get_playbooks_by_tags(engine, ["python"], limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "Lesson"

    def test_get_by_tags_multiple_match(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        self._insert_playbook(engine, title="A", tags=["python"])
        self._insert_playbook(engine, title="B", tags=["python", "api"])

        results = get_playbooks_by_tags(engine, ["python"], limit=10)
        assert len(results) == 2

    def test_get_by_tags_no_match(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        self._insert_playbook(engine, tags=["python"])
        results = get_playbooks_by_tags(engine, ["rust"], limit=10)
        assert results == []

    def test_get_by_tags_empty_input(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        results = get_playbooks_by_tags(engine, [], limit=10)
        assert results == []

    def test_get_by_tags_respects_limit(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        self._insert_playbook(engine, title="A", tags=["python"])
        self._insert_playbook(engine, title="B", tags=["python"])
        self._insert_playbook(engine, title="C", tags=["python"])

        results = get_playbooks_by_tags(engine, ["python"], limit=2)
        assert len(results) == 2

    def test_list_recent_playbooks(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        self._insert_playbook(engine, title="A")
        self._insert_playbook(engine, title="B")
        self._insert_playbook(engine, title="C")

        results = list_recent_playbooks(engine, limit=2)
        assert len(results) == 2

    def test_list_recent_playbooks_empty(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        results = list_recent_playbooks(engine)
        assert results == []


class TestTagsOverlap:
    """Pure tag-matching helper (dialect-free core of the playbook queries)."""

    def test_overlap_matches(self) -> None:
        assert tags_overlap('["python", "api"]', {"api"}) is True

    def test_no_overlap(self) -> None:
        assert tags_overlap('["python"]', {"rust"}) is False

    def test_malformed_json_never_matches(self) -> None:
        assert tags_overlap("{not json", {"python"}) is False

    def test_non_list_payload_never_matches(self) -> None:
        assert tags_overlap('{"tag": "python"}', {"python"}) is False


# ---------------------------------------------------------------------------
# get_unextracted_runs (Phase 6)
# ---------------------------------------------------------------------------


class TestGetUnextractedRuns:
    def test_returns_runs_without_playbooks(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        # Add two runs
        for i in range(2):
            save_run(
                engine,
                TaskResult(task_id=f"t{i}", status=TransitionSignal.SUCCESS),
                f"wf-{i}",
                f"run-{i}",
            )

        runs = get_unextracted_runs(engine, limit=50)
        assert len(runs) == 2

    def test_excludes_extracted_runs(self, tmp_path: Path) -> None:
        import json

        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        # Add two runs
        save_run(
            engine,
            TaskResult(task_id="t1", status=TransitionSignal.SUCCESS),
            "wf-1",
            "run-1",
        )
        save_run(
            engine,
            TaskResult(task_id="t2", status=TransitionSignal.SUCCESS),
            "wf-2",
            "run-2",
        )

        # Mark wf-1 as extracted
        save_playbooks(
            engine,
            [
                {
                    "title": "Lesson",
                    "content": "Content.",
                    "tags_json": json.dumps(["python"]),
                    "source_task_id": "t1",
                    "source_workflow_id": "wf-1",
                    "extraction_workflow_id": "extract-1",
                }
            ],
        )

        runs = get_unextracted_runs(engine, limit=50)
        assert len(runs) == 1
        assert runs[0]["workflow_id"] == "wf-2"

    def test_empty_when_all_extracted(self, tmp_path: Path) -> None:
        import json

        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        save_run(
            engine,
            TaskResult(task_id="t1", status=TransitionSignal.SUCCESS),
            "wf-1",
            "run-1",
        )

        save_playbooks(
            engine,
            [
                {
                    "title": "Lesson",
                    "content": "Content.",
                    "tags_json": json.dumps(["python"]),
                    "source_task_id": "t1",
                    "source_workflow_id": "wf-1",
                    "extraction_workflow_id": "extract-1",
                }
            ],
        )

        runs = get_unextracted_runs(engine, limit=50)
        assert runs == []


# ---------------------------------------------------------------------------
# 002 migration
# ---------------------------------------------------------------------------


class TestMigration002:
    def test_creates_playbooks_table(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        import json

        # Should be able to insert into playbooks
        save_playbooks(
            engine,
            [
                {
                    "title": "Lesson",
                    "content": "Content.",
                    "tags_json": json.dumps(["python"]),
                    "source_task_id": "t1",
                    "source_workflow_id": "wf-1",
                    "extraction_workflow_id": "extract-1",
                }
            ],
        )
        results = list_recent_playbooks(engine, limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "Lesson"


# ---------------------------------------------------------------------------
# 003 migration — cache token columns
# ---------------------------------------------------------------------------


class TestMigration003:
    def test_cache_columns_exist(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        save_interaction(
            engine,
            _interaction_row(cache_creation_input_tokens=500, cache_read_input_tokens=1000),
        )

        rows = get_interactions(engine, "test-task")
        assert len(rows) == 1
        assert rows[0]["cache_creation_input_tokens"] == 500
        assert rows[0]["cache_read_input_tokens"] == 1000

    def test_cache_columns_default_zero(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        save_interaction(engine, _interaction_row())

        rows = get_interactions(engine, "test-task")
        assert len(rows) == 1
        assert rows[0]["cache_creation_input_tokens"] == 0
        assert rows[0]["cache_read_input_tokens"] == 0


# ---------------------------------------------------------------------------
# Alembic revision 003 — nullable stop_reason column
# ---------------------------------------------------------------------------


class TestStopReasonColumn:
    def test_column_exists_and_nullable_by_default(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        save_interaction(engine, _interaction_row())

        rows = get_interactions(engine, "test-task")
        assert len(rows) == 1
        assert rows[0]["stop_reason"] is None

    def test_stop_reason_round_trips(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        engine = _migrate(db_path)

        save_interaction(engine, _interaction_row(stop_reason="max_tokens"))

        rows = get_interactions(engine, "test-task")
        assert rows[0]["stop_reason"] == "max_tokens"
