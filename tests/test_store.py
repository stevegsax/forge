"""Tests for forge.store — observability store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge.models import (
    AssembledContext,
    ContextStats,
    FileOutput,
    LLMCallResult,
    LLMResponse,
    PlanCallResult,
    TaskResult,
    TransitionSignal,
)
from forge.store import (
    build_interaction_dict,
    get_db_path,
    get_engine,
    get_interactions,
    get_run,
    list_recent_runs,
    run_migrations,
    save_interaction,
    save_run,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


# ---------------------------------------------------------------------------
# get_db_path
# ---------------------------------------------------------------------------


class TestGetDbPath:
    def test_default_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FORGE_DB_PATH", raising=False)
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        result = get_db_path()
        assert result is not None
        assert str(result).endswith(".local/state/forge/forge.db")

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        db_file = tmp_path / "custom.db"
        monkeypatch.setenv("FORGE_DB_PATH", str(db_file))
        result = get_db_path()
        assert result == db_file

    def test_empty_string_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_DB_PATH", "")
        result = get_db_path()
        assert result is None

    def test_xdg_state_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("FORGE_DB_PATH", raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        result = get_db_path()
        assert result == tmp_path / "forge" / "forge.db"


# ---------------------------------------------------------------------------
# build_interaction_dict
# ---------------------------------------------------------------------------


def _make_context(
    *,
    step_id: str | None = None,
    sub_task_id: str | None = None,
) -> AssembledContext:
    return AssembledContext(
        task_id="test-task",
        system_prompt="You are a code generator.",
        user_prompt="Generate code.",
        step_id=step_id,
        sub_task_id=sub_task_id,
    )


def _make_llm_result() -> LLMCallResult:
    return LLMCallResult(
        task_id="test-task",
        response=LLMResponse(
            files=[FileOutput(file_path="a.py", content="pass")],
            explanation="Created file.",
        ),
        model_name="test-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=250.0,
    )


class TestBuildInteractionDict:
    def test_basic(self) -> None:
        context = _make_context()
        result = _make_llm_result()
        data = build_interaction_dict(
            task_id="test-task",
            step_id=None,
            sub_task_id=None,
            role="llm",
            context=context,
            llm_result=result,
        )
        assert data["task_id"] == "test-task"
        assert data["role"] == "llm"
        assert data["model_name"] == "test-model"
        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50
        assert data["latency_ms"] == 250.0
        assert data["explanation"] == "Created file."
        assert data["system_prompt"] == "You are a code generator."
        assert data["user_prompt"] == "Generate code."
        assert data["context_stats_json"] is None

    def test_with_step_id(self) -> None:
        context = _make_context(step_id="step-1")
        result = _make_llm_result()
        data = build_interaction_dict(
            task_id="test-task",
            step_id="step-1",
            sub_task_id=None,
            role="llm",
            context=context,
            llm_result=result,
        )
        assert data["step_id"] == "step-1"

    def test_with_context_stats(self) -> None:
        context = AssembledContext(
            task_id="test-task",
            system_prompt="sys",
            user_prompt="usr",
            context_stats=ContextStats(files_discovered=5),
        )
        result = _make_llm_result()
        data = build_interaction_dict(
            task_id="test-task",
            step_id=None,
            sub_task_id=None,
            role="llm",
            context=context,
            llm_result=result,
        )
        assert data["context_stats_json"] is not None
        assert "files_discovered" in data["context_stats_json"]

    def test_with_planner_result(self) -> None:
        from forge.models import Plan, PlanStep

        context = _make_context()
        plan = Plan(
            task_id="t",
            steps=[PlanStep(step_id="s1", description="d", target_files=["a.py"])],
            explanation="Plan explanation.",
        )
        planner_result = PlanCallResult(
            task_id="t",
            plan=plan,
            model_name="planner-model",
            input_tokens=200,
            output_tokens=100,
            latency_ms=500.0,
        )
        data = build_interaction_dict(
            task_id="t",
            step_id=None,
            sub_task_id=None,
            role="planner",
            context=context,
            llm_result=planner_result,
        )
        assert data["model_name"] == "planner-model"
        assert data["explanation"] == "Plan explanation."


# ---------------------------------------------------------------------------
# save_interaction / get_interactions roundtrip
# ---------------------------------------------------------------------------


class TestInteractionRoundtrip:
    def test_save_and_get(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        context = _make_context(step_id="step-1")
        result = _make_llm_result()
        data = build_interaction_dict(
            task_id="test-task",
            step_id="step-1",
            sub_task_id=None,
            role="llm",
            context=context,
            llm_result=result,
        )
        save_interaction(engine, **data)

        rows = get_interactions(engine, "test-task")
        assert len(rows) == 1
        assert rows[0]["task_id"] == "test-task"
        assert rows[0]["step_id"] == "step-1"
        assert rows[0]["model_name"] == "test-model"

    def test_filter_by_step(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        for step in ["step-1", "step-2"]:
            context = _make_context(step_id=step)
            result = _make_llm_result()
            data = build_interaction_dict(
                task_id="test-task",
                step_id=step,
                sub_task_id=None,
                role="llm",
                context=context,
                llm_result=result,
            )
            save_interaction(engine, **data)

        rows = get_interactions(engine, "test-task", step_id="step-1")
        assert len(rows) == 1
        assert rows[0]["step_id"] == "step-1"

    def test_empty_result(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        rows = get_interactions(engine, "nonexistent")
        assert rows == []


# ---------------------------------------------------------------------------
# save_run / get_run / list_recent_runs roundtrip
# ---------------------------------------------------------------------------


class TestRunRoundtrip:
    def test_save_and_get(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        task_result = TaskResult(task_id="t1", status=TransitionSignal.SUCCESS)
        save_run(engine, task_result, "wf-123")

        run_data = get_run(engine, "wf-123")
        assert run_data is not None
        assert run_data["task_id"] == "t1"
        assert run_data["workflow_id"] == "wf-123"
        assert run_data["status"] == "success"
        assert run_data["result"]["task_id"] == "t1"

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        assert get_run(engine, "nonexistent") is None

    def test_list_recent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        for i in range(3):
            result = TaskResult(task_id=f"t{i}", status=TransitionSignal.SUCCESS)
            save_run(engine, result, f"wf-{i}")

        runs = list_recent_runs(engine, limit=2)
        assert len(runs) == 2

    def test_list_empty(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        runs = list_recent_runs(engine)
        assert runs == []


# ---------------------------------------------------------------------------
# Store disabled
# ---------------------------------------------------------------------------


class TestStoreDisabled:
    def test_get_db_path_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORGE_DB_PATH", "")
        assert get_db_path() is None


# ---------------------------------------------------------------------------
# run_migrations creates tables
# ---------------------------------------------------------------------------


class TestRunMigrations:
    def test_creates_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "fresh.db"
        run_migrations(db_path)
        engine = get_engine(db_path)

        # Should be able to insert and query
        context = _make_context()
        result = _make_llm_result()
        data = build_interaction_dict(
            task_id="t",
            step_id=None,
            sub_task_id=None,
            role="llm",
            context=context,
            llm_result=result,
        )
        save_interaction(engine, **data)
        rows = get_interactions(engine, "t")
        assert len(rows) == 1

    def test_idempotent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "fresh.db"
        run_migrations(db_path)
        run_migrations(db_path)  # Should not raise
