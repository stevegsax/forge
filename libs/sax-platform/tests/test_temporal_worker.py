"""Tests for the pure Worker-construction helpers in ``sax_platform.temporal.worker``.

Only ``build_sandbox_runner`` and ``worker_kwargs`` are exercised — both are
pure and require no live client, event loop, or Temporal connection.
``run_worker`` itself (which drives a real ``Worker.run()``/signal-handled
drain loop) is intentionally not exercised here; see the module docstring in
``worker.py`` for the ported drain-loop design.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner

from sax_platform.temporal.worker import (
    DEFAULT_GRACEFUL_SHUTDOWN,
    DEFAULT_MAX_CONCURRENT_ACTIVITIES,
    build_sandbox_runner,
    worker_kwargs,
)


class TestDefaults:
    def test_graceful_shutdown_default_is_five_minutes(self) -> None:
        """AC fix: forge's prior hardcoded 30s canceled in-flight LLM calls
        (up to 5-minute start_to_close) on every deploy."""
        assert timedelta(minutes=5) == DEFAULT_GRACEFUL_SHUTDOWN
        assert DEFAULT_GRACEFUL_SHUTDOWN.total_seconds() == 300

    def test_max_concurrent_activities_default_is_bounded(self) -> None:
        # Explicit, modest, not unbounded (activities spawn subprocesses).
        assert isinstance(DEFAULT_MAX_CONCURRENT_ACTIVITIES, int)
        assert 0 < DEFAULT_MAX_CONCURRENT_ACTIVITIES <= 32
        assert DEFAULT_MAX_CONCURRENT_ACTIVITIES == 8


class TestBuildSandboxRunner:
    def test_returns_sandboxed_workflow_runner(self) -> None:
        runner = build_sandbox_runner()
        assert isinstance(runner, SandboxedWorkflowRunner)

    def test_default_passthrough_modules_are_pydantic_and_pydantic_core(self) -> None:
        runner = build_sandbox_runner()
        passthrough = runner.restrictions.passthrough_modules
        assert "pydantic" in passthrough
        assert "pydantic_core" in passthrough

    def test_custom_passthrough_modules_are_honored(self) -> None:
        runner = build_sandbox_runner(passthrough_modules=("pydantic", "pydantic_core", "numpy"))
        assert "numpy" in runner.restrictions.passthrough_modules


class TestWorkerKwargs:
    def test_assembles_expected_keys_and_values(self) -> None:
        runner = build_sandbox_runner()

        class _Workflow:
            pass

        async def _activity() -> None:
            pass

        kwargs = worker_kwargs(
            task_queue="my-task-queue",
            workflows=[_Workflow],
            activities=[_activity],
            workflow_runner=runner,
            max_concurrent_activities=4,
            graceful_shutdown_timeout=timedelta(minutes=2),
        )

        assert kwargs["task_queue"] == "my-task-queue"
        assert kwargs["workflows"] == [_Workflow]
        assert kwargs["activities"] == [_activity]
        assert kwargs["workflow_runner"] is runner
        assert kwargs["max_concurrent_activities"] == 4
        assert kwargs["graceful_shutdown_timeout"] == timedelta(minutes=2)
        assert set(kwargs) == {
            "task_queue",
            "workflows",
            "activities",
            "workflow_runner",
            "max_concurrent_activities",
            "graceful_shutdown_timeout",
        }
