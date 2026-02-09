"""Tests for Phase 7 exploration data models."""

from __future__ import annotations

from forge.models import (
    ContextProviderSpec,
    ContextRequest,
    ContextResult,
    ExplorationInput,
    ExplorationResponse,
    ForgeTaskInput,
    FulfillContextInput,
    TaskDefinition,
)

# ---------------------------------------------------------------------------
# ContextProviderSpec
# ---------------------------------------------------------------------------


class TestContextProviderSpec:
    def test_basic_creation(self) -> None:
        spec = ContextProviderSpec(
            name="read_file",
            description="Read file contents.",
            parameters={"path": "File path."},
        )
        assert spec.name == "read_file"
        assert spec.parameters["path"] == "File path."

    def test_empty_parameters(self) -> None:
        spec = ContextProviderSpec(
            name="repo_map",
            description="Generate repo map.",
            parameters={},
        )
        assert spec.parameters == {}


# ---------------------------------------------------------------------------
# ContextRequest
# ---------------------------------------------------------------------------


class TestContextRequest:
    def test_basic_creation(self) -> None:
        req = ContextRequest(
            provider="read_file",
            params={"path": "foo.py"},
            reasoning="Need to see the implementation.",
        )
        assert req.provider == "read_file"
        assert req.params["path"] == "foo.py"

    def test_default_empty_params(self) -> None:
        req = ContextRequest(
            provider="repo_map",
            reasoning="Need overview.",
        )
        assert req.params == {}


# ---------------------------------------------------------------------------
# ExplorationResponse
# ---------------------------------------------------------------------------


class TestExplorationResponse:
    def test_with_requests(self) -> None:
        resp = ExplorationResponse(
            requests=[
                ContextRequest(
                    provider="read_file",
                    params={"path": "a.py"},
                    reasoning="Need it.",
                ),
            ]
        )
        assert len(resp.requests) == 1

    def test_empty_requests_signals_ready(self) -> None:
        resp = ExplorationResponse(requests=[])
        assert resp.requests == []

    def test_json_roundtrip(self) -> None:
        resp = ExplorationResponse(
            requests=[
                ContextRequest(
                    provider="search_code",
                    params={"pattern": "def test_"},
                    reasoning="Find tests.",
                ),
            ]
        )
        json_str = resp.model_dump_json()
        restored = ExplorationResponse.model_validate_json(json_str)
        assert restored.requests[0].provider == "search_code"


# ---------------------------------------------------------------------------
# ContextResult
# ---------------------------------------------------------------------------


class TestContextResult:
    def test_basic_creation(self) -> None:
        result = ContextResult(
            provider="read_file",
            content="print('hello')",
            estimated_tokens=5,
        )
        assert result.provider == "read_file"
        assert result.estimated_tokens == 5


# ---------------------------------------------------------------------------
# FulfillContextInput
# ---------------------------------------------------------------------------


class TestFulfillContextInput:
    def test_basic_creation(self) -> None:
        inp = FulfillContextInput(
            requests=[
                ContextRequest(
                    provider="read_file",
                    params={"path": "x.py"},
                    reasoning="Need it.",
                ),
            ],
            repo_root="/repo",
            worktree_path="/repo/.forge-worktrees/task-1",
        )
        assert len(inp.requests) == 1
        assert inp.repo_root == "/repo"


# ---------------------------------------------------------------------------
# ExplorationInput
# ---------------------------------------------------------------------------


class TestExplorationInput:
    def test_basic_creation(self) -> None:
        task = TaskDefinition(
            task_id="t1",
            description="Fix bug.",
            target_files=["bug.py"],
        )
        inp = ExplorationInput(
            task=task,
            available_providers=[],
            round_number=1,
            max_rounds=5,
        )
        assert inp.round_number == 1
        assert inp.accumulated_context == []

    def test_with_accumulated_context(self) -> None:
        task = TaskDefinition(
            task_id="t1",
            description="Fix bug.",
        )
        inp = ExplorationInput(
            task=task,
            available_providers=[],
            accumulated_context=[
                ContextResult(provider="read_file", content="data", estimated_tokens=1),
            ],
            round_number=2,
            max_rounds=5,
        )
        assert len(inp.accumulated_context) == 1


# ---------------------------------------------------------------------------
# ForgeTaskInput.max_exploration_rounds
# ---------------------------------------------------------------------------


class TestForgeTaskInputExploration:
    def test_default_exploration_rounds(self) -> None:
        task = TaskDefinition(task_id="t1", description="Test.")
        inp = ForgeTaskInput(task=task, repo_root="/repo")
        assert inp.max_exploration_rounds == 10

    def test_custom_exploration_rounds(self) -> None:
        task = TaskDefinition(task_id="t1", description="Test.")
        inp = ForgeTaskInput(task=task, repo_root="/repo", max_exploration_rounds=0)
        assert inp.max_exploration_rounds == 0

    def test_json_roundtrip_preserves_exploration_rounds(self) -> None:
        task = TaskDefinition(task_id="t1", description="Test.")
        inp = ForgeTaskInput(task=task, repo_root="/repo", max_exploration_rounds=5)
        json_str = inp.model_dump_json()
        restored = ForgeTaskInput.model_validate_json(json_str)
        assert restored.max_exploration_rounds == 5
