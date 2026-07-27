"""Tests for forge.blocks.dispatch — the arm table and the pure result builders.

No Temporal here: the table is a plain mapping and the batch-lane builders are
pure functions of ``(context, ParsedLLMResponse)``, so they get microsecond unit
tests. The lane fork and the activity sequencing they drive are covered by
tests/test_workflows.py; the committed replay histories pin the batch planner
and both generation lanes command-for-command.
"""

from __future__ import annotations

import pytest

from forge.blocks.dispatch import (
    ARMS,
    call_stats,
    conflict_result,
    exploration_result,
    generation_result,
    plan_result,
    sanity_result,
)
from forge.models import (
    AssembledContext,
    ConflictResolutionResponse,
    ContextRequest,
    ExplorationResponse,
    FileOutput,
    LLMResponse,
    LLMStats,
    ParsedLLMResponse,
    Plan,
    PlanStep,
    SanityCheckResponse,
    SanityCheckVerdict,
    build_llm_stats,
)
from forge.output_types import OUTPUT_TYPES
from forge.presets import THINKING_MAX_TOKENS

# One distinctive parsed response reused by every builder: every field is
# non-default, so a builder that drops one is caught.
_STATS = LLMStats(
    model_name="mock-model",
    input_tokens=1101,
    output_tokens=202,
    latency_ms=303.5,
    cache_creation_input_tokens=44,
    cache_read_input_tokens=55,
    stop_reason="end_turn",
)


def _parsed(payload: str) -> ParsedLLMResponse:
    return ParsedLLMResponse(parsed_json=payload, **_STATS.model_dump())


def _context() -> AssembledContext:
    return AssembledContext(
        task_id="t1",
        system_prompt="system prompt",
        user_prompt="user prompt",
    )


def _plan_json() -> str:
    return Plan(
        task_id="t1",
        steps=[PlanStep(step_id="s1", description="do it", target_files=["a.py"])],
        explanation="one step",
    ).model_dump_json()


def _generation_json() -> str:
    return LLMResponse(
        files=[FileOutput(file_path="a.py", content="x = 1\n")],
        explanation="made a.py",
    ).model_dump_json()


def _sanity_json() -> str:
    return SanityCheckResponse(
        verdict=SanityCheckVerdict.CONTINUE, explanation="looks fine"
    ).model_dump_json()


def _conflict_json() -> str:
    return ConflictResolutionResponse(
        resolved_files=[FileOutput(file_path="a.py", content="merged\n")],
        explanation="merged both",
    ).model_dump_json()


def _exploration_json() -> str:
    return ExplorationResponse(
        requests=[ContextRequest(provider="read_file", reasoning="need it")]
    ).model_dump_json()


# ---------------------------------------------------------------------------
# The arm table
# ---------------------------------------------------------------------------


class TestArmTable:
    def test_exactly_five_arms(self) -> None:
        assert set(ARMS) == {
            "generation",
            "planner",
            "sanity_check",
            "conflict_resolution",
            "exploration",
        }

    @pytest.mark.parametrize(
        ("arm_name", "role", "sync_activity", "output_type_name"),
        [
            ("generation", "llm", "call_llm", "LLMResponse"),
            ("planner", "planner", "call_planner", "Plan"),
            ("sanity_check", "sanity_check", "call_sanity_check", "SanityCheckResponse"),
            (
                "conflict_resolution",
                "conflict_resolution",
                "call_conflict_resolution",
                "ConflictResolutionResponse",
            ),
            ("exploration", "exploration", "call_exploration_llm", "ExplorationResponse"),
        ],
    )
    def test_row(self, arm_name: str, role: str, sync_activity: str, output_type_name: str) -> None:
        arm = ARMS[arm_name]  # type: ignore[index]
        assert (arm.role, arm.sync_activity, arm.output_type_name) == (
            role,
            sync_activity,
            output_type_name,
        )

    def test_roles_are_distinct(self) -> None:
        """Roles key the per-role occurrence counter; a duplicate would make two
        arms collide on the interaction idempotency key (T1.6a)."""
        roles = [arm.role for arm in ARMS.values()]
        assert len(set(roles)) == len(roles)

    def test_thinking_enabled_arms_carry_the_raised_cap(self) -> None:
        """Exactly the three thinking arms get the adaptive-thinking cap; the two
        thinking-disabled arms keep the transport's own 4096 default."""
        raised = {name for name, arm in ARMS.items() if arm.max_tokens == THINKING_MAX_TOKENS}
        assert raised == {"planner", "sanity_check", "conflict_resolution"}
        assert {ARMS["generation"].max_tokens, ARMS["exploration"].max_tokens} == {4096}

    def test_every_output_type_is_resolvable(self) -> None:
        """A batch arm naming a type the registry doesn't hold would fail at
        submit time, in production, after the run had already paid for planning."""
        assert all(arm.output_type_name in OUTPUT_TYPES for arm in ARMS.values())


# ---------------------------------------------------------------------------
# The shared stats mapping
# ---------------------------------------------------------------------------


class TestCallStats:
    def test_maps_every_llm_stats_field(self) -> None:
        assert call_stats(_parsed("{}")) == _STATS.model_dump()

    @pytest.mark.parametrize(
        ("builder", "payload"),
        [
            (lambda parsed: generation_result(_context(), parsed), _generation_json()),
            (lambda parsed: plan_result("t1", parsed), _plan_json()),
            (lambda parsed: sanity_result("t1", parsed), _sanity_json()),
            (lambda parsed: conflict_result("t1", parsed), _conflict_json()),
            (lambda parsed: exploration_result(_context(), parsed), _exploration_json()),
        ],
        ids=["generation", "planner", "sanity_check", "conflict_resolution", "exploration"],
    )
    def test_every_arm_stamps_identical_stats(self, builder: object, payload: str) -> None:
        """One helper, five arms: an arm that dropped stop_reason or the cache
        tokens would under-report spend in the interactions store."""
        assert callable(builder)
        result = builder(_parsed(payload))
        assert build_llm_stats(result) == _STATS


# ---------------------------------------------------------------------------
# The typed builders
# ---------------------------------------------------------------------------


class TestResultBuilders:
    def test_generation_takes_its_task_id_from_the_context(self) -> None:
        result = generation_result(_context(), _parsed(_generation_json()))
        assert result.task_id == "t1"
        assert result.response.explanation == "made a.py"
        assert [f.file_path for f in result.response.files] == ["a.py"]

    def test_plan_parses_the_plan(self) -> None:
        result = plan_result("t1", _parsed(_plan_json()))
        assert result.task_id == "t1"
        assert [step.step_id for step in result.plan.steps] == ["s1"]

    def test_sanity_parses_the_verdict(self) -> None:
        result = sanity_result("t1", _parsed(_sanity_json()))
        assert result.response.verdict == SanityCheckVerdict.CONTINUE

    def test_conflict_maps_resolved_files_to_a_path_keyed_dict(self) -> None:
        result = conflict_result("t1", _parsed(_conflict_json()))
        assert result.resolved_files == {"a.py": "merged\n"}
        assert result.explanation == "merged both"

    def test_exploration_carries_the_assembled_prompts(self) -> None:
        """The batch lane's envelope is the same shape the sync activity returns,
        so the persist reads the prompts the same way on both lanes."""
        result = exploration_result(_context(), _parsed(_exploration_json()))
        assert result.task_id == "t1"
        assert (result.system_prompt, result.user_prompt) == ("system prompt", "user prompt")
        assert [req.provider for req in result.response.requests] == ["read_file"]
