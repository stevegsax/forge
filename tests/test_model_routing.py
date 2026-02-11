"""Tests for Phase 11 — Model routing types and functions."""

from __future__ import annotations

from forge.models import (
    AssembledContext,
    CapabilityTier,
    ExplorationInput,
    ExtractionInput,
    ExtractionWorkflowInput,
    ForgeTaskInput,
    ModelConfig,
    PlannerInput,
    PlanStep,
    SubTask,
    SubTaskInput,
    TaskDefinition,
    resolve_model,
)

# ---------------------------------------------------------------------------
# CapabilityTier enum
# ---------------------------------------------------------------------------


class TestCapabilityTier:
    def test_values(self) -> None:
        assert CapabilityTier.REASONING == "reasoning"
        assert CapabilityTier.GENERATION == "generation"
        assert CapabilityTier.SUMMARIZATION == "summarization"
        assert CapabilityTier.CLASSIFICATION == "classification"

    def test_is_str_enum(self) -> None:
        assert isinstance(CapabilityTier.REASONING, str)

    def test_serialization_round_trip(self) -> None:
        tier = CapabilityTier.REASONING
        assert CapabilityTier(str(tier)) == tier


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_defaults_match_spec(self) -> None:
        config = ModelConfig()
        assert config.reasoning == "anthropic:claude-opus-4-6"
        assert config.generation == "anthropic:claude-sonnet-4-5-20250929"
        assert config.summarization == "anthropic:claude-sonnet-4-5-20250929"
        assert config.classification == "anthropic:claude-haiku-4-5-20251001"

    def test_custom_overrides(self) -> None:
        config = ModelConfig(reasoning="custom:model-a", generation="custom:model-b")
        assert config.reasoning == "custom:model-a"
        assert config.generation == "custom:model-b"
        # Others keep defaults
        assert config.summarization == "anthropic:claude-sonnet-4-5-20250929"

    def test_json_round_trip(self) -> None:
        config = ModelConfig(reasoning="custom:fast")
        json_str = config.model_dump_json()
        restored = ModelConfig.model_validate_json(json_str)
        assert restored.reasoning == "custom:fast"
        assert restored == config


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_each_tier_maps_correctly(self) -> None:
        config = ModelConfig()
        assert resolve_model(CapabilityTier.REASONING, config) == config.reasoning
        assert resolve_model(CapabilityTier.GENERATION, config) == config.generation
        assert resolve_model(CapabilityTier.SUMMARIZATION, config) == config.summarization
        assert resolve_model(CapabilityTier.CLASSIFICATION, config) == config.classification

    def test_custom_config_respected(self) -> None:
        config = ModelConfig(reasoning="custom:reasoning-model")
        assert resolve_model(CapabilityTier.REASONING, config) == "custom:reasoning-model"


# ---------------------------------------------------------------------------
# PlanStep.capability_tier
# ---------------------------------------------------------------------------


class TestPlanStepCapabilityTier:
    def test_default_is_none(self) -> None:
        step = PlanStep(step_id="s1", description="desc", target_files=["a.py"])
        assert step.capability_tier is None

    def test_can_set_tier(self) -> None:
        step = PlanStep(
            step_id="s1",
            description="desc",
            target_files=["a.py"],
            capability_tier=CapabilityTier.REASONING,
        )
        assert step.capability_tier == CapabilityTier.REASONING

    def test_backward_compat_from_dict(self) -> None:
        """Old payloads without capability_tier still parse."""
        data = {
            "step_id": "s1",
            "description": "desc",
            "target_files": ["a.py"],
        }
        step = PlanStep.model_validate(data)
        assert step.capability_tier is None


# ---------------------------------------------------------------------------
# ForgeTaskInput.model_routing
# ---------------------------------------------------------------------------


class TestForgeTaskInputModelRouting:
    def _make_task(self) -> TaskDefinition:
        return TaskDefinition(task_id="t1", description="desc")

    def test_default_model_routing(self) -> None:
        inp = ForgeTaskInput(task=self._make_task(), repo_root="/tmp/repo")
        assert inp.model_routing == ModelConfig()

    def test_custom_model_routing(self) -> None:
        config = ModelConfig(reasoning="custom:r")
        inp = ForgeTaskInput(
            task=self._make_task(),
            repo_root="/tmp/repo",
            model_routing=config,
        )
        assert inp.model_routing.reasoning == "custom:r"

    def test_backward_compat_from_dict(self) -> None:
        """Old payloads without model_routing still parse."""
        data = {
            "task": {"task_id": "t1", "description": "desc"},
            "repo_root": "/tmp/repo",
        }
        inp = ForgeTaskInput.model_validate(data)
        assert inp.model_routing == ModelConfig()


# ---------------------------------------------------------------------------
# ExtractionWorkflowInput.model_routing
# ---------------------------------------------------------------------------


class TestExtractionWorkflowInputModelRouting:
    def test_default(self) -> None:
        inp = ExtractionWorkflowInput()
        assert inp.model_routing == ModelConfig()

    def test_custom(self) -> None:
        config = ModelConfig(summarization="custom:s")
        inp = ExtractionWorkflowInput(model_routing=config)
        assert inp.model_routing.summarization == "custom:s"


# ---------------------------------------------------------------------------
# Activity input model_name fields
# ---------------------------------------------------------------------------


class TestActivityInputModelName:
    def test_assembled_context_default(self) -> None:
        ctx = AssembledContext(task_id="t1", system_prompt="sys", user_prompt="usr")
        assert ctx.model_name == ""

    def test_assembled_context_set(self) -> None:
        ctx = AssembledContext(
            task_id="t1", system_prompt="sys", user_prompt="usr", model_name="custom:m"
        )
        assert ctx.model_name == "custom:m"

    def test_planner_input_default(self) -> None:
        inp = PlannerInput(task_id="t1", system_prompt="sys", user_prompt="usr")
        assert inp.model_name == ""

    def test_planner_input_set(self) -> None:
        inp = PlannerInput(
            task_id="t1", system_prompt="sys", user_prompt="usr", model_name="custom:p"
        )
        assert inp.model_name == "custom:p"

    def test_exploration_input_default(self) -> None:
        inp = ExplorationInput(
            task=TaskDefinition(task_id="t1", description="desc"),
            available_providers=[],
            round_number=1,
            max_rounds=5,
        )
        assert inp.model_name == ""

    def test_exploration_input_set(self) -> None:
        inp = ExplorationInput(
            task=TaskDefinition(task_id="t1", description="desc"),
            available_providers=[],
            round_number=1,
            max_rounds=5,
            model_name="custom:e",
        )
        assert inp.model_name == "custom:e"

    def test_extraction_input_default(self) -> None:
        inp = ExtractionInput(system_prompt="sys", user_prompt="usr", source_workflow_ids=[])
        assert inp.model_name == ""

    def test_extraction_input_set(self) -> None:
        inp = ExtractionInput(
            system_prompt="sys",
            user_prompt="usr",
            source_workflow_ids=[],
            model_name="custom:x",
        )
        assert inp.model_name == "custom:x"

    def test_sub_task_input_default(self) -> None:
        inp = SubTaskInput(
            parent_task_id="t1",
            parent_description="desc",
            sub_task=SubTask(sub_task_id="st1", description="sub", target_files=["a.py"]),
            repo_root="/tmp/repo",
            parent_branch="main",
        )
        assert inp.model_name == ""

    def test_sub_task_input_set(self) -> None:
        inp = SubTaskInput(
            parent_task_id="t1",
            parent_description="desc",
            sub_task=SubTask(sub_task_id="st1", description="sub", target_files=["a.py"]),
            repo_root="/tmp/repo",
            parent_branch="main",
            model_name="custom:st",
        )
        assert inp.model_name == "custom:st"
