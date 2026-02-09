"""Layer 2: LLM-as-judge for plan quality assessment.

Follows Function Core / Imperative Shell:
- Pure functions: build_judge_system_prompt, build_judge_user_prompt
- Testable function: execute_judge_call (takes injected agent)
- Imperative shell: create_judge_agent, judge_plan
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from forge.eval.models import EvalCase, JudgeCriterion, JudgeVerdict

if TYPE_CHECKING:
    from pydantic_ai import Agent

    from forge.models import Plan

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "anthropic:claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_judge_system_prompt(
    case: EvalCase,
    plan: Plan,
    repo_context: str | None = None,
) -> str:
    """Build the system prompt for the LLM judge.

    Includes the task description, the plan to evaluate, and scoring criteria.
    """
    parts: list[str] = []

    parts.append("You are a plan quality evaluator for an LLM task orchestrator.")
    parts.append("")
    parts.append("## Task")
    parts.append(f"Task ID: {case.task.task_id}")
    parts.append(f"Description: {case.task.description}")

    if case.task.target_files:
        parts.append("")
        parts.append("### Target Files")
        for f in case.task.target_files:
            parts.append(f"- {f}")

    if case.task.context_files:
        parts.append("")
        parts.append("### Context Files")
        for f in case.task.context_files:
            parts.append(f"- {f}")

    if repo_context:
        parts.append("")
        parts.append("## Repository Context")
        parts.append(repo_context)

    parts.append("")
    parts.append("## Plan to Evaluate")
    parts.append(f"Explanation: {plan.explanation}")
    parts.append(f"Steps: {len(plan.steps)}")
    for step in plan.steps:
        parts.append("")
        parts.append(f"### {step.step_id}")
        parts.append(f"Description: {step.description}")
        parts.append(f"Target files: {', '.join(step.target_files) or '(none)'}")
        if step.context_files:
            parts.append(f"Context files: {', '.join(step.context_files)}")
        if step.sub_tasks:
            parts.append(f"Sub-tasks: {len(step.sub_tasks)}")
            for st in step.sub_tasks:
                parts.append(f"  - {st.sub_task_id}: {st.description}")
                parts.append(f"    targets: {', '.join(st.target_files)}")

    parts.append("")
    parts.append("## Scoring Criteria")
    parts.append("Score each criterion on a 1-5 scale:")
    parts.append("")

    criteria_descriptions = {
        JudgeCriterion.COMPLETENESS: (
            "Does the plan cover all required target files and task requirements?"
        ),
        JudgeCriterion.GRANULARITY: (
            "Are steps appropriately sized — not too coarse (everything in one step) "
            "or too fine (trivial steps)?"
        ),
        JudgeCriterion.ORDERING: (
            "Are steps in a logical order where each step can build on prior steps?"
        ),
        JudgeCriterion.CONTEXT_QUALITY: (
            "Do steps reference appropriate context files for the work they need to do?"
        ),
        JudgeCriterion.FAN_OUT_APPROPRIATENESS: (
            "If fan-out is used, is it for genuinely independent work? "
            "If not used, would it have been appropriate?"
        ),
        JudgeCriterion.EXPLANATION_QUALITY: (
            "Does the plan explanation clearly describe the decomposition strategy?"
        ),
    }

    for criterion, description in criteria_descriptions.items():
        parts.append(f"- **{criterion.value}**: {description}")

    return "\n".join(parts)


def build_judge_user_prompt() -> str:
    """Build the user prompt for the judge call."""
    return (
        "Evaluate the plan above. Score each criterion from 1 (poor) to 5 (excellent) "
        "and provide a brief rationale for each score. End with an overall assessment."
    )


# ---------------------------------------------------------------------------
# Testable function
# ---------------------------------------------------------------------------


async def execute_judge_call(
    system_prompt: str,
    user_prompt: str,
    agent: Agent[None, JudgeVerdict],
) -> JudgeVerdict:
    """Call the judge agent and return the verdict.

    Separated from the imperative shell so tests can inject a mock agent.
    """
    start = time.monotonic()

    result = await agent.run(
        user_prompt,
        instructions=system_prompt,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    usage = result.usage()
    logger.info(
        "Judge call completed in %.0fms (input=%d, output=%d)",
        elapsed_ms,
        usage.input_tokens or 0,
        usage.output_tokens or 0,
    )

    return result.output


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def create_judge_agent(model_name: str | None = None) -> Agent[None, JudgeVerdict]:
    """Create a pydantic-ai Agent configured for plan evaluation."""
    from pydantic_ai import Agent

    if model_name is None:
        model_name = DEFAULT_JUDGE_MODEL

    return Agent(
        model_name,
        output_type=JudgeVerdict,
    )


async def judge_plan(
    case: EvalCase,
    plan: Plan,
    *,
    repo_context: str | None = None,
    model_name: str | None = None,
) -> JudgeVerdict:
    """Evaluate a plan using the LLM judge.

    This is the imperative shell entry point — creates the agent and
    delegates to pure/testable functions.
    """
    system_prompt = build_judge_system_prompt(case, plan, repo_context)
    user_prompt = build_judge_user_prompt()
    agent = create_judge_agent(model_name)
    return await execute_judge_call(system_prompt, user_prompt, agent)
