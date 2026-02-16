"""Action dispatch activities for MCP tools and Agent Skills (Phase 15).

Two activities:
1. execute_mcp_tool — Direct MCP tool invocation (D86: direct dispatch).
2. execute_skill — LLM-mediated skill execution (D86: hybrid dispatch).

Design follows Function Core / Imperative Shell:
- Pure functions: build_skill_execution_prompt
- Imperative shell: execute_mcp_tool, execute_skill
"""

from __future__ import annotations

import logging
import time

from temporalio import activity

from forge.models import (
    ExecuteMCPToolInput,
    ExecuteMCPToolResult,
    ExecuteSkillInput,
    ExecuteSkillResult,
)

logger = logging.getLogger(__name__)

DEFAULT_SKILL_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_SKILL_MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_skill_execution_prompt(
    input: ExecuteSkillInput,
) -> tuple[str, str]:
    """Build system and user prompts for the skill execution LLM call.

    Returns (system_prompt, user_prompt).
    """
    parts: list[str] = []
    parts.append("You are a skill execution assistant.")
    parts.append("")
    parts.append(
        "You have been given a skill with specific instructions. "
        "Follow the instructions carefully to fulfill the user's request."
    )
    parts.append("")
    parts.append(f"## Skill: {input.skill.name}")
    parts.append(f"**Description:** {input.skill.description}")
    parts.append("")
    parts.append("## Skill Instructions")
    parts.append(input.skill.instructions)

    # Load references if available
    from forge.skills.loader import load_skill_references

    references = load_skill_references(input.skill)
    if references:
        parts.append("")
        parts.append("## Reference Materials")
        for ref_name, ref_content in references.items():
            parts.append(f"### {ref_name}")
            # Truncate very long references
            if len(ref_content) > 4000:
                ref_content = ref_content[:4000] + "\n... (truncated)"
            parts.append(ref_content)

    system_prompt = "\n".join(parts)

    user_parts: list[str] = []
    user_parts.append("Execute this skill with the following request:")
    user_parts.append("")
    if input.reasoning:
        user_parts.append(f"**Reasoning:** {input.reasoning}")
        user_parts.append("")
    if input.action_params:
        user_parts.append("**Parameters:**")
        for k, v in input.action_params.items():
            user_parts.append(f"  - {k}: {v}")
        user_parts.append("")
    user_parts.append(
        "Respond with a clear, complete result of executing the skill. "
        "Include any relevant output, data, or findings."
    )

    user_prompt = "\n".join(user_parts)
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


@activity.defn
async def execute_mcp_tool(input: ExecuteMCPToolInput) -> ExecuteMCPToolResult:
    """Activity: invoke an MCP tool directly (D86: direct dispatch).

    Opens a connection to the MCP server, calls the tool, returns
    the result. Connection lifecycle is per-activity (D87).
    """
    from forge.mcp.client import call_tool
    from forge.tracing import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.execute_mcp_tool") as span:
        span.set_attributes(
            {
                "forge.mcp.server_name": input.server_name,
                "forge.mcp.tool_name": input.tool_name,
            }
        )

        content, success = await call_tool(
            server_config=input.server_config,
            tool_name=input.tool_name,
            arguments=input.arguments,
        )

        span.set_attributes(
            {
                "forge.mcp.success": success,
                "forge.mcp.result_length": len(content),
            }
        )

        return ExecuteMCPToolResult(content=content, success=success)


@activity.defn
async def execute_skill(input: ExecuteSkillInput) -> ExecuteSkillResult:
    """Activity: execute an Agent Skill via LLM-mediated subagent (D86: hybrid dispatch).

    Sends the SKILL.md instructions and the action request to an LLM,
    which interprets and executes the skill.
    """
    from pydantic import BaseModel, Field

    from forge.llm_client import build_messages_params, extract_tool_result, get_anthropic_client
    from forge.tracing import get_tracer

    class SkillOutput(BaseModel):
        """Output from a skill execution."""

        result: str = Field(description="The complete result of executing the skill.")
        success: bool = Field(
            description="Whether the skill executed successfully.",
        )

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.execute_skill") as span:
        span.set_attributes(
            {
                "forge.skill.name": input.skill.name,
            }
        )

        system_prompt, user_prompt = build_skill_execution_prompt(input)
        model = input.model_name or DEFAULT_SKILL_MODEL

        client = get_anthropic_client()

        params = build_messages_params(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_type=SkillOutput,
            model=model,
            max_tokens=DEFAULT_SKILL_MAX_TOKENS,
        )

        start = time.monotonic()
        message = await client.messages.create(**params)
        elapsed_ms = (time.monotonic() - start) * 1000

        skill_output = extract_tool_result(message, SkillOutput)

        from forge.llm_client import extract_usage

        input_tokens, output_tokens, _, _ = extract_usage(message)

        span.set_attributes(
            {
                "forge.skill.success": skill_output.success,
                "forge.skill.latency_ms": elapsed_ms,
            }
        )

        return ExecuteSkillResult(
            content=skill_output.result,
            success=skill_output.success,
            model_name=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
