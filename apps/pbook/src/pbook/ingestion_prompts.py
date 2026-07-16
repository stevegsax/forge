"""Prompt construction for Claude Code transcript analysis.

Pure functions that build system and user prompts for the LLM that
analyzes conversation transcripts to identify experiences (unexpected
problems and their resolutions).

Also defines the structured output models used by the LLM response.

No dependencies on forge, Temporal, or LLM providers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class AnalyzedExperience(BaseModel):
    """A single experience identified from a conversation transcript."""

    problem: str = Field(description="What unexpected situation occurred")
    resolution: str = Field(description="How it was resolved")
    context: str = Field(default="", description="Relevant technical context")
    situation: str = Field(
        default="",
        description=(
            "Rich rationale describing the situation that produced this "
            "experience. Should include short verbatim excerpts from the "
            "transcript that show what was happening when the problem "
            "appeared and what led to the resolution. Used later to "
            "reconstruct the original context when discussing the playbook."
        ),
    )


class TranscriptAnalysisResult(BaseModel):
    """Structured output from the transcript analysis LLM call."""

    experiences: list[AnalyzedExperience] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_analysis_system_prompt() -> str:
    """Build the system prompt for analyzing a Claude Code conversation.

    Instructs the LLM to identify situations where the agent encountered
    unexpected problems and found resolutions.
    """
    parts: list[str] = []

    parts.append(
        "You are analyzing a Claude Code conversation transcript to identify "
        "situations where the LLM agent encountered unexpected problems and "
        "found resolutions."
    )
    parts.append("")
    parts.append("## What to look for")
    parts.append("")
    parts.append(
        "Extract experiences where something UNEXPECTED happened — "
        "the first approach was wrong, a workaround was needed, or "
        "behavior was surprising. Each experience must have a clear "
        "problem and a clear resolution."
    )
    parts.append("")
    parts.append("Signals that something is worth extracting:")
    parts.append("- The agent tried an approach, it failed, then tried a different approach")
    parts.append("- An error message revealed non-obvious behavior")
    parts.append("- A library or API behaved differently than expected")
    parts.append("- A workaround was needed for a tool, framework, or platform quirk")
    parts.append("- The user corrected the agent's approach")
    parts.append("- Multiple retries were needed before finding the right solution")
    parts.append("")
    parts.append("## What to skip")
    parts.append("")
    parts.append("Do NOT extract:")
    parts.append("- Routine tasks that succeeded on the first attempt")
    parts.append("- Simple Q&A where the agent answered correctly immediately")
    parts.append("- File exploration or code reading without any surprises")
    parts.append("- Plan-mode planning discussions")
    parts.append("- Generic advice ('use proper error handling', 'write tests')")
    parts.append("- Standard debugging that followed expected patterns")
    parts.append("- Conversations that are too short to contain meaningful experiences")
    parts.append("")
    parts.append(
        "It is better to extract NOTHING than to extract a misleading "
        "or overly generic experience. Quality over quantity."
    )
    parts.append("")
    parts.append("## Output format")
    parts.append("")
    parts.append("For each experience, provide:")
    parts.append(
        "- problem: A specific description of what unexpected situation occurred "
        "(2-4 sentences)"
    )
    parts.append(
        "- resolution: How it was resolved — the specific fix, workaround, "
        "or correct approach (2-4 sentences)"
    )
    parts.append(
        "- context: Relevant technical context — libraries, tools, error messages, "
        "or configuration details that would help someone encountering this "
        "situation (1-3 sentences, optional)"
    )
    parts.append(
        "- situation: A rich rationale that captures what was actually happening "
        "in the conversation when this experience occurred. Include 1-3 short "
        "verbatim excerpts from the transcript (use ASSISTANT: / USER: prefixes "
        "and ellipses for elision) that show the problem appearing and the "
        "resolution being reached. This will be used later to reconstruct the "
        "original context when discussing why a playbook was created."
    )
    parts.append("")
    parts.append("If the conversation contains no extractable experiences, return an empty list.")

    return "\n".join(parts)


def build_analysis_user_prompt(transcript_text: str, project: str) -> str:
    """Build the user prompt containing the rendered transcript.

    Parameters
    ----------
    transcript_text:
        The rendered transcript text from ``render_transcript()``.
    project:
        The project name for context.
    """
    parts: list[str] = []

    parts.append(f"## Project: {project}")
    parts.append("")
    parts.append("## Conversation transcript")
    parts.append("")
    parts.append(transcript_text)
    parts.append("")
    parts.append(
        "Analyze the conversation above. Extract only unexpected problems "
        "and their resolutions. If nothing meets the quality bar, return "
        "an empty list."
    )

    return "\n".join(parts)
