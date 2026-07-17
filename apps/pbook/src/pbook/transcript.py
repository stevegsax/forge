"""Claude Code transcript parsing, filtering, and session discovery.

Pure functions for reading JSONL session files, reducing them to compact
textual summaries suitable for LLM analysis, and discovering sessions
across all projects.

No dependencies on forge, Temporal, or LLM providers.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TranscriptMessage(BaseModel):
    """A single meaningful message extracted from a JSONL transcript."""

    role: str  # "user" | "assistant"
    text: str
    tool_name: str = ""
    is_error: bool = False


class TranscriptMeta(BaseModel):
    """Metadata extracted from a transcript session."""

    session_id: str = ""
    project_dir: str = ""
    project_name: str = ""
    git_branch: str = ""


class ParsedTranscript(BaseModel):
    """A parsed and filtered Claude Code conversation transcript."""

    meta: TranscriptMeta = Field(default_factory=TranscriptMeta)
    messages: list[TranscriptMessage] = Field(default_factory=list)


class SessionInfo(BaseModel):
    """Metadata about a discovered Claude Code session file."""

    path: str
    session_id: str
    project_dir_name: str
    project_name: str
    size_bytes: int


# ---------------------------------------------------------------------------
# Line types to skip entirely
# ---------------------------------------------------------------------------

_SKIP_TYPES = frozenset(
    {
        "file-history-snapshot",
        "permission-mode",
        "attachment",
        "last-prompt",
    }
)

# Regex to strip <system-reminder>...</system-reminder> blocks
_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)

# Regex to detect slash-command-only messages
_COMMAND_ONLY_RE = re.compile(r"^\s*<command-name>/\w+</command-name>", re.DOTALL)

# Regex to detect local-command-caveat wrapper
_LOCAL_COMMAND_RE = re.compile(
    r"^\s*<local-command-caveat>.*?</local-command-caveat>\s*$", re.DOTALL
)


# ---------------------------------------------------------------------------
# Session discovery
# ---------------------------------------------------------------------------


def discover_sessions(
    claude_dir: Path | None = None,
    *,
    min_size: int = 10240,
    exclude_subagents: bool = True,
) -> list[SessionInfo]:
    """Scan ~/.claude/projects/ for JSONL session files.

    Returns a list of SessionInfo sorted by size descending.
    Skips files smaller than ``min_size`` bytes.
    """
    if claude_dir is None:
        claude_dir = Path.home() / ".claude" / "projects"

    if not claude_dir.is_dir():
        return []

    sessions: list[SessionInfo] = []
    for jsonl_file in claude_dir.rglob("*.jsonl"):
        if exclude_subagents and "subagents" in jsonl_file.parts:
            continue

        size = jsonl_file.stat().st_size
        if size < min_size:
            continue

        session_id = jsonl_file.stem
        project_dir_name = jsonl_file.parent.name

        sessions.append(
            SessionInfo(
                path=str(jsonl_file),
                session_id=session_id,
                project_dir_name=project_dir_name,
                project_name=infer_project_name(project_dir_name),
                size_bytes=size,
            )
        )

    sessions.sort(key=lambda s: s.size_bytes, reverse=True)
    return sessions


def infer_project_name(project_dir_name: str) -> str:
    """Derive a human-readable project name from a Claude Code project directory.

    Claude Code encodes paths as ``-Users-stevengreenberg-repos-sax-forge``.
    We take the last segment as the project name.
    """
    parts = project_dir_name.split("-")
    # Filter out empty strings from leading dashes
    parts = [p for p in parts if p]
    return parts[-1] if parts else project_dir_name


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------


def parse_jsonl_file(path: Path) -> ParsedTranscript:
    """Parse a Claude Code JSONL session file into structured messages.

    Filters out non-conversational noise and compacts tool results.
    """
    meta = TranscriptMeta()
    messages: list[TranscriptMessage] = []

    with open(path) as f:
        for line_num, raw_line in enumerate(f, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSON at line %d in %s", line_num, path)
                continue

            # Extract metadata from the first user message
            if not meta.session_id and data.get("sessionId"):
                meta.session_id = data["sessionId"]
            if not meta.project_dir and data.get("cwd"):
                meta.project_dir = data["cwd"]
                meta.project_name = infer_project_name(Path(data["cwd"]).name)
            if not meta.git_branch and data.get("gitBranch"):
                meta.git_branch = data["gitBranch"]

            parsed = _parse_line(data)
            if parsed:
                messages.extend(parsed)

    return ParsedTranscript(meta=meta, messages=messages)


def _parse_line(data: dict[str, Any]) -> list[TranscriptMessage] | None:
    """Parse a single JSONL line into TranscriptMessages.

    Returns None for lines that should be skipped entirely.
    Returns a list because a single line can contain multiple content blocks.
    """
    line_type = data.get("type", "")

    if line_type in _SKIP_TYPES:
        return None

    if line_type == "system":
        # Only keep system messages that contain error information
        content = data.get("content", "")
        if isinstance(content, str) and "error" in content.lower():
            return [TranscriptMessage(role="system", text=content[:500], is_error=True)]
        return None

    if line_type not in ("user", "assistant"):
        return None

    msg = data.get("message", {})
    role = msg.get("role", line_type)
    content = msg.get("content")

    if content is None:
        return None

    # String content (simple user messages)
    if isinstance(content, str):
        text = _clean_text(content)
        if not text:
            return None
        return [TranscriptMessage(role=role, text=text)]

    # List of content blocks
    if not isinstance(content, list):
        return None

    result: list[TranscriptMessage] = []
    for block in content:
        block_type = block.get("type", "")

        if block_type == "thinking":
            continue

        if block_type == "text":
            text = _clean_text(block.get("text", ""))
            if text:
                result.append(TranscriptMessage(role=role, text=text))

        elif block_type == "tool_use":
            summary = _summarize_tool_use(block)
            if summary:
                result.append(
                    TranscriptMessage(
                        role=role,
                        text=summary,
                        tool_name=block.get("name", ""),
                    )
                )

        elif block_type == "tool_result":
            summary = _compact_tool_result(block)
            if summary:
                result.append(TranscriptMessage(role=role, text=summary))

    return result if result else None


def _clean_text(text: str) -> str:
    """Clean user/assistant text content.

    Strips system-reminder tags, detects command-only messages, and
    removes local-command wrappers.
    """
    if not text:
        return ""

    # Skip command-only messages (/clear, /exit, etc.)
    if _COMMAND_ONLY_RE.match(text):
        return ""

    # Skip local-command-caveat wrapper messages
    if _LOCAL_COMMAND_RE.match(text):
        return ""

    # Strip <system-reminder> blocks
    text = _SYSTEM_REMINDER_RE.sub("", text)

    # Strip local-command-stdout tags (keep content)
    text = re.sub(
        r"<local-command-stdout>(.*?)</local-command-stdout>",
        r"\1",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"<local-command-caveat>.*?</local-command-caveat>", "", text, flags=re.DOTALL)

    return text.strip()


def _summarize_tool_use(block: dict[str, Any]) -> str:
    """Summarize a tool_use block as a compact string."""
    name = block.get("name", "unknown")
    inp = block.get("input", {})

    # Brief summary of input based on tool type
    if name in ("Read", "Glob", "Grep"):
        path = inp.get("file_path", inp.get("path", inp.get("pattern", "")))
        return f"[tool: {name} {path}]" if path else f"[tool: {name}]"

    if name in ("Edit", "Write"):
        path = inp.get("file_path", "")
        return f"[tool: {name} {path}]" if path else f"[tool: {name}]"

    if name == "Bash":
        cmd = inp.get("command", "")
        # Truncate long commands
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        return f"[tool: Bash `{cmd}`]"

    if name == "Agent":
        desc = inp.get("description", "")
        return f'[tool: Agent "{desc}"]' if desc else "[tool: Agent]"

    return f"[tool: {name}]"


def _compact_tool_result(block: dict[str, Any]) -> str:
    """Compact a tool_result block to a size summary."""
    content = block.get("content", "")

    if isinstance(content, str):
        n = len(content)
        if n == 0:
            return ""
        return f"[tool result: {n} chars]"

    if isinstance(content, list):
        total = sum(len(str(b.get("text", b.get("content", "")))) for b in content)
        return f"[tool result: {total} chars]"

    return "[tool result]"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_transcript(transcript: ParsedTranscript) -> str:
    """Render a parsed transcript as readable text for LLM analysis.

    Produces a compact format:
        USER: <text>
        ASSISTANT: <text>
        [tool: Read src/foo.py]
        [tool result: 200 chars]
        ASSISTANT: <text>
    """
    lines: list[str] = []
    for msg in transcript.messages:
        if msg.tool_name or msg.text.startswith("[tool"):
            lines.append(msg.text)
        else:
            label = msg.role.upper()
            lines.append(f"{label}: {msg.text}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_transcript(
    transcript: ParsedTranscript,
    *,
    max_chars: int = 100_000,
    overlap_messages: int = 5,
) -> list[ParsedTranscript]:
    """Split a large transcript into overlapping chunks.

    Each chunk shares the same metadata. Overlap ensures the LLM has
    context at chunk boundaries.

    If the transcript fits within ``max_chars``, returns a single-element list.
    """
    rendered = render_transcript(transcript)
    if len(rendered) <= max_chars:
        return [transcript]

    messages = transcript.messages
    chunks: list[ParsedTranscript] = []
    start = 0

    while start < len(messages):
        # Find how many messages fit in this chunk
        end = start
        char_count = 0
        while end < len(messages):
            msg_len = len(messages[end].text) + 20  # overhead for label
            if char_count + msg_len > max_chars and end > start:
                break
            char_count += msg_len
            end += 1

        chunks.append(
            ParsedTranscript(
                meta=transcript.meta,
                messages=messages[start:end],
            )
        )

        # Advance with overlap
        start = max(start + 1, end - overlap_messages)

    return chunks
