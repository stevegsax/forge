"""Activity for preparing Claude Code transcripts for LLM analysis.

Reads a JSONL session file, parses and filters it using pbook's pure
functions, and returns the rendered transcript text ready for batch
LLM submission.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pbook.ingestion_prompts import build_analysis_system_prompt, build_analysis_user_prompt
from pbook.transcript import parse_jsonl_file, render_transcript
from temporalio import activity

logger = logging.getLogger(__name__)


@activity.defn
async def prepare_transcript(input_json: str) -> str:
    """Read and render a Claude Code JSONL transcript.

    Input JSON: {"path": str, "project": str, "session_id": str}
    Output JSON: {
        "transcript_text": str,
        "system_prompt": str,
        "user_prompt": str,
        "project": str,
        "session_id": str,
        "message_count": int,
        "char_count": int,
    }

    Returns empty transcript_text (and omits system/user prompts) if the
    session file does not exist. Caller should also treat small sessions
    (fewer than ~3 messages) as not worth analyzing.
    """
    data = json.loads(input_json)
    path = Path(data["path"])
    project = data.get("project", "")
    session_id = data.get("session_id", path.stem)

    if not path.exists():
        logger.warning("Transcript file not found: %s", path)
        return json.dumps({
            "transcript_text": "",
            "project": project,
            "session_id": session_id,
            "message_count": 0,
            "char_count": 0,
        })

    transcript = parse_jsonl_file(path)

    # Fall back to transcript metadata for project name
    if not project and transcript.meta.project_name:
        project = transcript.meta.project_name

    rendered = render_transcript(transcript)

    # Build prompts here so the workflow doesn't need to import pbook
    system_prompt = build_analysis_system_prompt()
    user_prompt = build_analysis_user_prompt(rendered, project)

    logger.info(
        "Prepared transcript %s: %d messages, %d chars rendered",
        session_id, len(transcript.messages), len(rendered),
    )

    return json.dumps({
        "transcript_text": rendered,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "project": project,
        "session_id": session_id,
        "message_count": len(transcript.messages),
        "char_count": len(rendered),
    })
