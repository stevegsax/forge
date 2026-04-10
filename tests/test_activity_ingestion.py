"""Tests for forge.activities.ingestion.prepare_transcript.

These tests exercise the activity as a plain async function (not through
a Temporal worker), which is sufficient since prepare_transcript is a
pure data-transformation wrapper around pbook's transcript parsing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from forge.activities.ingestion import prepare_transcript

if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a minimal Claude Code JSONL session file.

    Each record is serialized to a single line. pbook's parse_jsonl_file
    reads session metadata from the top-level "sessionId", "cwd", and
    "gitBranch" keys on any record.
    """
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def _user_msg(text: str, *, session_id: str = "sess-1", cwd: str = "/repo/proj") -> dict:
    return {
        "type": "user",
        "sessionId": session_id,
        "cwd": cwd,
        "gitBranch": "main",
        "message": {"role": "user", "content": text},
    }


def _assistant_msg(text: str, *, session_id: str = "sess-1") -> dict:
    return {
        "type": "assistant",
        "sessionId": session_id,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    }


class TestPrepareTranscriptMissingFile:
    @pytest.mark.asyncio
    async def test_missing_file_returns_empty_result(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.jsonl"
        input_json = json.dumps(
            {"path": str(missing), "project": "myproj", "session_id": "s1"}
        )

        result_json = await prepare_transcript(input_json)
        result = json.loads(result_json)

        assert result["transcript_text"] == ""
        assert result["message_count"] == 0
        assert result["char_count"] == 0
        assert result["project"] == "myproj"
        assert result["session_id"] == "s1"

    @pytest.mark.asyncio
    async def test_missing_file_omits_prompts(self, tmp_path: Path) -> None:
        """When the file is missing, no system/user prompts are built."""
        missing = tmp_path / "missing.jsonl"
        input_json = json.dumps({"path": str(missing)})

        result_json = await prepare_transcript(input_json)
        result = json.loads(result_json)

        assert "system_prompt" not in result
        assert "user_prompt" not in result

    @pytest.mark.asyncio
    async def test_missing_file_uses_stem_for_session_id(self, tmp_path: Path) -> None:
        missing = tmp_path / "session-abc.jsonl"
        input_json = json.dumps({"path": str(missing)})

        result_json = await prepare_transcript(input_json)
        result = json.loads(result_json)

        assert result["session_id"] == "session-abc"


class TestPrepareTranscriptValidFile:
    @pytest.mark.asyncio
    async def test_valid_file_produces_full_output_schema(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "sess.jsonl"
        _write_jsonl(
            path,
            [
                _user_msg("How do I write a Python function?"),
                _assistant_msg("Use the def keyword followed by the function name."),
                _user_msg("Thanks, that helps."),
            ],
        )

        input_json = json.dumps(
            {"path": str(path), "project": "myproj", "session_id": "sess-1"}
        )
        result_json = await prepare_transcript(input_json)
        result = json.loads(result_json)

        # All documented fields must be present
        assert "transcript_text" in result
        assert "system_prompt" in result
        assert "user_prompt" in result
        assert "project" in result
        assert "session_id" in result
        assert "message_count" in result
        assert "char_count" in result

    @pytest.mark.asyncio
    async def test_valid_file_produces_nonempty_transcript(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "sess.jsonl"
        _write_jsonl(
            path,
            [
                _user_msg("hello"),
                _assistant_msg("hi there"),
            ],
        )

        result_json = await prepare_transcript(
            json.dumps({"path": str(path), "project": "p", "session_id": "s"})
        )
        result = json.loads(result_json)

        assert result["transcript_text"] != ""
        assert result["char_count"] == len(result["transcript_text"])
        assert result["message_count"] >= 2

    @pytest.mark.asyncio
    async def test_valid_file_produces_nonempty_prompts(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "sess.jsonl"
        _write_jsonl(
            path,
            [_user_msg("hi"), _assistant_msg("hello")],
        )

        result_json = await prepare_transcript(
            json.dumps({"path": str(path), "project": "demo", "session_id": "s"})
        )
        result = json.loads(result_json)

        assert result["system_prompt"]  # non-empty
        assert result["user_prompt"]  # non-empty
        # User prompt should mention the project name
        assert "demo" in result["user_prompt"]

    @pytest.mark.asyncio
    async def test_char_count_matches_transcript_text(self, tmp_path: Path) -> None:
        path = tmp_path / "sess.jsonl"
        _write_jsonl(
            path,
            [_user_msg("test content"), _assistant_msg("response")],
        )

        result_json = await prepare_transcript(
            json.dumps({"path": str(path), "project": "p", "session_id": "s"})
        )
        result = json.loads(result_json)

        assert result["char_count"] == len(result["transcript_text"])


class TestPrepareTranscriptProjectFallback:
    @pytest.mark.asyncio
    async def test_caller_project_takes_precedence(self, tmp_path: Path) -> None:
        """When the caller provides a project, it must not be overwritten."""
        path = tmp_path / "sess.jsonl"
        _write_jsonl(
            path,
            [_user_msg("hi", cwd="/home/user/other-project"), _assistant_msg("hello")],
        )

        result_json = await prepare_transcript(
            json.dumps({"path": str(path), "project": "explicit-project"})
        )
        result = json.loads(result_json)

        assert result["project"] == "explicit-project"

    @pytest.mark.asyncio
    async def test_empty_project_falls_back_to_transcript_metadata(
        self, tmp_path: Path
    ) -> None:
        """When caller passes empty project, fall back to transcript meta."""
        path = tmp_path / "sess.jsonl"
        # pbook derives project_name from the cwd basename
        _write_jsonl(
            path,
            [
                _user_msg("hi", cwd="/home/user/my-awesome-project"),
                _assistant_msg("hello"),
            ],
        )

        result_json = await prepare_transcript(
            json.dumps({"path": str(path), "project": ""})
        )
        result = json.loads(result_json)

        # Should have fallen back to something derived from the transcript
        assert result["project"] != ""

    @pytest.mark.asyncio
    async def test_missing_project_key_defaults_to_empty_then_falls_back(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "sess.jsonl"
        _write_jsonl(
            path,
            [_user_msg("hi", cwd="/repo/fallback-proj"), _assistant_msg("hello")],
        )

        # Omit project entirely
        result_json = await prepare_transcript(json.dumps({"path": str(path)}))
        result = json.loads(result_json)

        # Either the fallback from metadata, or empty if pbook can't derive one.
        # We assert the key exists (schema contract).
        assert "project" in result


class TestPrepareTranscriptSessionId:
    @pytest.mark.asyncio
    async def test_caller_session_id_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "raw-filename.jsonl"
        _write_jsonl(path, [_user_msg("hi"), _assistant_msg("hello")])

        result_json = await prepare_transcript(
            json.dumps({"path": str(path), "session_id": "caller-supplied-id"})
        )
        result = json.loads(result_json)

        assert result["session_id"] == "caller-supplied-id"

    @pytest.mark.asyncio
    async def test_missing_session_id_uses_path_stem(self, tmp_path: Path) -> None:
        path = tmp_path / "sessname.jsonl"
        _write_jsonl(path, [_user_msg("hi"), _assistant_msg("hello")])

        result_json = await prepare_transcript(json.dumps({"path": str(path)}))
        result = json.loads(result_json)

        assert result["session_id"] == "sessname"
