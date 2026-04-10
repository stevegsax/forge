"""Tests for Claude Code transcript parsing, filtering, and session discovery."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pbook.transcript import (
    ParsedTranscript,
    TranscriptMessage,
    TranscriptMeta,
    chunk_transcript,
    discover_sessions,
    infer_project_name,
    parse_jsonl_file,
    render_transcript,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, lines: list[dict]) -> Path:
    """Write a list of dicts as JSONL to a file."""
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


def _user_msg(text: str, *, cwd: str = "/repos/myapp", session_id: str = "sess-1") -> dict:
    """Build a minimal user message line."""
    return {
        "type": "user",
        "message": {"role": "user", "content": text},
        "cwd": cwd,
        "sessionId": session_id,
        "gitBranch": "main",
    }


def _user_msg_blocks(blocks: list[dict], **kwargs) -> dict:
    """Build a user message line with content blocks."""
    base = _user_msg("", **kwargs)
    base["message"]["content"] = blocks
    return base


def _assistant_msg(blocks: list[dict]) -> dict:
    """Build an assistant message line with content blocks."""
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": blocks},
    }


def _tool_use_block(name: str, input_dict: dict | None = None) -> dict:
    return {"type": "tool_use", "name": name, "id": "tool-1", "input": input_dict or {}}


def _tool_result_block(content: str) -> dict:
    return {"type": "tool_result", "tool_use_id": "tool-1", "content": content}


# ---------------------------------------------------------------------------
# infer_project_name
# ---------------------------------------------------------------------------


class TestInferProjectName:
    def test_standard_path(self):
        assert infer_project_name("-Users-stevengreenberg-repos-sax-forge") == "forge"

    def test_short_path(self):
        assert infer_project_name("-Users-stevengreenberg-Downloads") == "Downloads"

    def test_single_segment(self):
        assert infer_project_name("myproject") == "myproject"

    def test_empty_string(self):
        assert infer_project_name("") == ""


# ---------------------------------------------------------------------------
# parse_jsonl_file
# ---------------------------------------------------------------------------


class TestParseJsonlFile:
    def test_basic_conversation(self, tmp_path):
        lines = [
            _user_msg("Hello, can you help me?"),
            _assistant_msg([{"type": "text", "text": "Sure, what do you need?"}]),
        ]
        path = _write_jsonl(tmp_path / "session.jsonl", lines)
        result = parse_jsonl_file(path)

        assert result.meta.session_id == "sess-1"
        assert result.meta.project_dir == "/repos/myapp"
        assert result.meta.project_name == "myapp"
        assert result.meta.git_branch == "main"
        assert len(result.messages) == 2
        assert result.messages[0].role == "user"
        assert "Hello" in result.messages[0].text
        assert result.messages[1].role == "assistant"

    def test_skips_file_history_snapshot(self, tmp_path):
        lines = [
            {"type": "file-history-snapshot", "messageId": "x", "snapshot": {}},
            _user_msg("Real message"),
        ]
        path = _write_jsonl(tmp_path / "session.jsonl", lines)
        result = parse_jsonl_file(path)

        assert len(result.messages) == 1
        assert "Real message" in result.messages[0].text

    def test_skips_permission_mode(self, tmp_path):
        lines = [
            {"type": "permission-mode", "permissionMode": "default", "sessionId": "s1"},
            _user_msg("After permission"),
        ]
        path = _write_jsonl(tmp_path / "session.jsonl", lines)
        result = parse_jsonl_file(path)
        assert len(result.messages) == 1

    def test_skips_attachment(self, tmp_path):
        lines = [
            {"type": "attachment", "content": "big binary data..."},
            _user_msg("After attachment"),
        ]
        path = _write_jsonl(tmp_path / "session.jsonl", lines)
        result = parse_jsonl_file(path)
        assert len(result.messages) == 1

    def test_skips_command_only_messages(self, tmp_path):
        lines = [
            _user_msg('<command-name>/clear</command-name>\n<command-message>clear</command-message>'),
            _user_msg("Real question"),
        ]
        path = _write_jsonl(tmp_path / "session.jsonl", lines)
        result = parse_jsonl_file(path)
        assert len(result.messages) == 1
        assert "Real question" in result.messages[0].text

    def test_strips_system_reminders(self, tmp_path):
        text = "Hello <system-reminder>ignore this</system-reminder> world"
        lines = [_user_msg(text)]
        path = _write_jsonl(tmp_path / "session.jsonl", lines)
        result = parse_jsonl_file(path)
        assert "ignore this" not in result.messages[0].text
        assert "Hello" in result.messages[0].text
        assert "world" in result.messages[0].text

    def test_compacts_tool_use(self, tmp_path):
        lines = [
            _assistant_msg([
                _tool_use_block("Read", {"file_path": "src/main.py"}),
            ]),
        ]
        path = _write_jsonl(tmp_path / "session.jsonl", lines)
        result = parse_jsonl_file(path)
        assert len(result.messages) == 1
        assert "Read" in result.messages[0].text
        assert "src/main.py" in result.messages[0].text
        assert result.messages[0].tool_name == "Read"

    def test_compacts_tool_result(self, tmp_path):
        lines = [
            _user_msg_blocks([
                _tool_result_block("x" * 500),
            ]),
        ]
        path = _write_jsonl(tmp_path / "session.jsonl", lines)
        result = parse_jsonl_file(path)
        assert len(result.messages) == 1
        assert "500 chars" in result.messages[0].text

    def test_skips_thinking_blocks(self, tmp_path):
        lines = [
            _assistant_msg([
                {"type": "thinking", "thinking": "Let me think about this..."},
                {"type": "text", "text": "Here is the answer."},
            ]),
        ]
        path = _write_jsonl(tmp_path / "session.jsonl", lines)
        result = parse_jsonl_file(path)
        assert len(result.messages) == 1
        assert "answer" in result.messages[0].text
        assert "think" not in result.messages[0].text

    def test_handles_malformed_json_gracefully(self, tmp_path):
        path = tmp_path / "session.jsonl"
        with open(path, "w") as f:
            f.write('{"type": "user", "message": {"role": "user", "content": "ok"}}\n')
            f.write("not valid json\n")
            f.write('{"type": "user", "message": {"role": "user", "content": "still here"}}\n')
        result = parse_jsonl_file(path)
        assert len(result.messages) == 2

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        result = parse_jsonl_file(path)
        assert result.messages == []

    def test_bash_tool_use_truncation(self, tmp_path):
        lines = [
            _assistant_msg([
                _tool_use_block("Bash", {"command": "a" * 200}),
            ]),
        ]
        path = _write_jsonl(tmp_path / "session.jsonl", lines)
        result = parse_jsonl_file(path)
        assert len(result.messages[0].text) < 200

    def test_local_command_caveat_skipped(self, tmp_path):
        text = (
            "<local-command-caveat>Caveat: The messages below were generated by the user "
            "while running local commands.</local-command-caveat>"
        )
        lines = [_user_msg(text)]
        path = _write_jsonl(tmp_path / "session.jsonl", lines)
        result = parse_jsonl_file(path)
        assert len(result.messages) == 0


# ---------------------------------------------------------------------------
# render_transcript
# ---------------------------------------------------------------------------


class TestRenderTranscript:
    def test_basic_rendering(self):
        transcript = ParsedTranscript(
            meta=TranscriptMeta(project_name="test"),
            messages=[
                TranscriptMessage(role="user", text="Fix the bug"),
                TranscriptMessage(role="assistant", text="I found the issue"),
            ],
        )
        rendered = render_transcript(transcript)
        assert "USER: Fix the bug" in rendered
        assert "ASSISTANT: I found the issue" in rendered

    def test_tool_messages_rendered_without_label(self):
        transcript = ParsedTranscript(messages=[
            TranscriptMessage(role="assistant", text="[tool: Read src/main.py]", tool_name="Read"),
            TranscriptMessage(role="user", text="[tool result: 200 chars]"),
        ])
        rendered = render_transcript(transcript)
        assert "[tool: Read src/main.py]" in rendered
        assert "[tool result: 200 chars]" in rendered
        assert "USER:" not in rendered
        assert "ASSISTANT:" not in rendered


# ---------------------------------------------------------------------------
# chunk_transcript
# ---------------------------------------------------------------------------


class TestChunkTranscript:
    def test_small_transcript_single_chunk(self):
        transcript = ParsedTranscript(messages=[
            TranscriptMessage(role="user", text="short"),
        ])
        chunks = chunk_transcript(transcript, max_chars=1000)
        assert len(chunks) == 1
        assert chunks[0].messages == transcript.messages

    def test_large_transcript_multiple_chunks(self):
        messages = [
            TranscriptMessage(role="user", text=f"Message {i}: " + "x" * 100)
            for i in range(50)
        ]
        transcript = ParsedTranscript(messages=messages)
        chunks = chunk_transcript(transcript, max_chars=500, overlap_messages=2)
        assert len(chunks) > 1

        # All messages should appear in at least one chunk
        all_texts = set()
        for chunk in chunks:
            for msg in chunk.messages:
                all_texts.add(msg.text)
        for msg in messages:
            assert msg.text in all_texts

    def test_chunks_share_metadata(self):
        meta = TranscriptMeta(session_id="s1", project_name="proj")
        messages = [
            TranscriptMessage(role="user", text="x" * 200)
            for _ in range(10)
        ]
        transcript = ParsedTranscript(meta=meta, messages=messages)
        chunks = chunk_transcript(transcript, max_chars=300)
        for chunk in chunks:
            assert chunk.meta.session_id == "s1"


# ---------------------------------------------------------------------------
# discover_sessions
# ---------------------------------------------------------------------------


class TestDiscoverSessions:
    def test_discovers_jsonl_files(self, tmp_path):
        proj_dir = tmp_path / "-Users-foo-repos-myapp"
        proj_dir.mkdir()
        session_file = proj_dir / "abc123.jsonl"
        session_file.write_text("x" * 20000)

        sessions = discover_sessions(tmp_path, min_size=1000)
        assert len(sessions) == 1
        assert sessions[0].session_id == "abc123"
        assert sessions[0].project_name == "myapp"
        assert sessions[0].size_bytes == 20000

    def test_skips_small_files(self, tmp_path):
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir()
        (proj_dir / "small.jsonl").write_text("tiny")

        sessions = discover_sessions(tmp_path, min_size=1000)
        assert len(sessions) == 0

    def test_skips_subagent_files(self, tmp_path):
        proj_dir = tmp_path / "proj"
        subagent_dir = proj_dir / "sess1" / "subagents"
        subagent_dir.mkdir(parents=True)
        (subagent_dir / "agent.jsonl").write_text("x" * 20000)

        sessions = discover_sessions(tmp_path, min_size=1000, exclude_subagents=True)
        assert len(sessions) == 0

    def test_includes_subagent_files_when_not_excluded(self, tmp_path):
        proj_dir = tmp_path / "proj"
        subagent_dir = proj_dir / "sess1" / "subagents"
        subagent_dir.mkdir(parents=True)
        (subagent_dir / "agent.jsonl").write_text("x" * 20000)

        sessions = discover_sessions(tmp_path, min_size=1000, exclude_subagents=False)
        assert len(sessions) == 1

    def test_sorted_by_size_descending(self, tmp_path):
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir()
        (proj_dir / "big.jsonl").write_text("x" * 50000)
        (proj_dir / "small.jsonl").write_text("x" * 20000)

        sessions = discover_sessions(tmp_path, min_size=1000)
        assert len(sessions) == 2
        assert sessions[0].size_bytes > sessions[1].size_bytes

    def test_nonexistent_directory(self, tmp_path):
        sessions = discover_sessions(tmp_path / "nonexistent")
        assert sessions == []
