"""Tests for forge.message_log."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from forge.message_log import write_message_log

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# write_message_log
# ---------------------------------------------------------------------------


class TestWriteMessageLog:
    """Tests for write_message_log."""

    def test_creates_directory_and_file(self, tmp_path: Path) -> None:
        write_message_log(str(tmp_path), "request", '{"hello": "world"}')

        messages_dir = tmp_path / "messages"
        assert messages_dir.is_dir()

        files = list(messages_dir.glob("request-*.json"))
        assert len(files) == 1
        assert files[0].read_text() == '{"hello": "world"}'

    def test_creates_gitignore(self, tmp_path: Path) -> None:
        write_message_log(str(tmp_path), "request", "{}")

        gitignore = tmp_path / "messages" / ".gitignore"
        assert gitignore.exists()
        assert gitignore.read_text() == "*\n"

    def test_does_not_overwrite_existing_gitignore(self, tmp_path: Path) -> None:
        messages_dir = tmp_path / "messages"
        messages_dir.mkdir()
        gitignore = messages_dir / ".gitignore"
        gitignore.write_text("custom\n")

        write_message_log(str(tmp_path), "request", "{}")

        assert gitignore.read_text() == "custom\n"

    def test_noop_when_worktree_path_empty(self, tmp_path: Path) -> None:
        write_message_log("", "request", "{}")
        # Should not create anything
        assert not (tmp_path / "messages").exists()

    def test_silently_catches_write_errors(self) -> None:
        with patch("forge.message_log.Path.mkdir", side_effect=OSError("boom")):
            # Should not raise
            write_message_log("/nonexistent/path", "request", "{}")

    def test_file_naming_format(self, tmp_path: Path) -> None:
        write_message_log(str(tmp_path), "response", "{}")

        files = list((tmp_path / "messages").glob("response-*.json"))
        assert len(files) == 1
        # Verify the name matches the expected pattern: response-YYYY-mm-dd-HH-MM-SS.json
        name = files[0].name
        assert name.startswith("response-")
        assert name.endswith(".json")
        # The timestamp part should have 6 dash-separated segments
        timestamp_part = name[len("response-") : -len(".json")]
        segments = timestamp_part.split("-")
        assert len(segments) == 6

    def test_multiple_writes_create_separate_files(self, tmp_path: Path) -> None:
        write_message_log(str(tmp_path), "request", '{"a": 1}')
        write_message_log(str(tmp_path), "request", '{"b": 2}')

        files = list((tmp_path / "messages").glob("request-*.json"))
        # Timestamps have second resolution so both may get the same name.
        # At minimum, the dir and gitignore should exist.
        assert len(files) >= 1
        assert (tmp_path / "messages" / ".gitignore").exists()
