"""Tests for pbook.cli."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from click.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

from pbook.cli import main
from pbook.models import PlaybookEntry
from pbook.store import build_entry_dict, get_engine, run_migrations, save_entries


def _setup_db(tmp_path: Path):
    """Create and migrate a test database, return the engine."""
    db_path = tmp_path / "test.db"
    run_migrations(db_path)
    return get_engine(db_path)


def _seed_entry(engine, **kwargs):
    """Create a default entry, override with kwargs, and save it."""
    defaults = {
        "title": "Test Entry",
        "content": "Test content",
        "tags": ["lang:python"],
    }
    defaults.update(kwargs)
    entry = PlaybookEntry(**defaults)
    save_entries(engine, [build_entry_dict(entry)])


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


class TestListCommand:
    def test_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        runner = CliRunner()
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        assert "No entries found" in result.output

    def test_list_entries(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        assert "Test Entry" in result.output

    def test_list_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1

    def test_list_filter_by_tag(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="Python tip", tags=["lang:python"])
        _seed_entry(engine, title="Go tip", tags=["lang:go"])

        runner = CliRunner()
        result = runner.invoke(main, ["list", "--tag", "lang:python"])
        assert result.exit_code == 0
        assert "Python tip" in result.output
        assert "Go tip" not in result.output


# ---------------------------------------------------------------------------
# get command
# ---------------------------------------------------------------------------


class TestGetCommand:
    def test_get_entry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["get", "1"])
        assert result.exit_code == 0
        assert "Test Entry" in result.output

    def test_get_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["get", "999"])
        assert result.exit_code != 0
        assert "not found" in result.output


# ---------------------------------------------------------------------------
# add command
# ---------------------------------------------------------------------------


class TestAddCommand:
    def test_add_entry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        entry_file = tmp_path / "entry.json"
        entry_file.write_text(json.dumps({
            "title": "New Entry",
            "content": "Advice here",
            "tags": ["lang:python", "domain:testing"],
        }))

        runner = CliRunner()
        result = runner.invoke(main, ["add", "--file", str(entry_file)])
        assert result.exit_code == 0
        assert "Added: New Entry" in result.output

    def test_add_invalid_tags(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        entry_file = tmp_path / "entry.json"
        entry_file.write_text(json.dumps({
            "title": "Bad Tags",
            "content": "Content",
            "tags": ["not-namespaced"],
        }))

        runner = CliRunner()
        result = runner.invoke(main, ["add", "--file", str(entry_file)])
        assert result.exit_code != 0
        assert "Tag error" in result.output

    def test_add_schema(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        runner = CliRunner()
        result = runner.invoke(main, ["add", "--schema"])
        assert result.exit_code == 0
        schema = json.loads(result.output)
        assert "properties" in schema


# ---------------------------------------------------------------------------
# approve / reject
# ---------------------------------------------------------------------------


class TestApproveReject:
    def test_approve(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine, needs_review=True)

        runner = CliRunner()
        result = runner.invoke(main, ["approve", "1"])
        assert result.exit_code == 0
        assert "Approved" in result.output

    def test_reject(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["reject", "1"])
        assert result.exit_code == 0
        assert "Rejected" in result.output

        # Verify deletion
        result = runner.invoke(main, ["get", "1"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# check-duplicate
# ---------------------------------------------------------------------------


class TestCheckDuplicate:
    def test_finds_duplicate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="Use dispose() in tests")

        runner = CliRunner()
        result = runner.invoke(main, ["check-duplicate", "--title", "dispose"])
        assert result.exit_code == 0
        assert "duplicate" in result.output.lower()

    def test_no_duplicate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["check-duplicate", "--title", "unique-title"])
        assert result.exit_code == 0
        assert "No duplicates" in result.output


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------


class TestMigrate:
    def test_migrate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        runner = CliRunner()
        result = runner.invoke(main, ["migrate"])
        assert result.exit_code == 0
        assert "Migrations complete" in result.output


# ---------------------------------------------------------------------------
# skill-prompt
# ---------------------------------------------------------------------------


class TestSkillPrompt:
    def test_stub(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        runner = CliRunner()
        result = runner.invoke(main, ["skill-prompt"])
        assert result.exit_code == 0
        assert "skill-prompt" in result.output
