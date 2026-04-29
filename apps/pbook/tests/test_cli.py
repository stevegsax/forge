"""Tests for pbook.cli."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from click.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

from pbook.cli import main
from pbook.models import PlaybookEntry
from pbook.store import (
    build_entry_dict,
    get_engine,
    get_entry_by_id,
    record_feedback,
    record_retrieval,
    run_migrations,
    save_entries,
)


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

    def test_list_filter_by_type(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="Curated tip", entry_type="curated")
        _seed_entry(engine, title="Pitfall tip", entry_type="pitfall")

        runner = CliRunner()
        result = runner.invoke(main, ["list", "--type", "pitfall"])
        assert result.exit_code == 0
        assert "Pitfall tip" in result.output
        assert "Curated tip" not in result.output

    def test_list_filter_by_project(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="Forge tip", source_project="forge")
        _seed_entry(engine, title="Other tip", source_project="other")

        runner = CliRunner()
        result = runner.invoke(main, ["list", "--project", "forge"])
        assert result.exit_code == 0
        assert "Forge tip" in result.output
        assert "Other tip" not in result.output

    def test_list_disabled_store(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", "")
        runner = CliRunner()
        result = runner.invoke(main, ["list"])
        assert result.exit_code != 0
        assert "disabled" in result.output.lower() or "disabled" in (result.output + result.output)


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

    def test_get_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["get", "1", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["title"] == "Test Entry"

    def test_get_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["get", "999"])
        assert result.exit_code != 0
        assert "not found" in result.output


# ---------------------------------------------------------------------------
# JSON contract — every --json site goes through these conventions
# ---------------------------------------------------------------------------


class TestJSONContract:
    def test_get_json_emits_tags_as_list_not_string(self, tmp_path, monkeypatch):
        """Regression: the on-disk shape stores tags as a JSON-string-in-JSON.
        Skill consumers must see a real list."""
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine, tags=["lang:python", "lib:pytest"])

        runner = CliRunner()
        result = runner.invoke(main, ["get", "1", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["tags"] == ["lang:python", "lib:pytest"]
        assert "tags_json" not in data  # raw column name shouldn't leak

    def test_get_json_strips_embedding(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["get", "1", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "embedding" not in data

    def test_get_json_datetime_iso_8601(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["get", "1", "--json"])
        data = json.loads(result.output)
        # ISO 8601 with T separator, e.g. "2026-04-28T16:23:45+00:00"
        assert "T" in data["created_at"]

    def test_get_json_error_envelope(self, tmp_path, monkeypatch):
        """When --json is set and the entry is missing, the error must
        come back as JSON on stdout with non-zero exit."""
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["get", "999", "--json"])
        assert result.exit_code != 0
        # stdout, not stderr — single parseable stream for the skill
        payload = json.loads(result.stdout)
        assert payload == {
            "error": "Entry 999 not found.",
            "code": "not_found",
        }

    def test_list_json_each_entry_has_tags_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="A", tags=["lang:python"])
        _seed_entry(engine, title="B", tags=["lang:go", "lib:cobra"])

        runner = CliRunner()
        result = runner.invoke(main, ["list", "--json"])
        data = json.loads(result.output)
        assert all(isinstance(e["tags"], list) for e in data)
        assert all("tags_json" not in e for e in data)
        assert all("embedding" not in e for e in data)

    def test_list_json_empty_returns_empty_array(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["list", "--json"])
        assert result.exit_code == 0
        assert json.loads(result.output) == []


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

    def test_add_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["add"])
        assert result.exit_code != 0
        assert "--file" in result.output or "required" in result.output.lower()

    def test_add_invalid_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        entry_file = tmp_path / "bad.json"
        entry_file.write_text("not valid json")

        runner = CliRunner()
        result = runner.invoke(main, ["add", "--file", str(entry_file)])
        assert result.exit_code != 0
        assert "Validation error" in result.output or "error" in result.output.lower()

    def test_add_schema(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        runner = CliRunner()
        result = runner.invoke(main, ["add", "--schema"])
        assert result.exit_code == 0
        schema = json.loads(result.output)
        assert "properties" in schema


# ---------------------------------------------------------------------------
# update command
# ---------------------------------------------------------------------------


class TestUpdateCommand:
    def test_update_entry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="Original Title")

        update_file = tmp_path / "update.json"
        update_file.write_text(json.dumps({"title": "Updated Title"}))

        runner = CliRunner()
        result = runner.invoke(main, ["update", "1", "--file", str(update_file)])
        assert result.exit_code == 0
        assert "Updated entry 1" in result.output

        # Verify the update applied
        result = runner.invoke(main, ["get", "1", "--json"])
        data = json.loads(result.output)
        assert data["title"] == "Updated Title"

    def test_update_tags(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        update_file = tmp_path / "update.json"
        update_file.write_text(json.dumps({"tags": ["lang:go", "lib:temporal"]}))

        runner = CliRunner()
        result = runner.invoke(main, ["update", "1", "--file", str(update_file)])
        assert result.exit_code == 0

    def test_update_invalid_tags(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        update_file = tmp_path / "update.json"
        update_file.write_text(json.dumps({"tags": ["bad-tag"]}))

        runner = CliRunner()
        result = runner.invoke(main, ["update", "1", "--file", str(update_file)])
        assert result.exit_code != 0
        assert "Tag error" in result.output

    def test_update_missing_entry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        update_file = tmp_path / "update.json"
        update_file.write_text(json.dumps({"title": "New"}))

        runner = CliRunner()
        result = runner.invoke(main, ["update", "999", "--file", str(update_file)])
        assert result.exit_code != 0
        assert "not found" in result.output


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

    def test_approve_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["approve", "999"])
        assert result.exit_code != 0
        assert "not found" in result.output

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

    def test_reject_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["reject", "999"])
        assert result.exit_code != 0
        assert "not found" in result.output


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
# review command
# ---------------------------------------------------------------------------


class TestReviewCommand:
    def test_review_shows_entries(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="Needs review", needs_review=True)
        _seed_entry(engine, title="Already approved", needs_review=False)

        runner = CliRunner()
        result = runner.invoke(main, ["review"])
        assert result.exit_code == 0
        assert "Needs review" in result.output
        assert "Already approved" not in result.output

    def test_review_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine, needs_review=False)

        runner = CliRunner()
        result = runner.invoke(main, ["review"])
        assert result.exit_code == 0
        assert "No entries need review" in result.output


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
# feedback
# ---------------------------------------------------------------------------


class TestFeedback:
    def test_helpful(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["feedback", "1", "--helpful"])
        assert result.exit_code == 0
        assert "helpful" in result.output

    def test_harmful(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["feedback", "1", "--harmful"])
        assert result.exit_code == 0
        assert "harmful" in result.output

    def test_missing_flag(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["feedback", "1"])
        assert result.exit_code != 0
        assert "--helpful" in result.output or "--harmful" in result.output

    def test_missing_entry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["feedback", "999", "--helpful"])
        assert result.exit_code != 0
        assert "not found" in result.output


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


class TestPrune:
    def test_dry_run_lists_candidates(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="Harmful entry")
        # Make it harmful: 6/10 retrievals marked harmful
        record_retrieval(engine, [1])
        for _ in range(9):
            record_retrieval(engine, [1])
        for _ in range(6):
            record_feedback(engine, 1, helpful=False)

        runner = CliRunner()
        result = runner.invoke(main, ["prune", "--dry-run"])
        assert result.exit_code == 0
        assert "Harmful entry" in result.output
        assert "harmful ratio" in result.output

        # Verify entry was NOT modified (dry run)
        entry = get_entry_by_id(engine, 1)
        assert entry["needs_review"] is False

    def test_apply_marks_for_review(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="Harmful entry")
        for _ in range(10):
            record_retrieval(engine, [1])
        for _ in range(6):
            record_feedback(engine, 1, helpful=False)

        runner = CliRunner()
        result = runner.invoke(main, ["prune", "--apply"])
        assert result.exit_code == 0
        assert "Marked" in result.output

        entry = get_entry_by_id(engine, 1)
        assert entry["needs_review"] is True
        tags = json.loads(entry["tags_json"])
        assert "pattern:prune-candidate" in tags

    def test_no_candidates(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["prune", "--dry-run"])
        assert result.exit_code == 0
        assert "No prune candidates" in result.output

    def test_missing_flag(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["prune"])
        assert result.exit_code != 0
        assert "--dry-run" in result.output or "--apply" in result.output


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
