"""Tests for pbook.cli."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

from pbook.cli import EXIT_CONFIG_ERROR, main
from pbook.models import PlaybookEntry
from pbook.store import (
    build_entry_dict,
    get_entry_by_id,
    record_feedback,
    record_retrieval,
    save_entries,
)
from tests.conftest import make_embedding, setup_db


def _setup_db(_tmp_path: Path | None = None):
    """Return the session test engine (migrations already applied)."""
    return setup_db()[0]


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
        runner = CliRunner()
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        assert "No entries found" in result.output

    def test_list_entries(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        assert "Test Entry" in result.output

    def test_list_json(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1

    def test_list_filter_by_tag(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="Python tip", tags=["lang:python"])
        _seed_entry(engine, title="Go tip", tags=["lang:go"])

        runner = CliRunner()
        result = runner.invoke(main, ["list", "--tag", "lang:python"])
        assert result.exit_code == 0
        assert "Python tip" in result.output
        assert "Go tip" not in result.output

    def test_list_filter_by_type(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="Curated tip", entry_type="curated")
        _seed_entry(engine, title="Pitfall tip", entry_type="pitfall")

        runner = CliRunner()
        result = runner.invoke(main, ["list", "--type", "pitfall"])
        assert result.exit_code == 0
        assert "Pitfall tip" in result.output
        assert "Curated tip" not in result.output

    def test_list_filter_by_project(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="Forge tip", source_project="forge")
        _seed_entry(engine, title="Other tip", source_project="other")

        runner = CliRunner()
        result = runner.invoke(main, ["list", "--project", "forge"])
        assert result.exit_code == 0
        assert "Forge tip" in result.output
        assert "Other tip" not in result.output

    def test_list_disabled_store(self, monkeypatch):
        monkeypatch.setenv("PBOOK_DATABASE_URL", "")
        runner = CliRunner()
        result = runner.invoke(main, ["list"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# get command
# ---------------------------------------------------------------------------


class TestGetCommand:
    def test_get_entry(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["get", "1"])
        assert result.exit_code == 0
        assert "Test Entry" in result.output

    def test_get_json(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["get", "1", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["title"] == "Test Entry"

    def test_get_missing(self, tmp_path, monkeypatch):
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
        engine = _setup_db(tmp_path)
        _seed_entry(engine, tags=["lang:python", "lib:pytest"])

        runner = CliRunner()
        result = runner.invoke(main, ["get", "1", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["tags"] == ["lang:python", "lib:pytest"]
        assert "tags_json" not in data  # raw column name shouldn't leak

    def test_get_json_strips_embedding(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["get", "1", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "embedding" not in data

    def test_get_json_datetime_iso_8601(self, tmp_path, monkeypatch):
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
        _setup_db(tmp_path)

        entry_file = tmp_path / "entry.json"
        entry_file.write_text(
            json.dumps(
                {
                    "title": "New Entry",
                    "content": "Advice here",
                    "tags": ["lang:python", "domain:testing"],
                }
            )
        )

        runner = CliRunner()
        result = runner.invoke(main, ["add", "--file", str(entry_file)])
        assert result.exit_code == 0
        assert "Added entry" in result.output
        assert "New Entry" in result.output

    def test_add_reads_stdin(self, tmp_path, monkeypatch):
        """When --file is omitted, JSON is read from stdin."""
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["add", "--json"],
            input=json.dumps(
                {
                    "title": "From stdin",
                    "content": "...",
                    "tags": ["lang:python"],
                }
            ),
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["title"] == "From stdin"
        assert data["approved"] is True
        assert data["needs_review"] is False
        assert data["rejected"] is False

    def test_add_needs_review_flag(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["add", "--needs-review", "--json"],
            input=json.dumps(
                {
                    "title": "Pending review",
                    "content": "...",
                    "tags": ["lang:python"],
                }
            ),
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["needs_review"] is True
        assert data["approved"] is False

    def test_add_validation_error_envelope(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["add", "--json"], input="not valid json")
        assert result.exit_code != 0
        payload = json.loads(result.stdout)
        assert payload["code"] == "validation_error"

    def test_add_tag_invalid_envelope(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["add", "--json"],
            input=json.dumps(
                {
                    "title": "Bad Tags",
                    "content": "...",
                    "tags": ["not-namespaced"],
                }
            ),
        )
        assert result.exit_code != 0
        payload = json.loads(result.stdout)
        assert payload["code"] == "tag_invalid"

    def test_add_invalid_tags(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        entry_file = tmp_path / "entry.json"
        entry_file.write_text(
            json.dumps(
                {
                    "title": "Bad Tags",
                    "content": "Content",
                    "tags": ["not-namespaced"],
                }
            )
        )

        runner = CliRunner()
        result = runner.invoke(main, ["add", "--file", str(entry_file)])
        assert result.exit_code != 0
        # Human-readable error path: stderr says "Tag must use namespace:value..."
        assert "namespace:value" in result.output or "Tag" in result.output

    def test_add_invalid_json(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        entry_file = tmp_path / "bad.json"
        entry_file.write_text("not valid json")

        runner = CliRunner()
        result = runner.invoke(main, ["add", "--file", str(entry_file)])
        assert result.exit_code != 0
        assert "Validation error" in result.output or "error" in result.output.lower()

    def test_add_schema(self, tmp_path, monkeypatch):
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
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        update_file = tmp_path / "update.json"
        update_file.write_text(json.dumps({"tags": ["lang:go", "lib:temporal"]}))

        runner = CliRunner()
        result = runner.invoke(main, ["update", "1", "--file", str(update_file)])
        assert result.exit_code == 0

    def test_update_invalid_tags(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        update_file = tmp_path / "update.json"
        update_file.write_text(json.dumps({"tags": ["bad-tag"]}))

        runner = CliRunner()
        result = runner.invoke(main, ["update", "1", "--file", str(update_file)])
        assert result.exit_code != 0
        assert "Tag error" in result.output

    def test_update_missing_entry(self, tmp_path, monkeypatch):
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
        engine = _setup_db(tmp_path)
        _seed_entry(engine, needs_review=True)

        runner = CliRunner()
        result = runner.invoke(main, ["approve", "1"])
        assert result.exit_code == 0
        assert "Approved" in result.output

    def test_approve_missing(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["approve", "999"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_reject_soft_marks_entry(self, tmp_path, monkeypatch):
        """`reject` soft-marks the row; it survives for audit and is
        hidden from default queries."""
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["reject", "1"])
        assert result.exit_code == 0
        assert "Rejected" in result.output

        # Row still exists — pbook get can find it.
        result = runner.invoke(main, ["get", "1", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["rejected"] is True

        # But list hides it by default.
        result = runner.invoke(main, ["list", "--json"])
        assert result.exit_code == 0
        assert json.loads(result.output) == []

        # --include-rejected surfaces it.
        result = runner.invoke(main, ["list", "--json", "--include-rejected"])
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["rejected"] is True

    def test_reject_with_reason(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["reject", "1", "--reason", "wrong project", "--json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        # Reject now goes through a workflow that returns the full
        # status dict (consistent shape with approve/update). The
        # critical fields are unchanged.
        assert data["id"] == 1
        assert data["title"] == "Test Entry"
        assert data["approved"] is False
        assert data["rejected"] is True
        assert data["rejection_reason"] == "wrong project"

    def test_reject_without_reason_emits_null(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["reject", "1", "--json"])
        data = json.loads(result.stdout)
        assert data["rejection_reason"] is None

    def test_approve_json(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine, needs_review=True)

        runner = CliRunner()
        result = runner.invoke(main, ["approve", "1", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["approved"] is True
        assert data["needs_review"] is False
        assert data["rejected"] is False

    def test_reject_missing(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["reject", "999"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_reject_missing_json_envelope(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["reject", "999", "--json"])
        assert result.exit_code != 0
        payload = json.loads(result.stdout)
        assert payload["code"] == "not_found"


# ---------------------------------------------------------------------------
# review --json (additional cases beyond the human-output tests below)
# ---------------------------------------------------------------------------


class TestReviewJSON:
    def test_review_json_lists_only_needs_review(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="approved", needs_review=False)
        _seed_entry(engine, title="pending", needs_review=True)

        runner = CliRunner()
        result = runner.invoke(main, ["review", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["title"] == "pending"

    def test_review_json_empty_returns_empty_array(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["review", "--json"])
        assert result.exit_code == 0
        assert json.loads(result.output) == []


# ---------------------------------------------------------------------------
# review --by-experience grouping
# ---------------------------------------------------------------------------


class TestGroupReviewByExperience:
    """Pure grouping helper — entries sharing an experience_hash form a
    cluster (>= 2 members); everything else is a singleton."""

    def test_two_entries_same_hash_form_cluster(self):
        from pbook.cli import _group_review_by_experience

        entries = [
            {"id": 1, "title": "A", "sources": [{"experience_hash": "h1"}]},
            {"id": 2, "title": "B", "sources": [{"experience_hash": "h1"}]},
        ]
        clusters, singletons = _group_review_by_experience(entries)
        assert singletons == []
        assert len(clusters) == 1
        h, ents = clusters[0]
        assert h == "h1"
        assert {e["id"] for e in ents} == {1, 2}

    def test_unique_hashes_become_singletons(self):
        from pbook.cli import _group_review_by_experience

        entries = [
            {"id": 1, "sources": [{"experience_hash": "h1"}]},
            {"id": 2, "sources": [{"experience_hash": "h2"}]},
        ]
        clusters, singletons = _group_review_by_experience(entries)
        assert clusters == []
        assert {e["id"] for e in singletons} == {1, 2}

    def test_no_sources_entry_is_singleton(self):
        from pbook.cli import _group_review_by_experience

        entries = [{"id": 5, "sources": []}]
        clusters, singletons = _group_review_by_experience(entries)
        assert clusters == []
        assert singletons == [{"id": 5, "sources": []}]

    def test_null_experience_hash_is_singleton(self):
        """Manual entries with no experience_hash shouldn't be clustered."""
        from pbook.cli import _group_review_by_experience

        entries = [
            {"id": 1, "sources": [{"experience_hash": None}]},
            {"id": 2, "sources": [{"experience_hash": None}]},
        ]
        clusters, singletons = _group_review_by_experience(entries)
        assert clusters == []
        assert len(singletons) == 2

    def test_mixed_clusters_and_singletons(self):
        from pbook.cli import _group_review_by_experience

        entries = [
            {"id": 1, "sources": [{"experience_hash": "h1"}]},
            {"id": 2, "sources": [{"experience_hash": "h1"}]},
            {"id": 3, "sources": [{"experience_hash": "h1"}]},
            {"id": 4, "sources": [{"experience_hash": "h2"}]},
            {"id": 5, "sources": []},
        ]
        clusters, singletons = _group_review_by_experience(entries)
        assert len(clusters) == 1
        assert {e["id"] for e in clusters[0][1]} == {1, 2, 3}
        assert {e["id"] for e in singletons} == {4, 5}


class TestReviewByExperienceCLI:
    def test_cluster_surfaces_in_json(self, tmp_path, monkeypatch):
        from pbook.store import add_entry_source

        engine = _setup_db(tmp_path)
        # Two needs_review entries from the same experience
        _seed_entry(engine, title="A from exp", needs_review=True)
        _seed_entry(engine, title="B from exp", needs_review=True)
        _seed_entry(engine, title="C alone", needs_review=True)
        add_entry_source(
            engine,
            entry_id=1,
            session_id="s1",
            project_name="p",
            experience_hash="shared",
            source_context="x",
        )
        add_entry_source(
            engine,
            entry_id=2,
            session_id="s2",
            project_name="p",
            experience_hash="shared",
            source_context="y",
        )
        add_entry_source(
            engine,
            entry_id=3,
            session_id="s3",
            project_name="p",
            experience_hash="lone",
            source_context="z",
        )

        runner = CliRunner()
        result = runner.invoke(main, ["review", "--by-experience", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["clusters"]) == 1
        cluster = data["clusters"][0]
        assert cluster["experience_hash"] == "shared"
        assert {e["id"] for e in cluster["entries"]} == {1, 2}
        assert {e["id"] for e in data["singletons"]} == {3}


# ---------------------------------------------------------------------------
# sources command
# ---------------------------------------------------------------------------


class TestSourcesCommand:
    def test_sources_lists_rows(self, tmp_path, monkeypatch):
        from pbook.store import add_entry_source

        engine = _setup_db(tmp_path)
        _seed_entry(engine)
        add_entry_source(
            engine,
            entry_id=1,
            session_id="abc",
            project_name="forge",
            experience_hash="h1",
            source_context="situation excerpt",
        )

        runner = CliRunner()
        result = runner.invoke(main, ["sources", "1"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert len(data) == 1
        assert data[0]["session_id"] == "abc"
        assert data[0]["source_context"] == "situation excerpt"
        # Embedding column stripped per JSON contract.
        assert "source_context_embedding" not in data[0]

    def test_sources_missing_entry_returns_not_found_envelope(
        self,
        tmp_path,
        monkeypatch,
    ):
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["sources", "999"])
        assert result.exit_code != 0
        payload = json.loads(result.stdout)
        assert payload["code"] == "not_found"


# ---------------------------------------------------------------------------
# session-text command
# ---------------------------------------------------------------------------


class TestSessionTextCommand:
    def test_session_text_path_override_renders(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        # Minimal valid Claude Code JSONL: one user message.
        jsonl = tmp_path / "fake.jsonl"
        jsonl.write_text(
            json.dumps(
                {
                    "type": "user",
                    "message": {"role": "user", "content": "hello"},
                }
            )
            + "\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["session-text", "fake", "--path", str(jsonl)],
        )
        assert result.exit_code == 0
        assert "USER" in result.output or "hello" in result.output

    def test_session_text_raw_returns_jsonl(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        jsonl = tmp_path / "fake.jsonl"
        jsonl.write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["session-text", "fake", "--path", str(jsonl), "--raw"],
        )
        assert result.exit_code == 0
        assert '"type":"user"' in result.output

    def test_session_text_missing_returns_envelope(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["session-text", "no-such-session", "--json"])
        assert result.exit_code != 0
        payload = json.loads(result.stdout)
        assert payload["code"] == "session_file_missing"


# ---------------------------------------------------------------------------
# tags command
# ---------------------------------------------------------------------------


class TestTagsCommand:
    def test_tags_json_includes_namespaces_and_values(
        self,
        tmp_path,
        monkeypatch,
    ):
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="A", tags=["lang:python", "lib:sqlalchemy"])
        _seed_entry(engine, title="B", tags=["lang:go", "domain:cli"])

        runner = CliRunner()
        result = runner.invoke(main, ["tags"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "lang" in data["namespaces"]["general"]
        assert "project" in data["namespaces"]["extracted"]
        assert "python" in data["values_in_use"]["lang"]
        assert "go" in data["values_in_use"]["lang"]
        assert "sqlalchemy" in data["values_in_use"]["lib"]
        assert "cli" in data["values_in_use"]["domain"]

    def test_tags_excludes_rejected_entries(self, tmp_path, monkeypatch):
        from pbook.store import mark_rejected

        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="kept", tags=["lang:python"])
        _seed_entry(engine, title="dropped", tags=["lang:elixir"])
        mark_rejected(engine, 2)

        runner = CliRunner()
        result = runner.invoke(main, ["tags"])
        data = json.loads(result.stdout)
        assert "python" in data["values_in_use"]["lang"]
        assert "elixir" not in data["values_in_use"]["lang"]


# ---------------------------------------------------------------------------
# skill-prompt command
# ---------------------------------------------------------------------------


class TestSkillPromptCommand:
    def test_full_payload(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["skill-prompt"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "commands" in data
        assert "workflows" in data
        assert "tags" in data
        assert set(data["workflows"]) == {
            "query",
            "discuss",
            "feedback",
            "review_queue",
            "add",
        }

    def test_operation_filter(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["skill-prompt", "--operation", "discuss"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "workflow" in data
        assert "## Discuss workflow" in data["workflow"]

    def test_unknown_operation_returns_validation_error(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["skill-prompt", "--operation", "bogus"])
        assert result.exit_code != 0
        payload = json.loads(result.stdout)
        assert payload["code"] == "validation_error"


# ---------------------------------------------------------------------------
# check-duplicate
# ---------------------------------------------------------------------------


class TestCheckDuplicate:
    def test_finds_duplicate(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="Use dispose() in tests")

        runner = CliRunner()
        result = runner.invoke(main, ["check-duplicate", "--title", "dispose"])
        assert result.exit_code == 0
        assert "duplicate" in result.output.lower()

    def test_no_duplicate(self, tmp_path, monkeypatch):
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
        engine = _setup_db(tmp_path)
        _seed_entry(engine, title="Needs review", needs_review=True)
        _seed_entry(engine, title="Already approved", needs_review=False)

        runner = CliRunner()
        result = runner.invoke(main, ["review"])
        assert result.exit_code == 0
        assert "Needs review" in result.output
        assert "Already approved" not in result.output

    def test_review_empty(self, tmp_path, monkeypatch):
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
        runner = CliRunner()
        result = runner.invoke(main, ["migrate"])
        assert result.exit_code == 0
        assert "Migrations complete" in result.output


# ---------------------------------------------------------------------------
# ST-G2 environment guard (T0.9)
# ---------------------------------------------------------------------------


class TestEnvGuard:
    """The root CLI group refuses to run without an explicitly declared FORGE_ENV.

    The guard runs in the group callback, ahead of every command body (including
    ``PbookSettings()``). A missing or invalid environment exits
    ``EXIT_CONFIG_ERROR`` (78) with the guard's actionable message on stderr; a
    declared environment lets the command proceed. ``add --schema`` is a cheap
    command whose body needs no database. ``FORGE_ENV=test`` comes from the
    autouse ``_forge_env`` fixture; the failure cases override it.
    """

    def test_missing_forge_env_exits_78(self, monkeypatch):
        monkeypatch.delenv("FORGE_ENV", raising=False)
        runner = CliRunner()
        result = runner.invoke(main, ["add", "--schema"])
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "no default environment" in result.stderr

    def test_invalid_forge_env_exits_78(self, monkeypatch):
        monkeypatch.setenv("FORGE_ENV", "staging")
        runner = CliRunner()
        result = runner.invoke(main, ["add", "--schema"])
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "not a valid environment" in result.stderr

    def test_prod_without_ack_refused(self, monkeypatch):
        monkeypatch.setenv("FORGE_ENV", "prod")
        monkeypatch.delenv("FORGE_ENV_TAG", raising=False)
        monkeypatch.delenv("FORGE_PROD_ACK", raising=False)
        runner = CliRunner()
        result = runner.invoke(main, ["add", "--schema"])
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "explicit act" in result.stderr

    def test_test_env_proceeds(self):
        # FORGE_ENV=test is set by the autouse _forge_env fixture, so the guard
        # passes and the command runs to completion.
        runner = CliRunner()
        result = runner.invoke(main, ["add", "--schema"])
        assert result.exit_code == 0
        schema = json.loads(result.output)
        assert "properties" in schema


# ---------------------------------------------------------------------------
# --env profile flag (T0.9 follow-up)
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_environ():
    """Snapshot ``os.environ`` and restore it on teardown.

    ``--env`` mutates the real process environment (that is the whole feature),
    and those writes are not tracked by ``monkeypatch``, so they would leak
    between tests. This fixture restores a full snapshot afterward.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


class TestEnvProfileFlag:
    """``pbook --env NAME|PATH`` loads a profile before the guard runs.

    It applies the parsed KEY=VALUE pairs to the process environment (overwriting
    ambient values), declares FORGE_ENV, then the guard runs unchanged. It never
    sets FORGE_PROD_ACK — ``--env prod`` still fails without a separately-exported
    ack. ``add --schema`` is the cheap observable command (no DB). All cases use
    ``restore_environ`` so the direct ``os.environ`` writes don't leak.
    """

    def test_path_profile_applies_vars_and_proceeds(self, tmp_path, restore_environ):
        profile = tmp_path / "dev.env"
        profile.write_text('export FORGE_ENV_TAG="dev"\nFORGE_DB_URL=sqlite:///from-profile.db\n')

        runner = CliRunner()
        result = runner.invoke(main, ["--env", str(profile), "add", "--schema"])

        assert result.exit_code == 0
        assert os.environ["FORGE_DB_URL"] == "sqlite:///from-profile.db"
        assert os.environ["FORGE_ENV"] == "dev"

    def test_name_resolves_under_xdg_config_home(self, tmp_path, monkeypatch, restore_environ):
        envs = tmp_path / "config" / "forge" / "envs"
        envs.mkdir(parents=True)
        (envs / "dev.env").write_text("FORGE_ENV_TAG=dev\nFORGE_DB_URL=sqlite:///named.db\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        runner = CliRunner()
        result = runner.invoke(main, ["--env", "dev", "add", "--schema"])

        assert result.exit_code == 0
        assert os.environ["FORGE_DB_URL"] == "sqlite:///named.db"
        assert os.environ["FORGE_ENV"] == "dev"

    def test_profile_overrides_ambient_var(self, tmp_path, restore_environ):
        os.environ["FORGE_DB_URL"] = "sqlite:///ambient.db"
        profile = tmp_path / "dev.env"
        profile.write_text("FORGE_ENV_TAG=dev\nFORGE_DB_URL=sqlite:///override.db\n")

        runner = CliRunner()
        result = runner.invoke(main, ["--env", str(profile), "add", "--schema"])

        assert result.exit_code == 0
        assert os.environ["FORGE_DB_URL"] == "sqlite:///override.db"

    def test_tag_mismatch_exits_78(self, tmp_path, monkeypatch, restore_environ):
        envs = tmp_path / "config" / "forge" / "envs"
        envs.mkdir(parents=True)
        (envs / "dev.env").write_text("FORGE_ENV_TAG=prod\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        runner = CliRunner()
        result = runner.invoke(main, ["--env", "dev", "add", "--schema"])

        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "does not match" in result.stderr

    def test_env_prod_still_requires_ack(self, tmp_path, monkeypatch, restore_environ):
        # The non-bypass proof: --env prod loads a prod-tagged profile but never
        # supplies FORGE_PROD_ACK, so the guard still refuses.
        envs = tmp_path / "config" / "forge" / "envs"
        envs.mkdir(parents=True)
        (envs / "prod.env").write_text("FORGE_ENV_TAG=prod\nFORGE_DB_URL=sqlite:///prod.db\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.delenv("FORGE_PROD_ACK", raising=False)

        runner = CliRunner()
        result = runner.invoke(main, ["--env", "prod", "add", "--schema"])

        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "explicit act" in result.stderr

    def test_missing_file_exits_78(self, tmp_path, restore_environ):
        missing = tmp_path / "nope.env"
        runner = CliRunner()
        result = runner.invoke(main, ["--env", str(missing), "add", "--schema"])
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert str(missing) in result.stderr

    def test_path_profile_without_tag_exits_78(self, tmp_path, restore_environ):
        profile = tmp_path / "notag.env"
        profile.write_text("FORGE_DB_URL=sqlite:///x.db\n")
        runner = CliRunner()
        result = runner.invoke(main, ["--env", str(profile), "add", "--schema"])
        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "FORGE_ENV_TAG" in result.stderr


# ---------------------------------------------------------------------------
# Staging-lane isolation: --env dev threads its namespace into the connect,
# and a dev env without a declared namespace is refused before connecting.
# ---------------------------------------------------------------------------


class TestNamespaceCoherence:
    """A ``--env dev`` profile carries its Temporal namespace into every connect."""

    def test_dev_profile_namespace_reaches_connect(self, tmp_path, restore_environ):
        from unittest.mock import AsyncMock, patch

        from pbook.models import RetrievalResult

        profile = tmp_path / "dev.env"
        profile.write_text("FORGE_ENV_TAG=dev\nFORGE_TEMPORAL_NAMESPACE=forge-dev\n")

        mock_client = AsyncMock()
        mock_client.execute_workflow.return_value = RetrievalResult()

        runner = CliRunner()
        with patch(
            "pbook.cli.connect_temporal", new=AsyncMock(return_value=mock_client)
        ) as mock_connect:
            result = runner.invoke(main, ["--env", str(profile), "search", "foo", "--json"])

        assert result.exit_code == 0, result.output
        mock_connect.assert_awaited_once()
        assert mock_connect.await_args.kwargs["namespace"] == "forge-dev"

    def test_dev_env_without_namespace_refuses_to_connect(self, monkeypatch, restore_environ):
        from unittest.mock import AsyncMock, patch

        # Dev without a declared namespace defaults to "default" — production's —
        # so a Temporal-touching command exits 78 with the coherence message and
        # never opens a connection.
        monkeypatch.setenv("FORGE_ENV", "dev")
        monkeypatch.delenv("FORGE_TEMPORAL_NAMESPACE", raising=False)

        runner = CliRunner()
        with patch("pbook.cli.connect_temporal", new=AsyncMock()) as mock_connect:
            result = runner.invoke(main, ["search", "foo"])

        assert result.exit_code == EXIT_CONFIG_ERROR
        assert "must not use the 'default'" in result.stderr
        mock_connect.assert_not_awaited()

    def test_pure_command_unaffected_by_namespace(self, monkeypatch, restore_environ):
        # ``add --schema`` never connects to Temporal, so a dev env with no
        # namespace (which would fail a Temporal command) leaves it alone.
        monkeypatch.setenv("FORGE_ENV", "dev")
        monkeypatch.delenv("FORGE_TEMPORAL_NAMESPACE", raising=False)

        runner = CliRunner()
        result = runner.invoke(main, ["add", "--schema"])

        assert result.exit_code == 0
        assert "properties" in json.loads(result.output)


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------


class TestFeedback:
    def test_helpful(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["feedback", "1", "--helpful"])
        assert result.exit_code == 0
        assert "helpful" in result.output

    def test_harmful(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["feedback", "1", "--harmful"])
        assert result.exit_code == 0
        assert "harmful" in result.output

    def test_missing_flag(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["feedback", "1"])
        assert result.exit_code != 0
        assert "--helpful" in result.output or "--harmful" in result.output

    def test_missing_entry(self, tmp_path, monkeypatch):
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
        assert "pattern:prune-candidate" in entry["tags"]

    def test_no_candidates(self, tmp_path, monkeypatch):
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["prune", "--dry-run"])
        assert result.exit_code == 0
        assert "No prune candidates" in result.output

    def test_missing_flag(self, tmp_path, monkeypatch):
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["prune"])
        assert result.exit_code != 0
        assert "--dry-run" in result.output or "--apply" in result.output


# ---------------------------------------------------------------------------
# skill-prompt
# ---------------------------------------------------------------------------


# Real skill-prompt tests are in TestSkillPromptCommand above.


# ---------------------------------------------------------------------------
# cli_ops activity wire-format
#
# Activities must return JSON-serializable data — embedding bytes (and
# any other raw BLOB) need to be stripped before crossing the Temporal
# boundary. The pydantic_data_converter uses pydantic's to_json, which
# fails UTF-8 validation on raw float32 bytes inside arbitrary dicts.
# Bypassed-tests don't catch this because they call activities
# in-process; these tests serialize the result explicitly.
# ---------------------------------------------------------------------------


class TestCLIOpsWireFormat:
    """JSON-serialize each activity result so the bytes-on-the-wire bug
    can't sneak back in through a new activity that forgets to strip."""

    @pytest.mark.asyncio
    async def test_get_entry_result_is_json_serializable(self, tmp_path):
        from pbook.models import EntryType
        from pbook.roots import StoreActivities

        engine = _setup_db(tmp_path)
        emb = make_embedding(1.0, 0.0, 0.0, 0.0)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="A",
                        content="x",
                        tags=["lang:python"],
                        entry_type=EntryType.CURATED,
                        embedding=emb,
                    )
                )
            ],
        )

        result = await StoreActivities(engine).get_entry_activity({"entry_id": 1})
        # Must be JSON-serializable (datetimes via default=str); crucially
        # no raw vector/BLOB columns leak across the wire.
        json.dumps(result, default=str)
        assert result is not None
        assert "embedding" not in result

    @pytest.mark.asyncio
    async def test_list_entries_result_is_json_serializable(self, tmp_path):
        from pbook.models import EntryType
        from pbook.roots import StoreActivities

        engine = _setup_db(tmp_path)
        emb = make_embedding(1.0, 0.0, 0.0, 0.0)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="A",
                        content="x",
                        tags=["lang:python"],
                        entry_type=EntryType.CURATED,
                        embedding=emb,
                    )
                )
            ],
        )

        result = await StoreActivities(engine).list_entries_activity({})
        json.dumps(result, default=str)
        assert all("embedding" not in e for e in result)

    @pytest.mark.asyncio
    async def test_review_queue_result_is_json_serializable(self, tmp_path):
        from pbook.models import EntryType
        from pbook.roots import StoreActivities

        engine = _setup_db(tmp_path)
        emb = make_embedding(1.0, 0.0, 0.0, 0.0)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="A",
                        content="x",
                        tags=["lang:python"],
                        entry_type=EntryType.CURATED,
                        embedding=emb,
                        needs_review=True,
                    )
                ),
            ],
        )

        # Both modes — flat and clustered — must be wire-safe.
        store = StoreActivities(engine)
        flat = await store.review_queue_activity({"limit": 20, "by_experience": False})
        json.dumps(flat, default=str)
        assert all("embedding" not in e for e in flat["entries"])

        clustered = await store.review_queue_activity({"limit": 20, "by_experience": True})
        json.dumps(clustered, default=str)

    @pytest.mark.asyncio
    async def test_list_sources_result_is_json_serializable(self, tmp_path):
        from pbook.models import EntryType
        from pbook.roots import StoreActivities
        from pbook.store import add_entry_source

        engine = _setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="A",
                        content="x",
                        tags=["lang:python"],
                        entry_type=EntryType.CURATED,
                    )
                )
            ],
        )
        add_entry_source(
            engine,
            entry_id=1,
            session_id="s",
            project_name="p",
            experience_hash="h",
            source_context="ctx",
            source_context_embedding=make_embedding(0.1, 0.2, 0.3),
        )

        result = await StoreActivities(engine).list_sources_activity({"entry_id": 1})
        json.dumps(result, default=str)
        for row in result["rows"]:
            assert "source_context_embedding" not in row
