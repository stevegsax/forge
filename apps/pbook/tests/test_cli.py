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
        assert "Added entry" in result.output
        assert "New Entry" in result.output

    def test_add_reads_stdin(self, tmp_path, monkeypatch):
        """When --file is omitted, JSON is read from stdin."""
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main, ["add", "--json"],
            input=json.dumps({
                "title": "From stdin",
                "content": "...",
                "tags": ["lang:python"],
            }),
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["title"] == "From stdin"
        assert data["approved"] is True
        assert data["needs_review"] is False
        assert data["rejected"] is False

    def test_add_needs_review_flag(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main, ["add", "--needs-review", "--json"],
            input=json.dumps({
                "title": "Pending review",
                "content": "...",
                "tags": ["lang:python"],
            }),
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["needs_review"] is True
        assert data["approved"] is False

    def test_add_validation_error_envelope(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["add", "--json"], input="not valid json")
        assert result.exit_code != 0
        payload = json.loads(result.stdout)
        assert payload["code"] == "validation_error"

    def test_add_tag_invalid_envelope(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            main, ["add", "--json"],
            input=json.dumps({
                "title": "Bad Tags",
                "content": "...",
                "tags": ["not-namespaced"],
            }),
        )
        assert result.exit_code != 0
        payload = json.loads(result.stdout)
        assert payload["code"] == "tag_invalid"

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
        # Human-readable error path: stderr says "Tag must use namespace:value..."
        assert "namespace:value" in result.output or "Tag" in result.output

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

    def test_reject_soft_marks_entry(self, tmp_path, monkeypatch):
        """`reject` soft-marks the row; it survives for audit and is
        hidden from default queries."""
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
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
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
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
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        _seed_entry(engine)

        runner = CliRunner()
        result = runner.invoke(main, ["reject", "1", "--json"])
        data = json.loads(result.stdout)
        assert data["rejection_reason"] is None

    def test_approve_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
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
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["reject", "999"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_reject_missing_json_envelope(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
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
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
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
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
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

        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        engine = _setup_db(tmp_path)
        # Two needs_review entries from the same experience
        _seed_entry(engine, title="A from exp", needs_review=True)
        _seed_entry(engine, title="B from exp", needs_review=True)
        _seed_entry(engine, title="C alone", needs_review=True)
        add_entry_source(
            engine, entry_id=1, session_id="s1", project_name="p",
            experience_hash="shared", source_context="x", source_context_embedding=b"",
        )
        add_entry_source(
            engine, entry_id=2, session_id="s2", project_name="p",
            experience_hash="shared", source_context="y", source_context_embedding=b"",
        )
        add_entry_source(
            engine, entry_id=3, session_id="s3", project_name="p",
            experience_hash="lone", source_context="z", source_context_embedding=b"",
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

        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
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
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
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
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        # Minimal valid Claude Code JSONL: one user message.
        jsonl = tmp_path / "fake.jsonl"
        jsonl.write_text(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "hello"},
        }) + "\n")

        runner = CliRunner()
        result = runner.invoke(
            main, ["session-text", "fake", "--path", str(jsonl)],
        )
        assert result.exit_code == 0
        assert "USER" in result.output or "hello" in result.output

    def test_session_text_raw_returns_jsonl(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        jsonl = tmp_path / "fake.jsonl"
        jsonl.write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')

        runner = CliRunner()
        result = runner.invoke(
            main, ["session-text", "fake", "--path", str(jsonl), "--raw"],
        )
        assert result.exit_code == 0
        assert '"type":"user"' in result.output

    def test_session_text_missing_returns_envelope(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
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
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
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

        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
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
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["skill-prompt"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "commands" in data
        assert "workflows" in data
        assert "tags" in data
        assert set(data["workflows"]) == {
            "query", "discuss", "feedback", "review_queue", "add",
        }

    def test_operation_filter(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
        _setup_db(tmp_path)

        runner = CliRunner()
        result = runner.invoke(main, ["skill-prompt", "--operation", "discuss"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "workflow" in data
        assert "## Discuss workflow" in data["workflow"]

    def test_unknown_operation_returns_validation_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", str(tmp_path / "test.db"))
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


# Real skill-prompt tests are in TestSkillPromptCommand above.
