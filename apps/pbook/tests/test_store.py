"""Tests for pbook.store."""

from __future__ import annotations

from pathlib import Path

from pbook.models import EntryType, PlaybookEntry
from pbook.store import (
    build_entry_dict,
    check_duplicate,
    delete_entry,
    get_db_path,
    get_entries_by_tags,
    get_entry_by_id,
    list_recent_entries,
    save_entries,
    update_entry,
)
from tests.conftest import setup_db

# ---------------------------------------------------------------------------
# get_db_path
# ---------------------------------------------------------------------------


class TestGetDbPath:
    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", "/tmp/custom.db")
        assert get_db_path() == Path("/tmp/custom.db")

    def test_empty_disables(self, monkeypatch):
        monkeypatch.setenv("PBOOK_DB_PATH", "")
        assert get_db_path() is None

    def test_xdg_state(self, monkeypatch):
        monkeypatch.delenv("PBOOK_DB_PATH", raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", "/tmp/xdg")
        assert get_db_path() == Path("/tmp/xdg/pbook/pbook.db")

    def test_default(self, monkeypatch):
        monkeypatch.delenv("PBOOK_DB_PATH", raising=False)
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        result = get_db_path()
        assert result is not None
        assert result.name == "pbook.db"
        assert "pbook" in str(result)


# ---------------------------------------------------------------------------
# build_entry_dict
# ---------------------------------------------------------------------------


class TestBuildEntryDict:
    def test_basic(self):
        entry = PlaybookEntry(
            title="Test",
            content="Content",
            tags=["lang:python"],
        )
        d = build_entry_dict(entry)
        assert d["title"] == "Test"
        assert d["content"] == "Content"
        assert d["tags_json"] == '["lang:python"]'
        assert d["entry_type"] == "curated"
        assert d["needs_review"] is False

    def test_pitfall(self):
        entry = PlaybookEntry(
            title="Gotcha",
            content="Watch out",
            tags=["project:forge"],
            entry_type=EntryType.PITFALL,
            needs_review=True,
        )
        d = build_entry_dict(entry)
        assert d["entry_type"] == "pitfall"
        assert d["needs_review"] is True


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


class TestCrud:
    def test_save_and_list(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        entry = PlaybookEntry(title="Test Entry", content="Content", tags=["lang:python"])
        save_entries(engine, [build_entry_dict(entry)])

        entries = list_recent_entries(engine)
        assert len(entries) == 1
        assert entries[0]["title"] == "Test Entry"

    def test_get_by_id(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        entry = PlaybookEntry(title="Find Me", content="Here", tags=["lang:python"])
        save_entries(engine, [build_entry_dict(entry)])

        entries = list_recent_entries(engine)
        entry_id = entries[0]["id"]

        result = get_entry_by_id(engine, entry_id)
        assert result is not None
        assert result["title"] == "Find Me"

    def test_get_by_id_missing(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        assert get_entry_by_id(engine, 999) is None

    def test_update_entry(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        entry = PlaybookEntry(title="Original", content="V1", tags=["lang:python"])
        save_entries(engine, [build_entry_dict(entry)])

        entries = list_recent_entries(engine)
        entry_id = entries[0]["id"]

        update_entry(engine, entry_id, {"title": "Updated", "content": "V2"})
        result = get_entry_by_id(engine, entry_id)
        assert result["title"] == "Updated"
        assert result["content"] == "V2"

    def test_delete_entry(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        entry = PlaybookEntry(title="Delete Me", content="Gone", tags=["lang:python"])
        save_entries(engine, [build_entry_dict(entry)])

        entries = list_recent_entries(engine)
        entry_id = entries[0]["id"]

        delete_entry(engine, entry_id)
        assert get_entry_by_id(engine, entry_id) is None

    def test_save_empty_list(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(engine, [])  # Should not raise
        assert list_recent_entries(engine) == []


# ---------------------------------------------------------------------------
# Tag-based queries
# ---------------------------------------------------------------------------


class TestTagQueries:
    def test_get_by_tags(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(engine, [
            build_entry_dict(PlaybookEntry(
                title="Python tip", content="Use type hints", tags=["lang:python"],
            )),
            build_entry_dict(PlaybookEntry(
                title="Go tip", content="Use interfaces", tags=["lang:go"],
            )),
        ])

        results = get_entries_by_tags(engine, ["lang:python"])
        assert len(results) == 1
        assert results[0]["title"] == "Python tip"

    def test_or_matching(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(engine, [
            build_entry_dict(PlaybookEntry(
                title="A", content="a", tags=["lang:python"],
            )),
            build_entry_dict(PlaybookEntry(
                title="B", content="b", tags=["lang:go"],
            )),
            build_entry_dict(PlaybookEntry(
                title="C", content="c", tags=["lang:rust"],
            )),
        ])

        results = get_entries_by_tags(engine, ["lang:python", "lang:go"])
        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert titles == {"A", "B"}

    def test_approved_only(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(engine, [
            build_entry_dict(PlaybookEntry(
                title="Reviewed", content="ok", tags=["lang:python"], needs_review=False,
            )),
            build_entry_dict(PlaybookEntry(
                title="Unreviewed", content="maybe", tags=["lang:python"], needs_review=True,
            )),
        ])

        all_results = get_entries_by_tags(engine, ["lang:python"])
        assert len(all_results) == 2

        approved = get_entries_by_tags(engine, ["lang:python"], approved_only=True)
        assert len(approved) == 1
        assert approved[0]["title"] == "Reviewed"

    def test_empty_tags(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        assert get_entries_by_tags(engine, []) == []


# ---------------------------------------------------------------------------
# Duplicate checking
# ---------------------------------------------------------------------------


class TestDuplicateChecking:
    def test_finds_duplicate(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(engine, [
            build_entry_dict(PlaybookEntry(
                title="Use dispose() in tests",
                content="SQLAlchemy engines cache by URL.",
                tags=["lib:sqlalchemy"],
            )),
        ])

        matches = check_duplicate(engine, "dispose")
        assert len(matches) == 1
        assert "dispose" in matches[0]["title"]

    def test_no_match(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(engine, [
            build_entry_dict(PlaybookEntry(
                title="Unrelated", content="Nothing to do", tags=["lang:go"],
            )),
        ])

        matches = check_duplicate(engine, "dispose")
        assert len(matches) == 0

    def test_tag_ordering(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(engine, [
            build_entry_dict(PlaybookEntry(
                title="SQLAlchemy tip A", content="a", tags=["lib:sqlalchemy", "domain:testing"],
            )),
            build_entry_dict(PlaybookEntry(
                title="SQLAlchemy tip B", content="b", tags=["lib:sqlalchemy"],
            )),
        ])

        matches = check_duplicate(engine, "SQLAlchemy", tags=["lib:sqlalchemy", "domain:testing"])
        assert len(matches) == 2
        # Entry with more tag overlap should come first
        assert matches[0]["title"] == "SQLAlchemy tip A"
