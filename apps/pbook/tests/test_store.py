"""Tests for pbook.store."""

from __future__ import annotations

from pbook.models import EntryType, PlaybookEntry
from pbook.settings import PbookDbSettings
from pbook.store import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_ERROR,
    SESSION_STATUS_RUNNING,
    SOURCE_DEDUP_THRESHOLD,
    add_entry_source,
    build_engine,
    build_entry_dict,
    check_duplicate,
    delete_entry,
    find_similar_source_contexts,
    get_entries_by_tags,
    get_entry_by_id,
    get_ingested_session_ids,
    insert_entry,
    list_entry_sources_for_entry,
    list_ingested_sessions,
    list_recent_entries,
    normalize_url,
    record_feedback,
    record_ingested_session,
    record_ingested_session_error,
    record_ingested_session_started,
    record_retrieval,
    reparent_entry_sources,
    save_entries,
    update_entry,
)
from tests.conftest import make_embedding, setup_db

# ---------------------------------------------------------------------------
# normalize_url
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    def test_normalize_url_variants(self):
        assert normalize_url("postgres://u@h/db") == "postgresql+psycopg://u@h/db"
        assert normalize_url("postgresql://u@h/db") == "postgresql+psycopg://u@h/db"
        # Already-qualified URLs pass through untouched.
        assert normalize_url("postgresql+psycopg://u@h/db") == "postgresql+psycopg://u@h/db"


# ---------------------------------------------------------------------------
# build_engine — store-disabled semantics + normalization
# ---------------------------------------------------------------------------


class TestBuildEngine:
    def test_none_url_returns_none(self):
        assert build_engine(PbookDbSettings(url=None)) is None

    def test_empty_url_returns_none(self):
        assert build_engine(PbookDbSettings(url="")) is None

    def test_builds_normalized_engine(self):
        # No connection is made here — just construction + URL normalization.
        engine = build_engine(PbookDbSettings(url="postgresql://u:p@host:5432/db"))
        assert engine is not None
        try:
            assert engine.url.drivername == "postgresql+psycopg"
        finally:
            engine.dispose()


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
        assert d["tags"] == ["lang:python"]
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
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="Python tip",
                        content="Use type hints",
                        tags=["lang:python"],
                    )
                ),
                build_entry_dict(
                    PlaybookEntry(
                        title="Go tip",
                        content="Use interfaces",
                        tags=["lang:go"],
                    )
                ),
            ],
        )

        results = get_entries_by_tags(engine, ["lang:python"])
        assert len(results) == 1
        assert results[0]["title"] == "Python tip"

    def test_or_matching(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="A",
                        content="a",
                        tags=["lang:python"],
                    )
                ),
                build_entry_dict(
                    PlaybookEntry(
                        title="B",
                        content="b",
                        tags=["lang:go"],
                    )
                ),
                build_entry_dict(
                    PlaybookEntry(
                        title="C",
                        content="c",
                        tags=["lang:rust"],
                    )
                ),
            ],
        )

        results = get_entries_by_tags(engine, ["lang:python", "lang:go"])
        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert titles == {"A", "B"}

    def test_approved_only(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="Reviewed",
                        content="ok",
                        tags=["lang:python"],
                        needs_review=False,
                    )
                ),
                build_entry_dict(
                    PlaybookEntry(
                        title="Unreviewed",
                        content="maybe",
                        tags=["lang:python"],
                        needs_review=True,
                    )
                ),
            ],
        )

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
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="Use dispose() in tests",
                        content="SQLAlchemy engines cache by URL.",
                        tags=["lib:sqlalchemy"],
                    )
                ),
            ],
        )

        matches = check_duplicate(engine, "dispose")
        assert len(matches) == 1
        assert "dispose" in matches[0]["title"]

    def test_no_match(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="Unrelated",
                        content="Nothing to do",
                        tags=["lang:go"],
                    )
                ),
            ],
        )

        matches = check_duplicate(engine, "dispose")
        assert len(matches) == 0

    def test_tag_ordering(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="SQLAlchemy tip A",
                        content="a",
                        tags=["lib:sqlalchemy", "domain:testing"],
                    )
                ),
                build_entry_dict(
                    PlaybookEntry(
                        title="SQLAlchemy tip B",
                        content="b",
                        tags=["lib:sqlalchemy"],
                    )
                ),
            ],
        )

        matches = check_duplicate(engine, "SQLAlchemy", tags=["lib:sqlalchemy", "domain:testing"])
        assert len(matches) == 2
        # Entry with more tag overlap should come first
        assert matches[0]["title"] == "SQLAlchemy tip A"


# ---------------------------------------------------------------------------
# Feedback counters
# ---------------------------------------------------------------------------


class TestFeedbackCounters:
    def test_new_entry_has_zero_counters(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(PlaybookEntry(title="T", content="C", tags=["lang:python"])),
            ],
        )
        entry = list_recent_entries(engine)[0]
        assert entry["helpful_count"] == 0
        assert entry["harmful_count"] == 0
        assert entry["retrieval_count"] == 0

    def test_record_retrieval_increments(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(PlaybookEntry(title="T", content="C", tags=["lang:python"])),
            ],
        )
        entry_id = list_recent_entries(engine)[0]["id"]

        record_retrieval(engine, [entry_id])
        assert get_entry_by_id(engine, entry_id)["retrieval_count"] == 1

        record_retrieval(engine, [entry_id])
        assert get_entry_by_id(engine, entry_id)["retrieval_count"] == 2

    def test_record_retrieval_bulk(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(PlaybookEntry(title="A", content="a", tags=["lang:python"])),
                build_entry_dict(PlaybookEntry(title="B", content="b", tags=["lang:python"])),
            ],
        )
        entries = list_recent_entries(engine)
        ids = [e["id"] for e in entries]

        record_retrieval(engine, ids)
        for eid in ids:
            assert get_entry_by_id(engine, eid)["retrieval_count"] == 1

    def test_record_retrieval_empty_list(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        record_retrieval(engine, [])  # Should not raise

    def test_record_feedback_helpful(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(PlaybookEntry(title="T", content="C", tags=["lang:python"])),
            ],
        )
        entry_id = list_recent_entries(engine)[0]["id"]

        record_feedback(engine, entry_id, helpful=True)
        entry = get_entry_by_id(engine, entry_id)
        assert entry["helpful_count"] == 1
        assert entry["harmful_count"] == 0

    def test_record_feedback_harmful(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(PlaybookEntry(title="T", content="C", tags=["lang:python"])),
            ],
        )
        entry_id = list_recent_entries(engine)[0]["id"]

        record_feedback(engine, entry_id, helpful=False)
        entry = get_entry_by_id(engine, entry_id)
        assert entry["helpful_count"] == 0
        assert entry["harmful_count"] == 1

    def test_record_feedback_accumulates(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(PlaybookEntry(title="T", content="C", tags=["lang:python"])),
            ],
        )
        entry_id = list_recent_entries(engine)[0]["id"]

        record_feedback(engine, entry_id, helpful=True)
        record_feedback(engine, entry_id, helpful=True)
        record_feedback(engine, entry_id, helpful=False)

        entry = get_entry_by_id(engine, entry_id)
        assert entry["helpful_count"] == 2
        assert entry["harmful_count"] == 1

    def test_record_feedback_missing_entry(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        # Should not raise — UPDATE on non-existent row is a no-op
        record_feedback(engine, 999, helpful=True)


# ---------------------------------------------------------------------------
# Ingested sessions
# ---------------------------------------------------------------------------


class TestListIngestedSessions:
    def test_returns_recorded_sessions_newest_first(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        record_ingested_session(
            engine,
            "s1",
            project_name="alpha",
            experiences_found=2,
            entries_created=1,
        )
        record_ingested_session(
            engine,
            "s2",
            project_name="bravo",
            experiences_found=0,
            entries_created=0,
        )

        rows = list_ingested_sessions(engine)
        assert {r["session_id"] for r in rows} == {"s1", "s2"}
        # Newest first
        assert rows[0]["session_id"] == "s2"
        # Counters preserved
        s1 = next(r for r in rows if r["session_id"] == "s1")
        assert s1["experiences_found"] == 2
        assert s1["entries_created"] == 1
        assert s1["project_name"] == "alpha"

    def test_filters_by_project(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        record_ingested_session(engine, "s1", project_name="alpha")
        record_ingested_session(engine, "s2", project_name="bravo")
        record_ingested_session(engine, "s3", project_name="alpha")

        rows = list_ingested_sessions(engine, project="alpha")
        assert {r["session_id"] for r in rows} == {"s1", "s3"}

    def test_respects_limit(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        for i in range(5):
            record_ingested_session(engine, f"s{i}", project_name="x")

        rows = list_ingested_sessions(engine, limit=2)
        assert len(rows) == 2

    def test_empty_when_no_sessions(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        assert list_ingested_sessions(engine) == []

    def test_completed_record_has_completed_status(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        record_ingested_session(engine, "s1", project_name="alpha")

        rows = list_ingested_sessions(engine)
        assert rows[0]["status"] == SESSION_STATUS_COMPLETED


class TestRecordIngestedSessionStarted:
    def test_seeds_running_row(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        record_ingested_session_started(
            engine,
            "s1",
            project_name="alpha",
            workflow_id="wf-1",
            run_id="run-1",
        )

        rows = list_ingested_sessions(engine)
        assert len(rows) == 1
        row = rows[0]
        assert row["status"] == SESSION_STATUS_RUNNING
        assert row["workflow_id"] == "wf-1"
        assert row["run_id"] == "run-1"
        assert row["error_message"] is None
        assert row["started_at"] is not None

    def test_completion_callback_flips_running_to_completed(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        record_ingested_session_started(
            engine,
            "s1",
            project_name="alpha",
            workflow_id="wf-1",
            run_id="run-1",
        )
        record_ingested_session(
            engine,
            "s1",
            project_name="alpha",
            experiences_found=3,
            entries_created=2,
        )

        rows = list_ingested_sessions(engine)
        assert len(rows) == 1
        row = rows[0]
        assert row["status"] == SESSION_STATUS_COMPLETED
        assert row["experiences_found"] == 3
        assert row["entries_created"] == 2
        # Workflow ids from the running row are preserved.
        assert row["workflow_id"] == "wf-1"
        assert row["run_id"] == "run-1"

    def test_re_seeding_clears_prior_error(self, tmp_path):
        """`pbook ingest --force` retries an errored session — the new run
        should reset status and clear the old error message."""
        engine, _ = setup_db(tmp_path)
        record_ingested_session_started(
            engine,
            "s1",
            project_name="alpha",
            workflow_id="wf-1",
            run_id="run-1",
        )
        record_ingested_session_error(engine, "s1", "boom")

        record_ingested_session_started(
            engine,
            "s1",
            project_name="alpha",
            workflow_id="wf-2",
            run_id="run-2",
        )

        rows = list_ingested_sessions(engine)
        row = rows[0]
        assert row["status"] == SESSION_STATUS_RUNNING
        assert row["error_message"] is None
        assert row["workflow_id"] == "wf-2"


class TestRecordIngestedSessionError:
    def test_flips_running_row_to_error(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        record_ingested_session_started(
            engine,
            "s1",
            project_name="alpha",
            workflow_id="wf-1",
            run_id="run-1",
        )
        record_ingested_session_error(engine, "s1", "malformed_llm_response")

        rows = list_ingested_sessions(engine)
        row = rows[0]
        assert row["status"] == SESSION_STATUS_ERROR
        assert row["error_message"] == "malformed_llm_response"

    def test_seeds_row_when_no_prior_running_record(self, tmp_path):
        """Failure callback must succeed even if the started callback
        never ran (e.g. workflow blew up before the seed)."""
        engine, _ = setup_db(tmp_path)
        record_ingested_session_error(
            engine,
            "s1",
            "boom",
            project_name="alpha",
        )

        rows = list_ingested_sessions(engine)
        row = rows[0]
        assert row["status"] == SESSION_STATUS_ERROR
        assert row["error_message"] == "boom"
        assert row["project_name"] == "alpha"


class TestGetIngestedSessionIds:
    def test_includes_running_and_completed_excludes_error(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        record_ingested_session(engine, "done", project_name="x")
        record_ingested_session_started(engine, "live", project_name="x")
        record_ingested_session_started(engine, "broken", project_name="x")
        record_ingested_session_error(engine, "broken", "boom")

        ids = get_ingested_session_ids(engine)
        assert ids == {"done", "live"}


# ---------------------------------------------------------------------------
# entry_sources
# ---------------------------------------------------------------------------


def _seed_entries(engine, n: int = 2) -> list[int]:
    """Insert N minimal entries and return their ids."""
    ids: list[int] = []
    for i in range(n):
        entry = PlaybookEntry(
            title=f"Entry {i}",
            content=f"Body {i}",
            tags=["lang:python"],
        )
        ids.append(insert_entry(engine, build_entry_dict(entry)))
    return ids


def _vec(*coords: float) -> list[float]:
    """Build a full-width float vector (for similarity testing)."""
    return make_embedding(*coords)


class TestInsertEntry:
    def test_returns_new_id(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        entry = PlaybookEntry(title="X", content="Y", tags=[])
        new_id = insert_entry(engine, build_entry_dict(entry))
        assert isinstance(new_id, int) and new_id > 0
        rows = list_recent_entries(engine)
        assert any(r["id"] == new_id for r in rows)

    def test_persists_tags(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        entry = PlaybookEntry(title="X", content="Y", tags=["lang:python", "lib:pytest"])
        new_id = insert_entry(engine, build_entry_dict(entry))
        row = get_entry_by_id(engine, new_id)
        assert sorted(row["tags"]) == ["lang:python", "lib:pytest"]


class TestAddEntrySource:
    def test_inserts_and_returns_id(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        [eid] = _seed_entries(engine, 1)
        sid = add_entry_source(
            engine,
            entry_id=eid,
            session_id="sess",
            project_name="proj",
            experience_hash="h1",
            source_context="why",
            source_context_embedding=_vec(1.0, 0.0),
        )
        assert sid is not None and sid > 0

    def test_unique_conflict_is_no_op(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        [eid] = _seed_entries(engine, 1)
        first = add_entry_source(
            engine,
            entry_id=eid,
            session_id="s",
            experience_hash="h",
            source_context="a",
        )
        second = add_entry_source(
            engine,
            entry_id=eid,
            session_id="s",
            experience_hash="h",
            source_context="b",  # different content but same key triplet
        )
        assert first is not None
        assert second is None
        rows = list_entry_sources_for_entry(engine, eid)
        assert len(rows) == 1
        # The original content survives — DO NOTHING preserves the first row.
        assert rows[0]["source_context"] == "a"


class TestFindSimilarSourceContexts:
    def test_threshold_excludes_dissimilar(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        [eid] = _seed_entries(engine, 1)
        add_entry_source(
            engine,
            entry_id=eid,
            session_id="s1",
            experience_hash="h1",
            source_context_embedding=_vec(1.0, 0.0),
        )
        # Orthogonal vector → cosine 0 → below threshold.
        out = find_similar_source_contexts(
            engine,
            eid,
            _vec(0.0, 1.0),
            threshold=SOURCE_DEDUP_THRESHOLD,
        )
        assert out == []

    def test_threshold_includes_near_identical(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        [eid] = _seed_entries(engine, 1)
        add_entry_source(
            engine,
            entry_id=eid,
            session_id="s1",
            experience_hash="h1",
            source_context_embedding=_vec(1.0, 0.0),
        )
        out = find_similar_source_contexts(
            engine,
            eid,
            _vec(1.0, 0.0001),
        )
        assert len(out) == 1


class TestReparentEntrySources:
    def test_moves_rows_to_target(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        a, b = _seed_entries(engine, 2)
        add_entry_source(engine, entry_id=a, session_id="s1", experience_hash="h1")
        add_entry_source(engine, entry_id=a, session_id="s2", experience_hash="h2")
        moved = reparent_entry_sources(engine, from_entry_ids=[a], to_entry_id=b)
        assert moved == 2
        assert list_entry_sources_for_entry(engine, a) == []
        rows_b = list_entry_sources_for_entry(engine, b)
        assert {r["session_id"] for r in rows_b} == {"s1", "s2"}

    def test_collision_drops_losing_row(self, tmp_path):
        """If both source and target hold the same (session_id, hash),
        the source's row is deleted instead of moved."""
        engine, _ = setup_db(tmp_path)
        a, b = _seed_entries(engine, 2)
        add_entry_source(
            engine, entry_id=a, session_id="s", experience_hash="h", source_context="from-a"
        )
        add_entry_source(
            engine, entry_id=b, session_id="s", experience_hash="h", source_context="from-b"
        )
        reparent_entry_sources(engine, from_entry_ids=[a], to_entry_id=b)
        rows_b = list_entry_sources_for_entry(engine, b)
        assert len(rows_b) == 1
        # The pre-existing target row survives; the colliding source row is deleted.
        assert rows_b[0]["source_context"] == "from-b"

    def test_empty_from_list_is_zero(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        [a] = _seed_entries(engine, 1)
        assert reparent_entry_sources(engine, from_entry_ids=[], to_entry_id=a) == 0


class TestEntrySourceCascade:
    def test_deleting_entry_drops_its_sources(self, tmp_path):
        engine, _ = setup_db(tmp_path)
        [eid] = _seed_entries(engine, 1)
        add_entry_source(engine, entry_id=eid, session_id="s", experience_hash="h")
        assert len(list_entry_sources_for_entry(engine, eid)) == 1
        delete_entry(engine, eid)
        assert list_entry_sources_for_entry(engine, eid) == []


# ---------------------------------------------------------------------------
# Soft-rejection
# ---------------------------------------------------------------------------


class TestMarkRejected:
    def test_persists_flag_and_reason(self, tmp_path):
        from pbook.store import mark_rejected

        engine, _ = setup_db(tmp_path)
        [eid] = _seed_entries(engine, 1)

        mark_rejected(engine, eid, reason="wrong project")

        row = get_entry_by_id(engine, eid)
        assert row is not None
        assert row["rejected"] is True
        assert row["rejection_reason"] == "wrong project"

    def test_default_reason_is_none(self, tmp_path):
        from pbook.store import mark_rejected

        engine, _ = setup_db(tmp_path)
        [eid] = _seed_entries(engine, 1)
        mark_rejected(engine, eid)

        row = get_entry_by_id(engine, eid)
        assert row is not None
        assert row["rejected"] is True
        assert row["rejection_reason"] is None


class TestRejectedFiltering:
    """Default queries hide rejected entries; include_rejected surfaces them."""

    def test_list_recent_entries_excludes_rejected_by_default(self, tmp_path):
        from pbook.store import mark_rejected

        engine, _ = setup_db(tmp_path)
        ids = _seed_entries(engine, 3)
        mark_rejected(engine, ids[0])

        rows = list_recent_entries(engine)
        surviving_ids = {r["id"] for r in rows}
        assert ids[0] not in surviving_ids
        assert ids[1] in surviving_ids and ids[2] in surviving_ids

    def test_list_recent_entries_include_rejected(self, tmp_path):
        from pbook.store import mark_rejected

        engine, _ = setup_db(tmp_path)
        ids = _seed_entries(engine, 2)
        mark_rejected(engine, ids[0])

        rows = list_recent_entries(engine, include_rejected=True)
        assert {r["id"] for r in rows} == set(ids)

    def test_get_entries_by_tags_excludes_rejected_by_default(self, tmp_path):
        from pbook.store import get_entries_by_tags, mark_rejected

        engine, _ = setup_db(tmp_path)
        ids = _seed_entries(engine, 2)
        mark_rejected(engine, ids[0])

        rows = get_entries_by_tags(engine, ["lang:python"])
        surviving_ids = {r["id"] for r in rows}
        assert ids[0] not in surviving_ids

    def test_get_entries_by_tags_include_rejected(self, tmp_path):
        from pbook.store import get_entries_by_tags, mark_rejected

        engine, _ = setup_db(tmp_path)
        ids = _seed_entries(engine, 2)
        mark_rejected(engine, ids[0])

        rows = get_entries_by_tags(
            engine,
            ["lang:python"],
            include_rejected=True,
        )
        assert {r["id"] for r in rows} == set(ids)


class TestListTagValuesInUse:
    def test_groups_by_namespace(self, tmp_path):
        from pbook.store import list_tag_values_in_use

        engine, _ = setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="A",
                        content="x",
                        tags=["lang:python", "lib:pytest", "domain:testing"],
                    )
                ),
                build_entry_dict(
                    PlaybookEntry(
                        title="B",
                        content="y",
                        tags=["lang:go", "lib:cobra"],
                    )
                ),
            ],
        )

        result = list_tag_values_in_use(engine)
        assert result["lang"] == ["go", "python"]
        assert result["lib"] == ["cobra", "pytest"]
        assert result["domain"] == ["testing"]
        # Empty namespaces return empty lists, not missing keys.
        assert result["project"] == []
        assert result["pattern"] == []

    def test_excludes_rejected_entries(self, tmp_path):
        from pbook.store import list_tag_values_in_use, mark_rejected

        engine, _ = setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="kept",
                        content="x",
                        tags=["lang:python"],
                    )
                ),
                build_entry_dict(
                    PlaybookEntry(
                        title="dropped",
                        content="x",
                        tags=["lang:elixir"],
                    )
                ),
            ],
        )
        # The "dropped" entry's id depends on insert order — find it.
        rejected_id = next(
            r["id"]
            for r in list_recent_entries(engine, include_rejected=True)
            if r["title"] == "dropped"
        )
        mark_rejected(engine, rejected_id)

        result = list_tag_values_in_use(engine)
        assert "python" in result["lang"]
        assert "elixir" not in result["lang"]

    def test_handles_malformed_tags_gracefully(self, tmp_path):
        from sqlalchemy import text

        from pbook.store import list_tag_values_in_use

        engine, _ = setup_db(tmp_path)
        save_entries(
            engine,
            [
                build_entry_dict(
                    PlaybookEntry(
                        title="A",
                        content="x",
                        tags=["lang:python"],
                    )
                ),
            ],
        )
        # Direct DB poke to introduce malformed tag rows (no colon, empty
        # value) the way a historical bad row might look.
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO pbook.pbk_entry_tags (entry_id, tag) "
                    "VALUES (1, 'malformed'), (1, 'x:')",
                )
            )

        result = list_tag_values_in_use(engine)
        # No crash; the well-formed namespaces are still correct.
        assert all(isinstance(v, list) for v in result.values())
        assert result["lang"] == ["python"]
