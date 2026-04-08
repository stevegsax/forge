"""Tests for pbook.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pbook.models import (
    ApiDocRecord,
    CapabilityTier,
    EntryType,
    ModelConfig,
    PlaybookEntry,
    PushExperienceInput,
    RetrievalInput,
    RetrievalMode,
    resolve_model,
)

# ---------------------------------------------------------------------------
# EntryType
# ---------------------------------------------------------------------------


class TestEntryType:
    def test_values(self):
        assert EntryType.PITFALL == "pitfall"
        assert EntryType.CURATED == "curated"
        assert EntryType.API_DOC == "api_doc"


# ---------------------------------------------------------------------------
# PlaybookEntry
# ---------------------------------------------------------------------------


class TestPlaybookEntry:
    def test_minimal_entry(self):
        entry = PlaybookEntry(title="Test", content="Content")
        assert entry.title == "Test"
        assert entry.content == "Content"
        assert entry.tags == []
        assert entry.entry_type == EntryType.CURATED
        assert entry.source_project == ""
        assert entry.needs_review is False

    def test_full_entry(self):
        entry = PlaybookEntry(
            title="Use dispose() in tests",
            content="SQLAlchemy create_engine caches by URL.",
            tags=["lib:sqlalchemy", "domain:testing"],
            entry_type=EntryType.PITFALL,
            source_project="forge",
            source_task_id="task-1",
            needs_review=True,
        )
        assert entry.entry_type == EntryType.PITFALL
        assert entry.source_project == "forge"
        assert entry.needs_review is True
        assert len(entry.tags) == 2

    def test_json_roundtrip(self):
        entry = PlaybookEntry(
            title="Test",
            content="Content",
            tags=["lang:python"],
        )
        json_str = entry.model_dump_json()
        restored = PlaybookEntry.model_validate_json(json_str)
        assert restored == entry


# ---------------------------------------------------------------------------
# ApiDocRecord
# ---------------------------------------------------------------------------


class TestApiDocRecord:
    def test_minimal(self):
        record = ApiDocRecord(
            library="sqlalchemy",
            method="sqlalchemy.create_engine",
            summary="Create a database engine.",
            signature="def create_engine(url: str, **kwargs) -> Engine",
        )
        assert record.library == "sqlalchemy"
        assert record.examples == []
        assert record.doc_url == ""

    def test_with_examples(self):
        record = ApiDocRecord(
            library="pydantic",
            method="pydantic.BaseModel",
            summary="Base class for data validation.",
            signature="class BaseModel",
            examples=["class User(BaseModel):\n    name: str"],
            doc_url="https://docs.pydantic.dev/latest/",
        )
        assert len(record.examples) == 1


# ---------------------------------------------------------------------------
# RetrievalInput
# ---------------------------------------------------------------------------


class TestRetrievalInput:
    def test_defaults(self):
        inp = RetrievalInput()
        assert inp.tags == []
        assert inp.mode == RetrievalMode.CREATE
        assert inp.token_budget == 5000
        assert inp.approved_only is False

    def test_fix_mode(self):
        inp = RetrievalInput(mode=RetrievalMode.FIX, approved_only=True)
        assert inp.mode == RetrievalMode.FIX
        assert inp.approved_only is True


# ---------------------------------------------------------------------------
# PushExperienceInput
# ---------------------------------------------------------------------------


class TestPushExperienceInput:
    def test_required_fields(self):
        inp = PushExperienceInput(
            project="forge",
            problem="Base64 prefix in OCR response",
            resolution="Strip data URI prefix before decoding",
        )
        assert inp.project == "forge"
        assert inp.context == ""
        assert inp.metadata == {}

    def test_validation_error(self):
        with pytest.raises(ValidationError):
            PushExperienceInput()  # Missing required fields


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------


class TestModelRouting:
    def test_default_config(self):
        config = ModelConfig()
        assert "opus" in config.reasoning
        assert "haiku" in config.classification

    def test_resolve_model(self):
        config = ModelConfig()
        model = resolve_model(CapabilityTier.CLASSIFICATION, config)
        assert "haiku" in model

    def test_resolve_custom(self):
        config = ModelConfig(classification="mistral:mistral-small-latest")
        model = resolve_model(CapabilityTier.CLASSIFICATION, config)
        assert model == "mistral:mistral-small-latest"
