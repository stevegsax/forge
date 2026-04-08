"""Tests for pbook.tags."""

from __future__ import annotations

import pytest

from pbook.tags import (
    EXTRACTED_NAMESPACES,
    GENERAL_NAMESPACES,
    VALID_NAMESPACES,
    infer_tags_from_context,
    is_extracted_tag,
    is_general_tag,
    parse_tag,
    validate_tag,
    validate_tags,
)

# ---------------------------------------------------------------------------
# parse_tag
# ---------------------------------------------------------------------------


class TestParseTag:
    def test_valid_tag(self):
        assert parse_tag("lang:python") == ("lang", "python")
        assert parse_tag("lib:sqlalchemy") == ("lib", "sqlalchemy")
        assert parse_tag("domain:testing") == ("domain", "testing")
        assert parse_tag("project:forge") == ("project", "forge")
        assert parse_tag("pattern:failure-pattern") == ("pattern", "failure-pattern")

    def test_missing_colon(self):
        with pytest.raises(ValueError, match="namespace:value"):
            parse_tag("python")

    def test_empty_value(self):
        with pytest.raises(ValueError, match="must not be empty"):
            parse_tag("lang:")

    def test_unknown_namespace(self):
        with pytest.raises(ValueError, match="Unknown namespace"):
            parse_tag("error:type-error")


# ---------------------------------------------------------------------------
# validate_tag / validate_tags
# ---------------------------------------------------------------------------


class TestValidateTag:
    def test_valid(self):
        assert validate_tag("lang:python") is True
        assert validate_tag("lib:pydantic") is True

    def test_invalid(self):
        assert validate_tag("python") is False
        assert validate_tag("error:crash") is False

    def test_validate_tags_all_valid(self):
        errors = validate_tags(["lang:python", "lib:sqlalchemy"])
        assert errors == []

    def test_validate_tags_mixed(self):
        errors = validate_tags(["lang:python", "bad-tag", "error:crash"])
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# is_general_tag / is_extracted_tag
# ---------------------------------------------------------------------------


class TestTagClassification:
    def test_general_tags(self):
        assert is_general_tag("lang:python") is True
        assert is_general_tag("lib:sqlalchemy") is True
        assert is_general_tag("domain:testing") is True

    def test_extracted_tags(self):
        assert is_extracted_tag("project:forge") is True
        assert is_extracted_tag("pattern:retry-pattern") is True

    def test_cross_classification(self):
        assert is_general_tag("project:forge") is False
        assert is_extracted_tag("lang:python") is False

    def test_invalid_tag(self):
        assert is_general_tag("bad") is False
        assert is_extracted_tag("bad") is False


# ---------------------------------------------------------------------------
# Namespace sets
# ---------------------------------------------------------------------------


class TestNamespaceSets:
    def test_all_namespaces(self):
        assert {"lang", "lib", "domain", "project", "pattern"} == VALID_NAMESPACES

    def test_general_namespaces(self):
        assert {"lang", "lib", "domain"} == GENERAL_NAMESPACES

    def test_extracted_namespaces(self):
        assert {"project", "pattern"} == EXTRACTED_NAMESPACES

    def test_no_overlap(self):
        assert set() == GENERAL_NAMESPACES & EXTRACTED_NAMESPACES

    def test_complete(self):
        assert GENERAL_NAMESPACES | EXTRACTED_NAMESPACES == VALID_NAMESPACES


# ---------------------------------------------------------------------------
# infer_tags_from_context
# ---------------------------------------------------------------------------


class TestInferTagsFromContext:
    def test_python_extension(self):
        tags = infer_tags_from_context(file_extensions=[".py"])
        assert tags == ["lang:python"]

    def test_multiple_extensions(self):
        tags = infer_tags_from_context(file_extensions=[".py", ".ts"])
        assert "lang:python" in tags
        assert "lang:typescript" in tags

    def test_description_keywords(self):
        tags = infer_tags_from_context(description="Fix the test for the API")
        assert "domain:bug-fix" in tags
        assert "domain:testing" in tags
        assert "domain:api" in tags

    def test_empty_input(self):
        tags = infer_tags_from_context()
        assert tags == []

    def test_deduplication(self):
        tags = infer_tags_from_context(file_extensions=[".py", ".py"])
        assert tags == ["lang:python"]
