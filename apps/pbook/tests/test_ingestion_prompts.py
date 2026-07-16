"""Tests for ingestion prompt construction."""

from __future__ import annotations

from pbook.ingestion_prompts import (
    AnalyzedExperience,
    TranscriptAnalysisResult,
    build_analysis_system_prompt,
    build_analysis_user_prompt,
)


class TestBuildAnalysisSystemPrompt:
    def test_contains_quality_bar(self):
        prompt = build_analysis_system_prompt()
        assert "NOTHING" in prompt
        assert "misleading" in prompt
        assert "Quality over quantity" in prompt

    def test_contains_extraction_signals(self):
        prompt = build_analysis_system_prompt()
        assert "retry" in prompt.lower() or "retries" in prompt.lower()
        assert "workaround" in prompt.lower()
        assert "error" in prompt.lower()

    def test_contains_skip_rules(self):
        prompt = build_analysis_system_prompt()
        assert "routine" in prompt.lower()
        assert "generic" in prompt.lower()

    def test_specifies_output_format(self):
        prompt = build_analysis_system_prompt()
        assert "problem" in prompt
        assert "resolution" in prompt
        assert "context" in prompt


class TestBuildAnalysisUserPrompt:
    def test_includes_transcript(self):
        transcript = "USER: fix the bug\nASSISTANT: I found it"
        prompt = build_analysis_user_prompt(transcript, "myapp")
        assert transcript in prompt

    def test_includes_project_name(self):
        prompt = build_analysis_user_prompt("content", "forge")
        assert "forge" in prompt

    def test_includes_quality_reminder(self):
        prompt = build_analysis_user_prompt("content", "proj")
        assert "quality bar" in prompt.lower() or "empty list" in prompt.lower()


class TestTranscriptAnalysisResult:
    def test_empty_result(self):
        result = TranscriptAnalysisResult()
        assert result.experiences == []

    def test_with_experiences(self):
        result = TranscriptAnalysisResult(
            experiences=[
                AnalyzedExperience(
                    problem="SQLite locked",
                    resolution="Use WAL mode",
                    context="Multi-threaded Temporal activities",
                ),
            ]
        )
        assert len(result.experiences) == 1
        assert result.experiences[0].problem == "SQLite locked"

    def test_serialization_roundtrip(self):
        result = TranscriptAnalysisResult(
            experiences=[
                AnalyzedExperience(problem="p", resolution="r"),
            ]
        )
        json_str = result.model_dump_json()
        restored = TranscriptAnalysisResult.model_validate_json(json_str)
        assert restored.experiences[0].problem == "p"
