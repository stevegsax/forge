"""Tests for the skill-prompt payload."""

from __future__ import annotations

import pytest

from pbook.skill_prompts import build_skill_prompt


class TestBuildSkillPrompt:
    def test_full_payload_shape(self):
        payload = build_skill_prompt()
        assert set(payload) == {"commands", "workflows", "tags"}

    def test_commands_cover_every_documented_subcommand(self):
        payload = build_skill_prompt()
        commands = payload["commands"]
        # The four user-facing capability bundles must each have at
        # least one corresponding command documented.
        assert "search" in commands       # query
        assert "list" in commands
        assert "get" in commands
        assert "sources" in commands      # discuss
        assert "session-text" in commands
        assert "approve" in commands      # review queue
        assert "reject" in commands
        assert "review" in commands
        assert "add" in commands          # add
        assert "tags" in commands

    def test_each_command_has_description_args_example(self):
        payload = build_skill_prompt()
        for name, info in payload["commands"].items():
            assert info.get("description"), f"{name} missing description"
            assert info.get("args"), f"{name} missing args"
            assert info.get("example"), f"{name} missing example"

    def test_workflows_cover_the_five_skill_capabilities(self):
        payload = build_skill_prompt()
        assert set(payload["workflows"]) == {
            "query", "discuss", "feedback", "review_queue", "add",
        }

    def test_each_workflow_has_nontrivial_markdown(self):
        payload = build_skill_prompt()
        for name, md in payload["workflows"].items():
            assert isinstance(md, str), f"{name} workflow should be a markdown string"
            assert len(md) > 200, f"{name} workflow looks too thin"
            assert md.startswith("##"), f"{name} workflow should start with a heading"

    def test_feedback_workflow_covers_load_bearing_rules(self):
        """The feedback workflow is the only place the skill learns
        when to record — these elements must be present so /skill-creator
        bakes them into SKILL.md, not paraphrased away."""
        payload = build_skill_prompt()
        md = payload["workflows"]["feedback"]
        # Confirm-before-recording is the cornerstone of the design.
        assert "confirm" in md.lower()
        # The skill must distinguish harmful (signal) from reject (hide).
        assert "--harmful" in md
        assert "reject" in md.lower()
        # The 3-retrieval threshold is non-obvious; the skill must surface it
        # so users don't expect immediate ranking shifts.
        assert "3" in md and "retriev" in md.lower()
        # Context capture is what makes the signal useful — the bare
        # counter is not.
        assert "--context" in md

    def test_workflow_markdown_uses_playbook_not_play(self):
        """Avoid the generic word 'play'; pbook entries are 'playbooks'."""
        import re

        payload = build_skill_prompt()
        # \bplay\b matches the bare word but not 'playbook'/'playbooks'/'replay'/'display'.
        bare_play = re.compile(r"\bplays?\b", re.IGNORECASE)
        for name, md in payload["workflows"].items():
            match = bare_play.search(md)
            assert match is None, (
                f"{name} workflow uses bare 'play(s)': "
                f"{md[max(0, match.start()-20):match.end()+20]!r}"
            )

    def test_tags_section_includes_namespaces_and_notes(self):
        payload = build_skill_prompt()
        tags = payload["tags"]
        assert "lang" in tags["namespaces"]["general"]
        assert "project" in tags["namespaces"]["extracted"]
        assert "namespaced" in tags["notes"].lower()

    def test_operation_filter_returns_single_workflow(self):
        payload = build_skill_prompt(operation="discuss")
        assert "workflow" in payload
        assert "workflows" not in payload  # no full set
        assert "discuss" not in payload  # the workflow is the value, not the key
        assert "## Discuss workflow" in payload["workflow"]

    def test_unknown_operation_raises_keyerror(self):
        with pytest.raises(KeyError) as excinfo:
            build_skill_prompt(operation="not-a-thing")
        # Error message should hint at the available names.
        assert "Available" in str(excinfo.value)
