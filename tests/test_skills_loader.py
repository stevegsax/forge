"""Tests for Agent Skills loader (Phase 15)."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.skills.loader import (
    build_skill_specs,
    discover_skills,
    load_skill_references,
    parse_frontmatter,
    parse_skill_md,
)
from forge.models import SkillDefinition


class TestParseFrontmatter:
    def test_basic_frontmatter(self):
        text = "---\nname: test-skill\ndescription: A test skill\n---\nBody content"
        metadata, body = parse_frontmatter(text)
        assert metadata["name"] == "test-skill"
        assert metadata["description"] == "A test skill"
        assert body == "Body content"

    def test_no_frontmatter(self):
        text = "Just regular content"
        metadata, body = parse_frontmatter(text)
        assert metadata == {}
        assert body == "Just regular content"

    def test_quoted_values(self):
        text = '---\nname: "quoted-name"\ndescription: \'single quoted\'\n---\nBody'
        metadata, body = parse_frontmatter(text)
        assert metadata["name"] == "quoted-name"
        assert metadata["description"] == "single quoted"

    def test_empty_body(self):
        text = "---\nname: test\ndescription: desc\n---\n"
        metadata, body = parse_frontmatter(text)
        assert metadata["name"] == "test"
        assert body == ""


class TestParseSkillMd:
    def test_valid_skill(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Does something useful\n---\n"
            "## Instructions\n1. Do this\n2. Do that"
        )

        skill = parse_skill_md(skill_dir)
        assert skill is not None
        assert skill.name == "my-skill"
        assert skill.description == "Does something useful"
        assert "Do this" in skill.instructions
        assert skill.skill_path == str(skill_dir)

    def test_missing_skill_md(self, tmp_path):
        skill_dir = tmp_path / "empty-skill"
        skill_dir.mkdir()
        assert parse_skill_md(skill_dir) is None

    def test_missing_name(self, tmp_path):
        skill_dir = tmp_path / "no-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: Has desc but no name\n---\nBody"
        )
        assert parse_skill_md(skill_dir) is None

    def test_missing_description(self, tmp_path):
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: has-name\n---\nBody")
        assert parse_skill_md(skill_dir) is None

    def test_allowed_tools(self, tmp_path):
        skill_dir = tmp_path / "with-tools"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: tool-skill\ndescription: Has tools\n"
            "allowed-tools: bash python\n---\nBody"
        )
        skill = parse_skill_md(skill_dir)
        assert skill is not None
        assert skill.allowed_tools == ["bash", "python"]


class TestDiscoverSkills:
    def test_discover_multiple_skills(self, tmp_path):
        for name in ["skill-a", "skill-b"]:
            skill_dir = tmp_path / name
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: Skill {name}\n---\nInstructions for {name}"
            )

        skills = discover_skills([str(tmp_path)])
        assert len(skills) == 2
        assert "skill_skill-a" in skills
        assert "skill_skill-b" in skills

    def test_skip_non_skill_dirs(self, tmp_path):
        # A directory without SKILL.md
        (tmp_path / "not-a-skill").mkdir()
        # A regular file
        (tmp_path / "some-file.txt").write_text("not a skill")
        # A hidden directory
        hidden = tmp_path / ".hidden-skill"
        hidden.mkdir()
        (hidden / "SKILL.md").write_text("---\nname: hidden\ndescription: d\n---\nBody")

        skills = discover_skills([str(tmp_path)])
        assert len(skills) == 0

    def test_nonexistent_dir(self):
        skills = discover_skills(["/nonexistent/path"])
        assert skills == {}

    def test_duplicate_names(self, tmp_path):
        dir_a = tmp_path / "dir-a" / "same-skill"
        dir_a.mkdir(parents=True)
        (dir_a / "SKILL.md").write_text(
            "---\nname: dupe\ndescription: First\n---\nFirst body"
        )
        dir_b = tmp_path / "dir-b" / "same-skill"
        dir_b.mkdir(parents=True)
        (dir_b / "SKILL.md").write_text(
            "---\nname: dupe\ndescription: Second\n---\nSecond body"
        )

        skills = discover_skills([str(tmp_path / "dir-a"), str(tmp_path / "dir-b")])
        # First one wins
        assert len(skills) == 1
        assert skills["skill_dupe"].description == "First"


class TestBuildSkillSpecs:
    def test_basic_specs(self):
        skills = {
            "skill_test": SkillDefinition(
                name="test",
                description="A test skill",
                instructions="Do stuff",
                skill_path="/tmp/test",
            )
        }
        specs = build_skill_specs(skills)
        assert len(specs) == 1
        assert specs[0].name == "skill_test"
        assert "[Skill]" in specs[0].description
        assert "request" in specs[0].parameters


class TestLoadSkillReferences:
    def test_load_references(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        refs_dir = skill_dir / "references"
        refs_dir.mkdir(parents=True)
        (refs_dir / "schema.json").write_text('{"type": "object"}')
        (refs_dir / "example.md").write_text("# Example")

        skill = SkillDefinition(
            name="test",
            description="test",
            instructions="test",
            skill_path=str(skill_dir),
        )
        refs = load_skill_references(skill)
        assert len(refs) == 2
        assert "schema.json" in refs
        assert "example.md" in refs

    def test_no_references_dir(self, tmp_path):
        skill = SkillDefinition(
            name="test",
            description="test",
            instructions="test",
            skill_path=str(tmp_path),
        )
        refs = load_skill_references(skill)
        assert refs == {}
