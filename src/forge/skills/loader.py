"""Agent Skills loader — SKILL.md parser and discovery (Phase 15).

Parses SKILL.md files following the Agent Skills specification
(agentskills.io). Supports YAML frontmatter with name, description,
and optional fields.

Design follows Function Core / Imperative Shell:
- Pure functions: parse_frontmatter, parse_skill_md, build_skill_specs
- I/O shell: discover_skills, load_skill_references
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from forge.models import ContextProviderSpec, SkillDefinition

logger = logging.getLogger(__name__)

# Match YAML frontmatter delimited by --- lines
_FRONTMATTER_PATTERN = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n(.*)",
    re.DOTALL,
)

# Simple YAML key-value parser for frontmatter (handles strings and lists)
_YAML_LINE_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$")


def _make_capability_name(skill_name: str) -> str:
    """Build a unique capability name for an Agent Skill.

    Format: skill_<name> to avoid collisions with providers.
    """
    return f"skill_{skill_name}"


def parse_frontmatter(text: str) -> tuple[dict[str, str | list[str]], str]:
    """Parse YAML frontmatter and return (metadata, body).

    Uses a simple line-based parser rather than requiring pyyaml.
    Handles:
    - Simple key: value pairs
    - Quoted strings
    - Space-delimited lists (for allowed-tools)

    Returns empty metadata and full text as body if no frontmatter found.
    """
    match = _FRONTMATTER_PATTERN.match(text)
    if not match:
        return {}, text

    frontmatter_text = match.group(1)
    body = match.group(2)

    metadata: dict[str, str | list[str]] = {}
    for line in frontmatter_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        kv_match = _YAML_LINE_PATTERN.match(line)
        if kv_match:
            key = kv_match.group(1).strip()
            value = kv_match.group(2).strip()
            # Remove surrounding quotes if present
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            metadata[key] = value

    return metadata, body


def parse_skill_md(skill_path: Path) -> SkillDefinition | None:
    """Parse a SKILL.md file into a SkillDefinition.

    Returns None if the file is missing, unreadable, or lacks required fields.
    """
    skill_md = skill_path / "SKILL.md"
    if not skill_md.is_file():
        logger.debug("No SKILL.md found at %s", skill_path)
        return None

    try:
        content = skill_md.read_text()
    except OSError as e:
        logger.warning("Failed to read SKILL.md at %s: %s", skill_md, e)
        return None

    metadata, body = parse_frontmatter(content)

    name = metadata.get("name", "")
    if isinstance(name, list):
        name = name[0] if name else ""
    name = str(name).strip()

    description = metadata.get("description", "")
    if isinstance(description, list):
        description = " ".join(description)
    description = str(description).strip()

    if not name:
        logger.warning("SKILL.md at %s missing required 'name' field", skill_path)
        return None
    if not description:
        logger.warning("SKILL.md at %s missing required 'description' field", skill_path)
        return None

    # Parse allowed-tools (space-delimited string)
    allowed_tools_raw = metadata.get("allowed-tools", "")
    if isinstance(allowed_tools_raw, list):
        allowed_tools = allowed_tools_raw
    else:
        allowed_tools = str(allowed_tools_raw).split() if allowed_tools_raw else []

    return SkillDefinition(
        name=name,
        description=description,
        instructions=body.strip(),
        skill_path=str(skill_path),
        allowed_tools=allowed_tools,
    )


def discover_skills(skills_dirs: list[str | Path]) -> dict[str, SkillDefinition]:
    """Scan directories for Agent Skills and return discovered definitions.

    Each immediate subdirectory of a skills_dir that contains a SKILL.md
    is treated as a skill. Returns a dict keyed by capability name
    (skill_<name>).
    """
    skills: dict[str, SkillDefinition] = {}

    for skills_dir in skills_dirs:
        dir_path = Path(skills_dir)
        if not dir_path.is_dir():
            logger.debug("Skills directory does not exist: %s", dir_path)
            continue

        for candidate in sorted(dir_path.iterdir()):
            if not candidate.is_dir():
                continue
            if candidate.name.startswith("."):
                continue

            skill = parse_skill_md(candidate)
            if skill is None:
                continue

            capability_name = _make_capability_name(skill.name)
            if capability_name in skills:
                logger.warning(
                    "Duplicate skill name '%s' from %s (already loaded from %s)",
                    skill.name,
                    candidate,
                    skills[capability_name].skill_path,
                )
                continue

            skills[capability_name] = skill
            logger.info("Discovered skill '%s' from %s", skill.name, candidate)

    return skills


def build_skill_specs(skills: dict[str, SkillDefinition]) -> list[ContextProviderSpec]:
    """Convert discovered skills into ContextProviderSpec entries for the LLM.

    Skills are presented with a generic 'request' parameter since their
    instructions determine how they're used.
    """
    specs: list[ContextProviderSpec] = []
    for capability_name, skill in skills.items():
        specs.append(
            ContextProviderSpec(
                name=capability_name,
                description=f"[Skill] {skill.description}",
                parameters={
                    "request": "What you want the skill to do. Be specific.",
                },
            )
        )
    return specs


def load_skill_references(skill: SkillDefinition) -> dict[str, str]:
    """Load reference files from a skill's references/ directory.

    Returns a dict of filename -> content. Used by the LLM-mediated
    subagent when executing a skill.
    """
    refs_dir = Path(skill.skill_path) / "references"
    if not refs_dir.is_dir():
        return {}

    references: dict[str, str] = {}
    for ref_file in sorted(refs_dir.iterdir()):
        if not ref_file.is_file():
            continue
        try:
            references[ref_file.name] = ref_file.read_text()
        except OSError as e:
            logger.warning("Failed to read reference file %s: %s", ref_file, e)

    return references
