"""Namespaced tag system for the playbook service.

Tags use ``namespace:value`` format.  Five namespaces are defined:

- General knowledge (human-curated, cross-project):

    - ``lang:``    — programming language (python, typescript, go)
    - ``lib:``     — library / framework (sqlalchemy, pydantic, temporal)
    - ``domain:``  — problem domain (testing, cli, api, database, validation)

- Extracted knowledge (LLM-produced, project-specific):

    - ``project:`` — source project (forge, pbook, …)
    - ``pattern:`` — lesson type (success-pattern, failure-pattern, retry-pattern)

All functions in this module are pure.
"""

from __future__ import annotations

VALID_NAMESPACES = frozenset({"lang", "lib", "domain", "project", "pattern"})

GENERAL_NAMESPACES = frozenset({"lang", "lib", "domain"})
EXTRACTED_NAMESPACES = frozenset({"project", "pattern"})

_EXTENSION_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
}

_KEYWORD_TO_DOMAIN: dict[str, str] = {
    "test": "testing",
    "refactor": "refactoring",
    "api": "api",
    "database": "database",
    "migration": "migration",
    "cli": "cli",
    "validate": "validation",
    "bug": "bug-fix",
    "fix": "bug-fix",
}


def parse_tag(tag: str) -> tuple[str, str]:
    """Split a namespaced tag into ``(namespace, value)``.

    Raises ``ValueError`` if the tag does not contain exactly one colon
    or uses an unrecognised namespace.
    """
    if ":" not in tag:
        msg = f"Tag must use namespace:value format, got {tag!r}"
        raise ValueError(msg)

    namespace, _, value = tag.partition(":")
    if not value:
        msg = f"Tag value must not be empty: {tag!r}"
        raise ValueError(msg)

    if namespace not in VALID_NAMESPACES:
        msg = f"Unknown namespace {namespace!r} in tag {tag!r}. Valid: {sorted(VALID_NAMESPACES)}"
        raise ValueError(msg)

    return namespace, value


def validate_tag(tag: str) -> bool:
    """Return ``True`` if *tag* is a well-formed namespaced tag."""
    try:
        parse_tag(tag)
    except ValueError:
        return False
    return True


def validate_tags(tags: list[str]) -> list[str]:
    """Return a list of error messages for invalid tags.  Empty list means all valid."""
    errors: list[str] = []
    for tag in tags:
        try:
            parse_tag(tag)
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def is_general_tag(tag: str) -> bool:
    """Return ``True`` if *tag* belongs to a general-knowledge namespace."""
    try:
        ns, _ = parse_tag(tag)
    except ValueError:
        return False
    return ns in GENERAL_NAMESPACES


def is_extracted_tag(tag: str) -> bool:
    """Return ``True`` if *tag* belongs to an extracted-knowledge namespace."""
    try:
        ns, _ = parse_tag(tag)
    except ValueError:
        return False
    return ns in EXTRACTED_NAMESPACES


def infer_tags_from_context(
    file_extensions: list[str] | None = None,
    description: str = "",
) -> list[str]:
    """Infer namespaced tags from file extensions and description keywords.

    Returns a sorted, deduplicated list of tags.
    """
    tags: list[str] = []

    if file_extensions:
        for ext in file_extensions:
            lang = _EXTENSION_TO_LANG.get(ext)
            if lang:
                tags.append(f"lang:{lang}")

    if description:
        desc_lower = description.lower()
        for keyword, domain in _KEYWORD_TO_DOMAIN.items():
            if keyword in desc_lower:
                tags.append(f"domain:{domain}")

    return sorted(set(tags))
