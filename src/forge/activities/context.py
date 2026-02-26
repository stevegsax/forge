"""Context assembly activity for Forge.

Builds the system and user prompts from a task definition and context files.

Design follows Function Core / Imperative Shell:
- Pure functions: build_system_prompt, build_user_prompt,
  build_step_system_prompt, build_step_user_prompt,
  build_system_prompt_with_context, parse_ruff_error_lines,
  find_enclosing_scope
- Imperative shell: _read_context_files, build_error_section,
  assemble_context, assemble_step_context
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from temporalio import activity

from forge.domains import get_domain_config
from forge.models import (
    AssembleContextInput,
    AssembledContext,
    AssembleStepContextInput,
    AssembleSubTaskContextInput,
    ContextStats,
    PlanStep,
    StepResult,
    SubTask,
    TaskDefinition,
    TaskDomain,
    ValidationResult,
)

if TYPE_CHECKING:
    from forge.code_intel.budget import ContextItem, PackedContext

logger = logging.getLogger(__name__)

_ERROR_DETAIL_LIMIT = 2000
_PROJECT_INSTRUCTIONS_FILENAME = "CLAUDE.md"

_OUTPUT_REQUIREMENTS = (
    "You MUST respond with a valid LLMResponse containing an `explanation` string "
    "and either `files`, `edits`, or both.\n\n"
    "- **`files`**: Use for NEW files that don't exist yet. Each entry needs "
    "`file_path` and `content` (complete file content).\n"
    "- **`edits`**: Use for EXISTING files that need changes. Each entry needs "
    "`file_path` and a list of `edits`, where each edit has `search` (exact text "
    "to find, must match exactly once) and `replace` (replacement text).\n\n"
    "A file path must NOT appear in both `files` and `edits`. "
    "Do NOT return an empty object."
)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def build_project_instructions_section(content: str) -> str:
    """Format CLAUDE.md content as a Project Instructions prompt section.

    Returns an empty string if *content* is blank.
    """
    stripped = content.strip()
    if not stripped:
        return ""
    return f"## Project Instructions\n\n{stripped}"


def parse_ruff_error_lines(error_output: str) -> list[tuple[str, int, str]]:
    """Parse ruff output into (file_path, line_number, error_message) tuples.

    Ruff format: ``path:line:col: code message``
    Best-effort: unparseable lines are skipped.
    """
    results: list[tuple[str, int, str]] = []
    for line in error_output.splitlines():
        m = re.match(r"^(.+?):(\d+):\d+:\s+(.+)$", line)
        if m:
            results.append((m.group(1), int(m.group(2)), m.group(3)))
    return results


def find_enclosing_scope(source: str, line_number: int) -> str | None:
    """Find the enclosing function/class for a line and return a context snippet.

    Uses ``ast.parse`` to find the innermost ``FunctionDef``, ``AsyncFunctionDef``,
    or ``ClassDef`` containing *line_number*. Returns a snippet showing the
    scope header and the error line marked with ``# <-- ERROR``.

    Returns ``None`` if parsing fails or no enclosing scope is found.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    source_lines = source.splitlines()
    if line_number < 1 or line_number > len(source_lines):
        return None

    # Walk the AST to find the innermost enclosing scope
    best: ast.AST | None = None
    best_start = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            node_start = node.lineno
            node_end = node.end_lineno or node.lineno
            if node_start <= line_number <= node_end and (best is None or node_start > best_start):
                best = node
                best_start = node_start

    if best is None:
        return None

    # Build snippet: scope header line + a few lines around the error
    scope_start = best_start
    context_start = max(scope_start, line_number - 3)
    context_end = min(len(source_lines), line_number + 3)

    snippet_lines: list[str] = []
    if context_start > scope_start:
        # Include the scope header
        snippet_lines.append(source_lines[scope_start - 1])
        if context_start > scope_start + 1:
            snippet_lines.append("    ...")

    for i in range(context_start, context_end + 1):
        src_line = source_lines[i - 1]
        if i == line_number:
            snippet_lines.append(f"{src_line}  # <-- ERROR")
        else:
            snippet_lines.append(src_line)

    return "\n".join(snippet_lines)


def build_error_section(
    prior_errors: list[ValidationResult],
    attempt: int,
    max_attempts: int,
    worktree_path: str,
) -> str:
    """Render a 'Previous Attempt Errors' prompt section.

    For ``ruff_lint``/``ruff_format`` errors: parses line numbers and enriches
    with AST context from files in *worktree_path*.
    For ``tests`` errors: includes output verbatim.
    Truncates individual error details to ``_ERROR_DETAIL_LIMIT`` chars.

    Returns an empty string when *prior_errors* is empty.
    """
    if not prior_errors:
        return ""

    failed = [e for e in prior_errors if not e.passed]
    if not failed:
        return ""

    parts: list[str] = []
    parts.append(f"## Previous Attempt Errors (Attempt {attempt} of {max_attempts})")
    parts.append("")
    parts.append("Your previous attempt failed validation. Fix these errors:")

    wt = Path(worktree_path)

    for error in failed:
        parts.append("")
        parts.append(f"### {error.check_name} failed")

        details = error.details or error.summary
        if len(details) > _ERROR_DETAIL_LIMIT:
            details = details[:_ERROR_DETAIL_LIMIT] + "\n... (truncated)"

        parts.append("```")
        parts.append(details)
        parts.append("```")

        # AST enrichment for ruff errors
        if error.check_name in ("ruff_lint", "ruff_format") and error.details:
            parsed = parse_ruff_error_lines(error.details)
            # Deduplicate by (file, line) to avoid redundant snippets
            seen: set[tuple[str, int]] = set()
            for file_path, line_num, _msg in parsed:
                if (file_path, line_num) in seen:
                    continue
                seen.add((file_path, line_num))
                full_path = wt / file_path
                if not full_path.is_file():
                    continue
                try:
                    source = full_path.read_text()
                except OSError:
                    continue
                snippet = find_enclosing_scope(source, line_num)
                if snippet:
                    fname = Path(file_path).name
                    parts.append("")
                    parts.append(f"#### Context around error ({fname}, line {line_num})")
                    parts.append("```python")
                    parts.append(snippet)
                    parts.append("```")

    parts.append("")
    parts.append("Do NOT repeat the same mistakes. Address each error listed above.")

    return "\n".join(parts)


def build_system_prompt(
    task: TaskDefinition,
    context_file_contents: dict[str, str],
    error_section: str = "",
    project_instructions: str = "",
) -> str:
    """Build the system prompt from a task definition and context file contents.

    Includes the task description, target files list, and any context files
    with delimiters. Sections are ordered for prompt caching efficiency:
    stable content first, volatile content (errors) last.
    """
    domain_config = get_domain_config(task.domain)
    parts: list[str] = []

    parts.append(domain_config.role_prompt)
    parts.append("")
    parts.append("## Output Requirements")
    parts.append(domain_config.output_requirements)

    if project_instructions:
        parts.append("")
        parts.append(project_instructions)

    parts.append("")
    parts.append("## Task")
    parts.append(task.description)
    parts.append("")
    parts.append("## Target Files")
    for f in task.target_files:
        parts.append(f"- {f}")

    if context_file_contents:
        parts.append("")
        parts.append("## Context Files")
        for file_path, content in context_file_contents.items():
            parts.append("")
            parts.append(f"### {file_path}")
            parts.append("```")
            parts.append(content)
            parts.append("```")

    if error_section:
        parts.append("")
        parts.append(error_section)

    return "\n".join(parts)


def build_user_prompt(task: TaskDefinition | None = None) -> str:
    """Build the user prompt.

    Short instruction directing the LLM to produce target files or edits.
    When *task* is provided, the prompt is domain-aware.
    """
    domain = task.domain if task is not None else TaskDomain.CODE_GENERATION
    domain_config = get_domain_config(domain)
    return domain_config.user_prompt_template


def build_system_prompt_with_context(
    task: TaskDefinition,
    packed: PackedContext,
    error_section: str = "",
    project_instructions: str = "",
) -> str:
    """Build the system prompt using auto-discovered packed context.

    Organizes context items by priority tier with labeled sections.
    Sections are ordered for prompt caching efficiency: stable content
    (role, output requirements, repo structure, playbooks) first,
    volatile content (errors) last.
    """

    domain_config = get_domain_config(task.domain)
    parts: list[str] = []

    # --- Stable across ALL calls ---
    parts.append(domain_config.role_prompt)
    parts.append("")
    parts.append("## Output Requirements")
    parts.append(domain_config.output_requirements)

    if project_instructions:
        parts.append("")
        parts.append(project_instructions)

    # Repo map (priority 5, REPO_MAP representation) — stable per repo
    from forge.code_intel.budget import Representation

    repo_map_items = [i for i in packed.items if i.representation == Representation.REPO_MAP]
    if repo_map_items:
        parts.append("")
        parts.append("## Repository Structure")
        parts.append("```")
        parts.append(repo_map_items[0].content)
        parts.append("```")

    # Playbooks (priority 5, PLAYBOOK representation) — stable per tag set
    playbook_items = [i for i in packed.items if i.representation == Representation.PLAYBOOK]
    if playbook_items:
        parts.append("")
        parts.append("## Relevant Playbooks")
        parts.append(
            "The following are lessons learned from previous tasks. "
            "Consider these when generating code."
        )
        for item in playbook_items:
            parts.append("")
            parts.append(item.content)

    # --- Task-specific content ---
    parts.append("")
    parts.append("## Task")
    parts.append(task.description)
    parts.append("")
    parts.append("## Target Files")
    for f in task.target_files:
        parts.append(f"- {f}")

    # Group items by representation type
    _append_context_section(
        parts,
        "Target File Contents",
        packed,
        priority=2,
    )
    _append_context_section(
        parts,
        "Direct Dependencies (full content)",
        packed,
        priority=3,
    )
    _append_context_section(
        parts,
        "Interface Context (signatures)",
        packed,
        priority=4,
    )

    _append_context_section(
        parts,
        "Additional Context",
        packed,
        priority=6,
    )

    # --- Most volatile — only on retry ---
    if error_section:
        parts.append("")
        parts.append(error_section)

    return "\n".join(parts)


def _append_context_section(
    parts: list[str],
    section_title: str,
    packed: PackedContext,
    priority: int,
) -> None:
    """Append a context section for items at the given priority level."""
    from forge.code_intel.budget import Representation

    items = [i for i in packed.items if i.priority == priority]
    if not items:
        return

    parts.append("")
    parts.append(f"## {section_title}")
    for item in items:
        parts.append("")
        if item.representation == Representation.SIGNATURES:
            parts.append(f"### {item.file_path} (signatures)")
        else:
            parts.append(f"### {item.file_path}")
        parts.append("```")
        parts.append(item.content)
        parts.append("```")


def _build_context_stats(packed: PackedContext) -> ContextStats:
    """Build ContextStats from a PackedContext."""
    from forge.code_intel.budget import Representation

    full_count = sum(1 for i in packed.items if i.representation == Representation.FULL)
    sig_count = sum(1 for i in packed.items if i.representation == Representation.SIGNATURES)
    repo_map_tokens = sum(
        i.estimated_tokens for i in packed.items if i.representation == Representation.REPO_MAP
    )

    return ContextStats(
        files_discovered=packed.items_included + packed.items_truncated,
        files_included_full=full_count,
        files_included_signatures=sig_count,
        files_truncated=packed.items_truncated,
        total_estimated_tokens=packed.total_estimated_tokens,
        budget_utilization=packed.budget_utilization,
        repo_map_tokens=repo_map_tokens,
    )


# ---------------------------------------------------------------------------
# Playbook injection (Phase 6)
# ---------------------------------------------------------------------------


def infer_task_tags(task: TaskDefinition) -> list[str]:
    """Infer search tags from a task definition for playbook retrieval.

    Deterministic: extracts tags from file extensions and description keywords.
    """
    tags: list[str] = []

    for f in task.target_files:
        if f.endswith(".py"):
            tags.append("python")
        elif f.endswith(".ts") or f.endswith(".tsx"):
            tags.append("typescript")
        elif f.endswith(".js") or f.endswith(".jsx"):
            tags.append("javascript")

    desc_lower = task.description.lower()
    keyword_map = {
        "test": "test-writing",
        "refactor": "refactoring",
        "api": "api",
        "database": "database",
        "migration": "migration",
        "cli": "cli",
        "validate": "validation",
        "bug": "bug-fix",
        "fix": "bug-fix",
    }
    for keyword, tag in keyword_map.items():
        if keyword in desc_lower:
            tags.append(tag)

    if not tags:
        tags.append(task.domain.value.replace("_", "-"))

    return sorted(set(tags))


def build_playbook_context_items(playbooks: list[dict]) -> list[ContextItem]:
    """Convert playbook dicts from the store to ContextItem objects at priority 5.

    Uses Representation.PLAYBOOK to distinguish from repo map items
    (also at priority 5 but using Representation.REPO_MAP).
    """
    from forge.code_intel.budget import ContextItem, Representation
    from forge.code_intel.repo_map import estimate_tokens

    items: list[ContextItem] = []
    for pb in playbooks:
        content = f"**{pb['title']}**\n{pb['content']}"
        items.append(
            ContextItem(
                file_path=f"playbook:{pb['title']}",
                content=content,
                representation=Representation.PLAYBOOK,
                priority=5,
                importance=0.0,
                estimated_tokens=estimate_tokens(content),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def _load_playbooks_for_task(task: TaskDefinition) -> list[dict]:
    """Best-effort load of relevant playbooks from the store.

    Returns empty list on any error (D42 pattern).
    """
    try:
        from forge.store import get_db_path, get_engine, get_playbooks_by_tags

        db_path = get_db_path()
        if db_path is None or not db_path.exists():
            return []

        tags = infer_task_tags(task)
        if not tags:
            return []

        engine = get_engine(db_path)
        return get_playbooks_by_tags(engine, tags, limit=5)
    except Exception:
        logger.warning("Failed to load playbooks from store", exc_info=True)
        return []


def _read_project_instructions(repo_root: Path) -> str:
    """Read CLAUDE.md from the repository root.

    Returns the file content, or an empty string if the file is missing
    or unreadable.  Best-effort — never raises.
    """
    try:
        instructions_path = repo_root / _PROJECT_INSTRUCTIONS_FILENAME
        if instructions_path.is_file():
            return instructions_path.read_text()
    except Exception:
        logger.warning("Failed to read %s", _PROJECT_INSTRUCTIONS_FILENAME, exc_info=True)
    return ""


def _read_context_files(base_path: Path, file_paths: list[str]) -> dict[str, str]:
    """Read context files from disk.

    Skips missing files with a warning log. Context files are read from
    the repo root (not the worktree) so the LLM sees the current state
    of reference files.
    """
    contents: dict[str, str] = {}
    for file_path in file_paths:
        full_path = base_path / file_path
        if not full_path.is_file():
            logger.warning("Context file not found, skipping: %s", full_path)
            continue
        contents[file_path] = full_path.read_text()
    return contents


@activity.defn
async def assemble_context(input: AssembleContextInput) -> AssembledContext:
    """Read context files and assemble the prompts for the LLM call.

    When auto_discover is enabled and target_files are specified, uses
    automatic context discovery via import graph analysis. Otherwise
    falls back to manual context_files.
    """
    from forge.tracing import get_tracer

    tracer = get_tracer()
    with tracer.start_as_current_span("forge.assemble_context"):
        logger.info("Assemble context: task_id=%s", input.task_id)
        return await _assemble_context_inner(input)


async def _assemble_context_inner(input: AssembleContextInput) -> AssembledContext:
    """Inner implementation of assemble_context (extracted for span wrapping)."""
    task_id = input.task_id
    repo_root = Path(input.repo_root)

    error_section = build_error_section(
        input.prior_errors, input.attempt, input.max_attempts, input.worktree_path
    )

    project_instructions = build_project_instructions_section(_read_project_instructions(repo_root))

    if input.context_config.auto_discover and input.target_files:
        from forge.code_intel import discover_context
        from forge.code_intel.budget import pack_context

        manual_contents = _read_context_files(repo_root, input.context_files)

        packed = discover_context(
            target_files=input.target_files,
            project_root=input.repo_root,
            package_name=input.context_config.package_name or _detect_package_name(input.repo_root),
            src_root="src",
            manual_context=manual_contents,
            token_budget=input.context_config.token_budget,
            max_import_depth=input.context_config.max_import_depth,
            include_repo_map=input.context_config.include_repo_map,
            repo_map_tokens=input.context_config.repo_map_tokens,
            include_dependencies=input.context_config.include_dependencies,
        )

        # Inject playbooks (best-effort, D42)
        # We'd need to reconstruct a task-like object or modify infer_task_tags
        task_mock = TaskDefinition(
            task_id=input.task_id,
            description=input.description,
            target_files=input.target_files,
            context_files=input.context_files,
            context=input.context_config,
        )
        playbooks = _load_playbooks_for_task(task_mock)
        if playbooks:
            playbook_items = build_playbook_context_items(playbooks)
            all_items = packed.items + playbook_items
            packed = pack_context(all_items, input.context_config.token_budget)

        system_prompt = build_system_prompt_with_context(
            task_mock, packed, error_section, project_instructions=project_instructions
        )
        context_stats = _build_context_stats(packed)
    else:
        task_mock = TaskDefinition(
            task_id=input.task_id,
            description=input.description,
            target_files=input.target_files,
            context_files=input.context_files,
            context=input.context_config,
        )
        context_contents = _read_context_files(repo_root, input.context_files)
        system_prompt = build_system_prompt(
            task_mock, context_contents, error_section, project_instructions=project_instructions
        )
        context_stats = None

    user_prompt = build_user_prompt(task_mock)

    return AssembledContext(
        task_id=input.task_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_stats=context_stats,
    )


def _detect_package_name(repo_root: str) -> str:
    """Detect the Python package name from the src/ directory.

    Looks for the first directory in src/ that contains __init__.py.
    Falls back to the repo directory name.
    """
    src_dir = Path(repo_root) / "src"
    if src_dir.is_dir():
        for child in sorted(src_dir.iterdir()):
            if child.is_dir() and (child / "__init__.py").exists():
                return child.name
    return Path(repo_root).name


# ---------------------------------------------------------------------------
# Step-level context (Phase 2)
# ---------------------------------------------------------------------------


def build_step_system_prompt(
    task: TaskDefinition,
    step: PlanStep,
    step_index: int,
    total_steps: int,
    completed_steps: list[StepResult],
    context_file_contents: dict[str, str],
    target_file_contents: dict[str, str] | None = None,
    error_section: str = "",
    project_instructions: str = "",
) -> str:
    """Build the system prompt for a single step execution.

    Includes the overall task summary, plan progress, current step details,
    context files, and current target file contents from the worktree.
    Sections are ordered for prompt caching efficiency: stable content first,
    volatile content (errors) last.
    """
    domain_config = get_domain_config(task.domain)
    parts: list[str] = []

    parts.append(domain_config.role_prompt)
    parts.append("")
    parts.append("### Output Requirements")
    parts.append(domain_config.output_requirements)

    if project_instructions:
        parts.append("")
        parts.append(project_instructions)

    parts.append("")
    parts.append("## Overall Task")
    parts.append(task.description)
    parts.append("")
    parts.append(f"## Plan Progress ({step_index + 1}/{total_steps})")

    if completed_steps:
        parts.append("")
        parts.append("Completed steps:")
        for cs in completed_steps:
            parts.append(f"- {cs.step_id}: {cs.status.value}")
    else:
        parts.append("No steps completed yet.")

    parts.append("")
    parts.append("## Current Step")
    parts.append(f"**Step ID:** {step.step_id}")
    parts.append(f"**Description:** {step.description}")
    parts.append("")
    parts.append("### Target Files")
    for f in step.target_files:
        parts.append(f"- {f}")

    if target_file_contents:
        parts.append("")
        parts.append("### Current Target File Contents")
        for file_path, content in target_file_contents.items():
            parts.append("")
            parts.append(f"#### {file_path}")
            parts.append("```")
            parts.append(content)
            parts.append("```")

    if context_file_contents:
        parts.append("")
        parts.append("### Context Files")
        for file_path, content in context_file_contents.items():
            parts.append("")
            parts.append(f"#### {file_path}")
            parts.append("```")
            parts.append(content)
            parts.append("```")

    if error_section:
        parts.append("")
        parts.append(error_section)

    return "\n".join(parts)


def build_step_user_prompt(
    step: PlanStep,
    domain: TaskDomain = TaskDomain.CODE_GENERATION,
) -> str:
    """Build the user prompt for a single step execution."""
    domain_config = get_domain_config(domain)
    return domain_config.step_user_prompt_template.format(
        step_id=step.step_id,
        step_description=step.description,
    )


@activity.defn
async def assemble_step_context(input: AssembleStepContextInput) -> AssembledContext:
    """Read context files from the worktree and assemble step-level prompts.

    Context files are read from the **worktree** (not repo root) so that
    later steps can see files created by earlier steps.
    Target files are also read from the worktree so the LLM can produce
    search/replace edits instead of rewriting entire files.
    """
    logger.info("Assemble step context: task_id=%s step_id=%s", input.task_id, input.step.step_id)
    worktree = Path(input.worktree_path)
    context_contents = _read_context_files(worktree, input.step.context_files)
    target_contents = _read_context_files(worktree, input.step.target_files)

    error_section = build_error_section(
        input.prior_errors, input.attempt, input.max_attempts, input.worktree_path
    )

    project_instructions = build_project_instructions_section(
        _read_project_instructions(Path(input.repo_root))
    )

    task_mock = TaskDefinition(
        task_id=input.task_id,
        description=input.task_description,
        context=input.context_config,
    )

    system_prompt = build_step_system_prompt(
        task=task_mock,
        step=input.step,
        step_index=input.step_index,
        total_steps=input.total_steps,
        completed_steps=input.completed_steps,
        context_file_contents=context_contents,
        target_file_contents=target_contents,
        error_section=error_section,
        project_instructions=project_instructions,
    )
    user_prompt = build_step_user_prompt(input.step, domain=task_mock.domain)

    return AssembledContext(
        task_id=input.task_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        step_id=input.step.step_id,
    )


# ---------------------------------------------------------------------------
# Sub-task context (Phase 3)
# ---------------------------------------------------------------------------


def build_sub_task_system_prompt(
    parent_task_id: str,
    parent_description: str,
    sub_task: SubTask,
    context_file_contents: dict[str, str],
    target_file_contents: dict[str, str] | None = None,
    error_section: str = "",
    project_instructions: str = "",
    domain: TaskDomain = TaskDomain.CODE_GENERATION,
) -> str:
    """Build the system prompt for a sub-task execution.

    Includes parent task context, sub-task description, target files,
    current target file contents, and context files read from the parent worktree.
    Sections are ordered for prompt caching efficiency: stable content first,
    volatile content (errors) last.
    """
    domain_config = get_domain_config(domain)
    parts: list[str] = []

    parts.append(domain_config.role_prompt)
    parts.append("")
    parts.append("### Output Requirements")
    parts.append(domain_config.output_requirements)

    if project_instructions:
        parts.append("")
        parts.append(project_instructions)

    parts.append("")
    parts.append("## Parent Task")
    parts.append(f"**Task ID:** {parent_task_id}")
    parts.append(f"**Description:** {parent_description}")
    parts.append("")
    parts.append("## Sub-Task")
    parts.append(f"**Sub-Task ID:** {sub_task.sub_task_id}")
    parts.append(f"**Description:** {sub_task.description}")
    parts.append("")
    parts.append("### Target Files")
    for f in sub_task.target_files:
        parts.append(f"- {f}")

    if target_file_contents:
        parts.append("")
        parts.append("### Current Target File Contents")
        for file_path, content in target_file_contents.items():
            parts.append("")
            parts.append(f"#### {file_path}")
            parts.append("```")
            parts.append(content)
            parts.append("```")

    if context_file_contents:
        parts.append("")
        parts.append("### Context Files")
        for file_path, content in context_file_contents.items():
            parts.append("")
            parts.append(f"#### {file_path}")
            parts.append("```")
            parts.append(content)
            parts.append("```")

    if error_section:
        parts.append("")
        parts.append(error_section)

    return "\n".join(parts)


def build_sub_task_user_prompt(
    sub_task: SubTask,
    domain: TaskDomain = TaskDomain.CODE_GENERATION,
) -> str:
    """Build the user prompt for a sub-task execution."""
    domain_config = get_domain_config(domain)
    return domain_config.sub_task_user_prompt_template.format(
        sub_task_id=sub_task.sub_task_id,
        sub_task_description=sub_task.description,
    )


@activity.defn
async def assemble_sub_task_context(
    input: AssembleSubTaskContextInput,
) -> AssembledContext:
    """Read context files from the parent worktree and assemble sub-task prompts.

    Context files are read from the **parent worktree** because the sub-task
    worktree starts empty (branched from parent branch).
    Target files are also read from the parent worktree so the LLM can produce
    search/replace edits instead of rewriting entire files.
    """
    logger.info(
        "Assemble sub-task context: task_id=%s sub_task_id=%s",
        input.parent_task_id,
        input.sub_task.sub_task_id,
    )
    parent_worktree = Path(input.worktree_path)
    context_contents = _read_context_files(parent_worktree, input.sub_task.context_files)
    target_contents = _read_context_files(parent_worktree, input.sub_task.target_files)

    error_section = build_error_section(
        input.prior_errors, input.attempt, input.max_attempts, input.worktree_path
    )

    repo_root = Path(input.repo_root) if input.repo_root else parent_worktree
    project_instructions = build_project_instructions_section(_read_project_instructions(repo_root))

    system_prompt = build_sub_task_system_prompt(
        parent_task_id=input.parent_task_id,
        parent_description=input.parent_description,
        sub_task=input.sub_task,
        context_file_contents=context_contents,
        target_file_contents=target_contents,
        error_section=error_section,
        project_instructions=project_instructions,
        domain=input.domain,
    )
    user_prompt = build_sub_task_user_prompt(input.sub_task, domain=input.domain)

    return AssembledContext(
        task_id=input.parent_task_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        sub_task_id=input.sub_task.sub_task_id,
    )
