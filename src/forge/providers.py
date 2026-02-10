"""Context provider registry for LLM-guided exploration (Phase 7).

Each provider wraps an existing tool and returns context as a string.
Handlers are thin I/O wrappers following Function Core / Imperative Shell.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Protocol

from forge.models import ContextProviderSpec

logger = logging.getLogger(__name__)

SUBPROCESS_TIMEOUT_SECONDS = 30


# ---------------------------------------------------------------------------
# Handler protocol
# ---------------------------------------------------------------------------


class ProviderHandler(Protocol):
    """Callable signature for provider handlers."""

    def __call__(self, params: dict[str, str], repo_root: str, worktree_path: str) -> str: ...


# ---------------------------------------------------------------------------
# Provider handlers
# ---------------------------------------------------------------------------


def handle_read_file(params: dict[str, str], repo_root: str, worktree_path: str) -> str:
    """Read file contents from the worktree."""
    path = params.get("path", "")
    if not path:
        return "Error: 'path' parameter is required."

    full_path = Path(worktree_path) / path
    if not full_path.is_file():
        return f"Error: File not found: {path}"

    try:
        return full_path.read_text()
    except OSError as e:
        return f"Error reading file: {e}"


def handle_search_code(params: dict[str, str], repo_root: str, worktree_path: str) -> str:
    """Search for a pattern across files in the worktree."""
    pattern = params.get("pattern", "")
    if not pattern:
        return "Error: 'pattern' parameter is required."

    glob_filter = params.get("glob", "*.py")
    wt = Path(worktree_path)

    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    matches: list[str] = []
    max_matches = 100

    for file_path in sorted(wt.rglob(glob_filter)):
        if not file_path.is_file():
            continue
        # Skip hidden dirs and common non-source directories
        rel = file_path.relative_to(wt)
        if any(part.startswith(".") for part in rel.parts):
            continue

        try:
            content = file_path.read_text()
        except (OSError, UnicodeDecodeError):
            continue

        for i, line in enumerate(content.splitlines(), start=1):
            if compiled.search(line):
                matches.append(f"{rel}:{i}: {line.rstrip()}")
                if len(matches) >= max_matches:
                    matches.append(f"... (truncated at {max_matches} matches)")
                    return "\n".join(matches)

    if not matches:
        return f"No matches found for pattern '{pattern}' in {glob_filter} files."
    return "\n".join(matches)


def handle_symbol_list(params: dict[str, str], repo_root: str, worktree_path: str) -> str:
    """List the public API of a module."""
    file_path = params.get("file_path", "")
    if not file_path:
        return "Error: 'file_path' parameter is required."

    full_path = Path(worktree_path) / file_path
    if not full_path.is_file():
        return f"Error: File not found: {file_path}"

    try:
        source = full_path.read_text()
    except OSError as e:
        return f"Error reading file: {e}"

    from forge.code_intel.graph import file_path_to_module
    from forge.code_intel.parser import extract_symbols, format_signatures

    try:
        module_name = file_path_to_module(file_path, "src")
    except (ValueError, KeyError):
        module_name = file_path

    summary = extract_symbols(source, file_path, module_name)
    return format_signatures(summary)


def handle_import_graph(params: dict[str, str], repo_root: str, worktree_path: str) -> str:
    """Show what a module imports and what imports it."""
    file_path = params.get("file_path", "")
    if not file_path:
        return "Error: 'file_path' parameter is required."

    from forge.code_intel.graph import build_import_graph, file_path_to_module

    try:
        module_name = file_path_to_module(file_path, "src")
    except (ValueError, KeyError):
        return f"Error: Cannot determine module for {file_path}"

    try:
        graph = build_import_graph("forge")
    except Exception as e:
        return f"Error building import graph: {e}"

    imports = sorted(graph.find_modules_directly_imported_by(module_name))
    imported_by = sorted(graph.find_modules_that_directly_import(module_name))

    lines = [f"Module: {module_name}", ""]
    if imports:
        lines.append("Imports:")
        for m in imports:
            lines.append(f"  - {m}")
    else:
        lines.append("Imports: (none)")

    lines.append("")
    if imported_by:
        lines.append("Imported by:")
        for m in imported_by:
            lines.append(f"  - {m}")
    else:
        lines.append("Imported by: (none)")

    return "\n".join(lines)


def handle_run_tests(params: dict[str, str], repo_root: str, worktree_path: str) -> str:
    """Execute tests and return results."""
    path = params.get("path", "")
    marker = params.get("marker", "")

    cmd = ["python", "-m", "pytest", "-x", "--tb=short", "-q"]
    if path:
        cmd.append(path)
    if marker:
        cmd.extend(["-m", marker])

    try:
        result = subprocess.run(
            cmd,
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "Error: Test execution timed out."

    output = result.stdout
    if result.stderr:
        output += "\n" + result.stderr

    # Truncate long output
    max_chars = 4000
    if len(output) > max_chars:
        output = output[:max_chars] + "\n... (truncated)"

    return output


def handle_lint_check(params: dict[str, str], repo_root: str, worktree_path: str) -> str:
    """Run ruff lint and return issues."""
    files = params.get("files", "").split(",") if params.get("files") else ["."]
    files = [f.strip() for f in files if f.strip()]

    cmd = ["ruff", "check", "--config", "tool-config/ruff.toml", "--no-fix", *files]

    try:
        result = subprocess.run(
            cmd,
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "Error: Lint check timed out."

    if result.returncode == 0:
        return "No lint issues found."

    output = result.stdout or result.stderr
    max_chars = 4000
    if len(output) > max_chars:
        output = output[:max_chars] + "\n... (truncated)"
    return output


def handle_git_log(params: dict[str, str], repo_root: str, worktree_path: str) -> str:
    """Show recent commit history."""
    n = params.get("n", "10")
    path = params.get("path", "")

    cmd = ["git", "log", f"--max-count={n}", "--oneline", "--no-decorate"]
    if path:
        cmd.extend(["--", path])

    try:
        result = subprocess.run(
            cmd,
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "Error: Git log timed out."

    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout.strip() or "No commits found."


def handle_git_diff(params: dict[str, str], repo_root: str, worktree_path: str) -> str:
    """Show diff against a base branch."""
    base = params.get("base", "main")

    cmd = ["git", "diff", f"{base}...HEAD", "--stat"]

    try:
        result = subprocess.run(
            cmd,
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "Error: Git diff timed out."

    if result.returncode != 0:
        return f"Error: {result.stderr}"
    return result.stdout.strip() or "No differences."


def handle_repo_map(params: dict[str, str], repo_root: str, worktree_path: str) -> str:
    """Generate a ranked project structure map."""
    from forge.code_intel import discover_context

    try:
        packed = discover_context(
            target_files=[],
            project_root=worktree_path,
            package_name="forge",
            src_root="src",
            include_repo_map=True,
            repo_map_tokens=4096,
            token_budget=5000,
        )
        if packed.repo_map and packed.repo_map.content:
            return packed.repo_map.content
        return "No repo map generated."
    except Exception as e:
        return f"Error generating repo map: {e}"


def handle_discover_context(params: dict[str, str], repo_root: str, worktree_path: str) -> str:
    """Run full auto-discovery for target files."""
    target_files_raw = params.get("target_files", "")
    if not target_files_raw:
        return "Error: 'target_files' parameter is required."

    target_files = [f.strip() for f in target_files_raw.split(",") if f.strip()]

    from forge.code_intel import discover_context

    try:
        packed = discover_context(
            target_files=target_files,
            project_root=worktree_path,
            package_name="forge",
            src_root="src",
            include_dependencies=True,
        )
    except Exception as e:
        return f"Error running context discovery: {e}"

    lines = [
        f"Files included: {packed.items_included}",
        f"Files truncated: {packed.items_truncated}",
        f"Estimated tokens: {packed.total_estimated_tokens}",
        f"Budget utilization: {packed.budget_utilization:.1%}",
        "",
        "Included files:",
    ]
    for item in packed.items:
        rep = item.representation.value
        lines.append(f"  - {item.file_path} ({rep}, {item.estimated_tokens} tokens)")

    return "\n".join(lines)


def handle_past_runs(params: dict[str, str], repo_root: str, worktree_path: str) -> str:
    """Show recent workflow run results."""
    limit = int(params.get("limit", "5"))

    try:
        from forge.store import get_db_path, get_engine, list_recent_runs

        db_path = get_db_path()
        if db_path is None or not db_path.exists():
            return "No store available."

        engine = get_engine(db_path)
        runs = list_recent_runs(engine, limit=limit)
    except Exception as e:
        return f"Error querying runs: {e}"

    if not runs:
        return "No recent runs found."

    lines: list[str] = []
    for r in runs:
        lines.append(f"{r['workflow_id']}  {r['task_id']}  {r['status']}  {r['created_at']}")
    return "\n".join(lines)


def handle_playbooks(params: dict[str, str], repo_root: str, worktree_path: str) -> str:
    """Show relevant playbook entries."""
    tags_raw = params.get("tags", "")
    if not tags_raw:
        return "Error: 'tags' parameter is required."

    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

    try:
        from forge.store import get_db_path, get_engine, get_playbooks_by_tags

        db_path = get_db_path()
        if db_path is None or not db_path.exists():
            return "No store available."

        engine = get_engine(db_path)
        entries = get_playbooks_by_tags(engine, tags, limit=5)
    except Exception as e:
        return f"Error querying playbooks: {e}"

    if not entries:
        return f"No playbooks found for tags: {', '.join(tags)}"

    lines: list[str] = []
    for entry in entries:
        lines.append(f"**{entry['title']}**")
        lines.append(entry["content"])
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


PROVIDER_REGISTRY: dict[str, ProviderHandler] = {
    "read_file": handle_read_file,
    "search_code": handle_search_code,
    "symbol_list": handle_symbol_list,
    "import_graph": handle_import_graph,
    "run_tests": handle_run_tests,
    "lint_check": handle_lint_check,
    "git_log": handle_git_log,
    "git_diff": handle_git_diff,
    "repo_map": handle_repo_map,
    "discover_context": handle_discover_context,
    "past_runs": handle_past_runs,
    "playbooks": handle_playbooks,
}


PROVIDER_SPECS: list[ContextProviderSpec] = [
    ContextProviderSpec(
        name="read_file",
        description="Read the full contents of a file.",
        parameters={"path": "Relative file path within the project."},
    ),
    ContextProviderSpec(
        name="search_code",
        description="Search for a regex pattern across source files.",
        parameters={
            "pattern": "Python regex pattern to search for.",
            "glob": "File glob to filter (default: '*.py').",
        },
    ),
    ContextProviderSpec(
        name="symbol_list",
        description="List the public API (functions, classes, constants) of a Python module.",
        parameters={"file_path": "Relative file path to the module."},
    ),
    ContextProviderSpec(
        name="import_graph",
        description="Show what a module imports and what modules import it.",
        parameters={"file_path": "Relative file path to the module."},
    ),
    ContextProviderSpec(
        name="run_tests",
        description="Execute pytest and return results.",
        parameters={
            "path": "Test file or directory (optional, runs all tests if empty).",
            "marker": "Pytest marker to filter tests (optional).",
        },
    ),
    ContextProviderSpec(
        name="lint_check",
        description="Run ruff linter and return any issues found.",
        parameters={"files": "Comma-separated file paths (optional, checks all if empty)."},
    ),
    ContextProviderSpec(
        name="git_log",
        description="Show recent git commit history.",
        parameters={
            "path": "File or directory to filter history (optional).",
            "n": "Number of commits to show (default: 10).",
        },
    ),
    ContextProviderSpec(
        name="git_diff",
        description="Show diff summary against a base branch.",
        parameters={"base": "Base branch to diff against (default: 'main')."},
    ),
    ContextProviderSpec(
        name="repo_map",
        description=(
            "Generate a ranked structural overview of the project (file paths and signatures)."
        ),
        parameters={},
    ),
    ContextProviderSpec(
        name="discover_context",
        description=(
            "Run full automatic context discovery for target files "
            "(import graph, PageRank, symbols)."
        ),
        parameters={"target_files": "Comma-separated list of target file paths."},
    ),
    ContextProviderSpec(
        name="past_runs",
        description="Show recent Forge workflow run results.",
        parameters={"limit": "Number of runs to show (default: 5)."},
    ),
    ContextProviderSpec(
        name="playbooks",
        description="Retrieve relevant playbook entries (lessons from previous tasks).",
        parameters={"tags": "Comma-separated tags to search for."},
    ),
]
