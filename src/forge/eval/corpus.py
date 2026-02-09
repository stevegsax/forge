"""Eval case corpus loading and repo file discovery.

Provides functions to load EvalCase instances from JSON fixtures
and list files tracked by git in a repository.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import TYPE_CHECKING

from forge.eval.models import EvalCase

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def load_eval_case(path: Path) -> EvalCase:
    """Load a single EvalCase from a JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is invalid or does not match the schema.
    """
    content = path.read_text()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in {path}: {e}"
        raise ValueError(msg) from e
    try:
        return EvalCase.model_validate(data)
    except Exception as e:
        msg = f"Invalid eval case in {path}: {e}"
        raise ValueError(msg) from e


def discover_eval_cases(corpus_dir: Path) -> list[EvalCase]:
    """Discover all eval case JSON files in a directory.

    Scans for ``*.json`` files, loads each as an EvalCase, and returns
    them sorted by case_id. Files that fail to parse are logged and skipped.
    """
    cases: list[EvalCase] = []
    if not corpus_dir.is_dir():
        logger.warning("Corpus directory does not exist: %s", corpus_dir)
        return cases

    for json_file in sorted(corpus_dir.glob("*.json")):
        try:
            case = load_eval_case(json_file)
            cases.append(case)
        except (ValueError, FileNotFoundError):
            logger.warning("Skipping invalid case file: %s", json_file, exc_info=True)

    return sorted(cases, key=lambda c: c.case_id)


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def list_repo_files(repo_root: Path) -> set[str]:
    """List files tracked by git in the given repository.

    Returns relative paths as strings. Falls back to an empty set on error.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return {line for line in result.stdout.strip().splitlines() if line}
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not list git files in %s", repo_root, exc_info=True)
        return set()
