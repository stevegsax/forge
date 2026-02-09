"""Tests for forge.eval.corpus — eval case loading and repo discovery."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from forge.eval.corpus import discover_eval_cases, list_repo_files, load_eval_case

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "eval"


# ---------------------------------------------------------------------------
# load_eval_case
# ---------------------------------------------------------------------------


class TestLoadEvalCase:
    def test_load_add_feature(self) -> None:
        case = load_eval_case(_FIXTURES / "cases" / "case_add_feature.json")
        assert case.case_id == "add-feature"
        assert case.task.task_id == "add-auth"
        assert "auth.py" in case.task.target_files[0]

    def test_load_refactor(self) -> None:
        case = load_eval_case(_FIXTURES / "cases" / "case_refactor.json")
        assert case.case_id == "refactor"
        assert case.tags == ["refactor", "simple"]

    def test_load_fan_out(self) -> None:
        case = load_eval_case(_FIXTURES / "cases" / "case_fan_out.json")
        assert case.case_id == "fan-out"
        assert "fan-out" in case.tags

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_eval_case(Path("/nonexistent/case.json"))

    def test_invalid_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not json {{{")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_eval_case(bad)

    def test_invalid_schema(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"case_id": "x"}))
        with pytest.raises(ValueError, match="Invalid eval case"):
            load_eval_case(bad)


# ---------------------------------------------------------------------------
# discover_eval_cases
# ---------------------------------------------------------------------------


class TestDiscoverEvalCases:
    def test_discover_all_cases(self) -> None:
        cases = discover_eval_cases(_FIXTURES / "cases")
        assert len(cases) == 3
        ids = [c.case_id for c in cases]
        assert ids == sorted(ids)

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        cases = discover_eval_cases(tmp_path / "nope")
        assert cases == []

    def test_skips_invalid_files(self, tmp_path: Path) -> None:
        good = tmp_path / "good.json"
        good.write_text(json.dumps({
            "case_id": "good",
            "task": {"task_id": "t1", "description": "Test."},
            "repo_root": "/tmp/repo",
        }))
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json")

        cases = discover_eval_cases(tmp_path)
        assert len(cases) == 1
        assert cases[0].case_id == "good"


# ---------------------------------------------------------------------------
# list_repo_files
# ---------------------------------------------------------------------------


class TestListRepoFiles:
    def test_lists_files_in_git_repo(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path, check=True, capture_output=True,
        )
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_path, check=True, capture_output=True,
        )

        files = list_repo_files(tmp_path)
        assert "a.py" in files
        assert "b.py" in files

    def test_returns_empty_for_non_repo(self, tmp_path: Path) -> None:
        files = list_repo_files(tmp_path)
        assert files == set()
