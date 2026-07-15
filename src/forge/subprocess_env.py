"""Allowlisted environment for model-influenced subprocess seams.

Two seams execute commands whose argv the LLM influences: the validation
activity's test/lint runners (``forge.activities.validate``) and the exploration
context providers (``forge.providers``). Handing those subprocesses the worker's
full environment would leak the platform's secrets — API keys, database URLs,
TLS material — to a command line the model shaped. This module builds an
explicit allowlist environment instead.

Pure core: ``allowlist_env`` is a deterministic function of the mapping it is
given. The shell (each subprocess runner) supplies ``os.environ`` at the edge.
"""

from collections.abc import Mapping

__all__ = ["ALLOWED_SUBPROCESS_ENV_KEYS", "allowlist_env"]

# The only variables a model-influenced subprocess inherits.
#
# - PATH        resolve the executables (python, pytest, ruff, rg, git, sh).
# - HOME        tool config + writable cache location (git config, uv/ruff cache).
# - VIRTUAL_ENV pytest/ruff resolve against the project venv, not a system one.
# - LANG        text encoding for output the parser reads back.
# - TMPDIR      scratch space for tools that spill to a temp file.
#
# Secrets (ANTHROPIC_API_KEY, FORGE_DB_URL, AWS_*, TLS paths, …) are excluded by
# construction: anything not named here never reaches the child.
ALLOWED_SUBPROCESS_ENV_KEYS: tuple[str, ...] = (
    "PATH",
    "HOME",
    "VIRTUAL_ENV",
    "LANG",
    "TMPDIR",
)


def allowlist_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Return the subset of *environ* limited to the subprocess allowlist.

    Keys absent from *environ* are simply omitted (no empty placeholders). The
    result is a fresh dict suitable to pass as ``subprocess.run(..., env=...)``,
    which replaces the child's environment wholesale rather than augmenting it.
    """
    return {key: environ[key] for key in ALLOWED_SUBPROCESS_ENV_KEYS if key in environ}
