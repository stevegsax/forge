"""Read-path confinement for context and provider file access.

Forge feeds file contents into LLM context. Those file paths originate from
task definitions and from the LLM itself (via exploration providers), so they
are untrusted: a ``../`` traversal or an absolute path must not be allowed to
read outside the worktree. "Context isolation is a feature" (AGENTS.md) — this
module is the single place that enforces it for every read path.

``resolve_within`` follows symlinks (via ``Path.resolve``) before comparing, so
a symlink inside the worktree that points outside is also rejected. That makes
it filesystem-touching rather than a pure lexical check, which is the correct
trade-off for a security boundary: a lexical-only check is defeated by symlinks.
"""

from pathlib import Path

__all__ = ["resolve_within"]


def resolve_within(base: Path, candidate: str) -> Path | None:
    """Resolve ``candidate`` against ``base``, confined to within ``base``.

    Returns the fully resolved path when it stays inside ``base`` (after symlink
    resolution), or ``None`` when ``candidate`` escapes via ``..`` traversal, an
    absolute path, or a symlink pointing outside. The returned path is not
    guaranteed to exist; callers check existence themselves.
    """
    base_resolved = base.resolve()
    full = (base_resolved / candidate).resolve()
    if full.is_relative_to(base_resolved):
        return full
    return None
