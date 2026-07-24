"""Temporal activities for Forge workflow steps.

Two kinds are registered by ``forge.worker``:

- **Free-function activities** — no process-wide dependency to inject (git,
  validate, output, the pure ``assemble_*_context`` prompt builders,
  ``detect_file_conflicts_activity``, ``validate_playbook_entry``). Exported
  here as functions. (Transition evaluation is no longer an activity — it is the
  pure ``forge.step_logic.determine_transition``, inlined into the workflows,
  D95/T5.1.)
- **Composition-root classes** (T3.6, ``forge.activities.roots``) — every
  activity that carries a dependency (store engine, LLM client, batch SDK
  client, blob store, Temporal client, OCR client) is a bound method on
  ``StoreActivities`` / ``ContextActivities`` / ``LlmActivities`` /
  ``BatchActivities``, constructed once in the worker main. The bound methods
  keep the former activity names, so workflows (which invoke by string) are
  unaffected.
"""

from __future__ import annotations

from forge.activities.conflict_resolution import (
    assemble_conflict_resolution_context,
    detect_file_conflicts_activity,
)
from forge.activities.exploration import assemble_exploration_context
from forge.activities.git_activities import (
    commit_changes_activity,
    create_worktree_activity,
    remove_worktree_activity,
    reset_worktree_activity,
)
from forge.activities.output import write_files, write_output
from forge.activities.planner import assemble_planner_context
from forge.activities.playbook_review import validate_playbook_entry
from forge.activities.roots import (
    BatchActivities,
    ContextActivities,
    LlmActivities,
    StoreActivities,
)
from forge.activities.sanity_check import assemble_sanity_check_context
from forge.activities.validate import validate_output

__all__ = [
    "BatchActivities",
    "ContextActivities",
    "LlmActivities",
    "StoreActivities",
    "assemble_conflict_resolution_context",
    "assemble_exploration_context",
    "assemble_planner_context",
    "assemble_sanity_check_context",
    "commit_changes_activity",
    "create_worktree_activity",
    "detect_file_conflicts_activity",
    "remove_worktree_activity",
    "reset_worktree_activity",
    "validate_output",
    "validate_playbook_entry",
    "write_files",
    "write_output",
]
