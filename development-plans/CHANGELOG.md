# Changelog

Append-only log of completed tasks. Most recent entries at the bottom.

| Date | Task | Summary | Commit |
|------|------|---------|--------|
| 2026-02-26 | 01 | Extract `detect_file_conflicts` into Temporal activity | 0f8e4ef |
| 2026-02-26 | 02 | Add timeout guards to wait_condition, workflow execution, and worker shutdown | d9d2dd0 |
| 2026-02-26 | 03 | Add explicit retry policies to all 52 execute_activity calls | pending |
| 2026-02-26 | 04 | Extract shared LLM dispatch and remove_worktree into module-level helpers (-96 lines) | pending |
| 2026-02-26 | 05 | Decompose _run_planned into _plan_task and _execute_step_with_retries helpers | cf33267 |
| 2026-02-26 | 06 | Extract shared persist_interaction helper to eliminate 5-way duplication | f9f0826 |
| 2026-02-26 | 07 | Add activity heartbeats to all long-running activities with heartbeat_timeout on invocations | pending |
| 2026-02-26 | 08 | Fix batch poll final_status using cumulative signals_sent instead of per-job count | pending |
