# Task Tracker

Source: [Code Review 2026-02-26](../reports/code-review-2026-02-26.md)

## How to Use

See [PROCESS.md](PROCESS.md) for the full workflow. Quick version:

1. Find the next unchecked task below
2. Open its task file for full context
3. Write a Plan and Sub-tasks before coding
4. Update the task file as you work
5. Check off the task here and append to [CHANGELOG.md](CHANGELOG.md) when done

## Tasks

- [x] [01 - Temporal Determinism Violation](01-temporal-determinism.md) `CRITICAL`
- [x] [02 - Wait Condition Timeouts](02-wait-condition-timeouts.md) `HIGH`
- [x] [03 - Retry Policies](03-retry-policies.md) `HIGH`
- [x] [04 - Workflow Duplication](04-workflow-duplication.md) `HIGH`
- [x] [05 - `_run_planned` Decomposition](05-run-planned-decomposition.md) `HIGH`
- [x] [06 - `_persist_interaction` Extraction](06-persist-interaction.md) `HIGH`
- [ ] [07 - Activity Heartbeats](07-activity-heartbeats.md) `HIGH`
- [ ] [08 - Batch Poll Bug](08-batch-poll-bug.md) `MEDIUM`
- [ ] [09 - Model Layer Cleanup](09-model-layer-cleanup.md) `MEDIUM`
- [ ] [10 - Dead Code Cleanup](10-dead-code-cleanup.md) `LOW`

## Dependencies

- Task 04 (workflow duplication) and Task 05 (`_run_planned` decomposition) both modify `workflows.py` heavily. Complete 04 before 05 to avoid merge conflicts.
- Task 01 (determinism) touches `workflows.py` but is a small, surgical change. Complete it first.
- Tasks 02 and 03 (timeouts and retry policies) are independent of each other but both touch `workflows.py`. They are small changes and unlikely to conflict.
- Tasks 08, 09, and 10 are independent of all other tasks.
