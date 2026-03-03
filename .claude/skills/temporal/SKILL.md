---
name: temporal
description: Inspect Temporal workflows and activities. Use when the user asks about workflow status, failures, errors, execution history, or activity results from the Temporal server.
argument-hint: [workflow-id] [run-id]
allowed-tools: Bash(temporal *), Bash(temporal * | jq *)
---

Use the `temporal` CLI to inspect Temporal workflows. The Temporal server runs at `localhost:7233` by default.

## Arguments

- `$ARGUMENTS[0]` — workflow ID (optional; if omitted, list recent workflows)
- `$ARGUMENTS[1]` — run ID (optional; narrows to a specific execution)

## What to do

### No arguments: list recent workflows

```bash
temporal workflow list --limit 10
```

### Workflow ID provided: show execution history

```bash
temporal workflow show --workflow-id $0
```

If the workflow failed, extract structured failure details using `jq`. Choose which command to run based on the text history output:

- If the text history contains `WorkflowTaskFailed` → run the **workflow task failure** command
- If the text history contains `ActivityTaskFailed` → run the **activity failure** command
- If both appear or it is unclear → run both

**Workflow task failures** (import errors, non-determinism, sandbox violations):

```bash
temporal workflow show --workflow-id $0 -o json | jq '
  [.events[]
   | select(.workflowTaskFailedEventAttributes)
   | .workflowTaskFailedEventAttributes.failure
   | {message, stackTrace,
      cause: (.cause // null | if . then {message, stackTrace} else null end)}]'
```

**Activity failures** (exceptions raised inside activities):

```bash
temporal workflow show --workflow-id $0 -o json | jq '
  [.events[]
   | select(.activityTaskFailedEventAttributes)
   | .activityTaskFailedEventAttributes
   | {activityType: .activityType.name,
      failure: .failure
        | {message, stackTrace,
           cause: (.cause // null | if . then {message, stackTrace} else null end)}}]'
```

When a run ID is provided (`$1`), add `--run-id $1` to all commands above.

### Both workflow ID and run ID provided

```bash
temporal workflow show --workflow-id $0 --run-id $1
```

### Describe workflow metadata

Always run this to show the current state, type, start time, and close status:

```bash
temporal workflow describe --workflow-id $0
```

## Interpreting results

- **WorkflowTaskFailed** with `RestrictedWorkflowAccessError` — code inside a `@workflow.defn` called a non-deterministic function (e.g. `uuid.uuid4()`, `datetime.now()`, file I/O). Fix by using Temporal's deterministic alternatives (`workflow.uuid4()`, `workflow.now()`) or moving the operation to an activity.
- **ActivityTaskFailed** — an activity raised an exception. The failure message and stack trace identify the root cause.
- **WorkflowExecutionTimedOut** — the workflow exceeded its execution timeout.
- **ContinuedAsNew** — the workflow continued as a new execution (normal for long-running pollers).

## Useful variations

List only failed workflows:

```bash
temporal workflow list --query "ExecutionStatus = 'Failed'" --limit 10
```

List workflows by type:

```bash
temporal workflow list --query "WorkflowType = 'OcrSubmitWorkflow'" --limit 10
```
