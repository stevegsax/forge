# Workers

## Overview

A Forge worker is a long-running process that polls the Temporal server for queued workflows and executes activities (LLM calls, context assembly, validation, git operations). The worker itself is stateless — all workflow state lives in the Temporal server. This means multiple workers can run on different machines, and if a worker crashes, another can pick up where it left off.

Start a worker:

```bash
forge worker
forge worker --temporal-address temporal.example.com:7233
```

The worker polls the `forge-task-queue` task queue. All Forge workflows and activities are registered on this single queue.

## Checking Whether Workers Are Running

Workers do not write a PID file or store local state. Because workers can run on any machine that can reach the Temporal server, the Temporal server itself is the source of truth.

### Temporal CLI

List active pollers on the Forge task queue:

```bash
temporal task-queue describe --task-queue forge-task-queue
```

This shows all workers currently polling the queue, including their identity, last access time, and the workflow/activity types they handle. If the list is empty, no workers are running.

### Temporal Web UI

The Temporal Web UI (default `http://localhost:8233`) shows active workers under the task queue view. Navigate to the `forge-task-queue` task queue to see connected pollers.

### Local process check

If you only need to check the local machine:

```bash
pgrep -f "forge worker"
```

## Worker Identity

Each worker reports an identity string to the Temporal server. By default, the Python SDK sets this to `{pid}@{hostname}`. This identity appears in workflow history events and in the task queue poller list, so you can trace which worker executed a given activity.

The default identity is adequate for single-machine development. In multi-machine or containerized deployments, the default is often unhelpful (container PIDs are always `1`, cloud hostnames are random strings). In those environments, set a custom identity that maps back to the machine or deployment unit (e.g., ECS task ID, k8s pod name).

Set a custom identity via the `--worker-identity` flag or `FORGE_WORKER_IDENTITY` environment variable:

```bash
forge worker --worker-identity "worker-us-east-1a-01"
FORGE_WORKER_IDENTITY="pod-abc123" forge worker
```

When omitted, the SDK default (`{pid}@{hostname}`) is used.

## Scaling

Multiple workers can poll the same task queue from different machines. The Temporal server distributes work across them automatically. Workers are stateless, so scaling out is as simple as starting more `forge worker` processes pointed at the same Temporal server.

Deploy at least two workers per task queue for redundancy.

## Further Reading

- [What is a Temporal Worker?](https://docs.temporal.io/workers) — Worker concepts, identity, and configuration.
- [Worker deployment and performance](https://docs.temporal.io/best-practices/worker) — Production deployment, monitoring metrics, and task queue separation.
- [Worker Versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning) — Pinning workflow versions to specific worker builds for safe rollouts.
