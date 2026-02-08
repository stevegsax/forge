# Forge

Goal: Forge is an LLM task orchestrator built around batch mode with document completion rather than iterative streaming. The core insight is that by investing heavily in upfront planning, we can identify parallelizable work units and submit them as independent batch requests. Each request is a single step in a state machine, not a turn in a conversation.

Forge is suitable for any task that benefits from structured decomposition, parallel execution, and deterministic validation: code generation, research, analysis, content production, data processing, and more. The architecture is task-agnostic -- the differentiation between use cases lives entirely in prompts, context, and validation criteria.

Git and worktrees serve as the general-purpose data store and isolation mechanism. Just as worktrees isolate parallel code branches, they equally isolate parallel research threads, analysis tracks, or any body of work that benefits from independent progress with controlled reconciliation.


