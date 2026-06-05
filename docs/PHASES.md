# Development Phases

The roadmap status for Forge's original 14-phase build: a brief line per **completed** phase (with the modules that implement it) and a detailed summary of what **remains**. This is the historical/roadmap view only.

- Current overall status, requirement completion, and known issues/tech debt → [OVERVIEW.md](OVERVIEW.md)
- Live task list (what's in flight) → [../development-plans/TASKS.md](../development-plans/TASKS.md)

> **Scope:** phases 1–14 only. Substantial work done *outside* this roadmap — the OCR pipeline, store externalization (Postgres + S3 + survivable writes), transcript ingestion, planner evaluation, and secure remote access — is tracked in [OVERVIEW.md](OVERVIEW.md), not here.

Completion is established by verifying against the code, not by prior documentation. "Done" means **built and wired into the worker** (see `src/forge/worker.py`); known gaps and debt within shipped phases live in OVERVIEW, not here. Module paths below are under `src/forge/`.

## Completed (Release 1)

| Phase | Capability | Implemented in |
|------|-----------|----------------|
| 1 | Minimal loop: universal workflow step, Temporal activity boundaries, git worktree lifecycle, OTel tracing | `workflows.py` (`ForgeTaskWorkflow`), `activities/transition.py`, `git.py`, `tracing.py` |
| 2 | Planning / multi-step execution: planner decomposes a task into ordered steps | `activities/planner.py`, `models.py` (`Plan`, `PlanStep`) |
| 3 | Fan-out / gather: parallel sub-tasks via Temporal child workflows; LLM conflict resolution | `workflows.py` (`ForgeSubTaskWorkflow`), `activities/conflict_resolution.py` |
| 4 | Context assembly: import graph, PageRank ranking, token budget | `code_intel/` (`graph.py`, `parser.py`, `repo_map.py`, `budget.py`), `activities/context.py` |
| 5 | Observability store: SQLite/Postgres persistence + Alembic migrations | `store.py`, `persist_models.py`, `alembic/` |
| 6 | Knowledge extraction → playbook entries | `extraction_workflow.py`, `activities/extraction.py`, `manual_playbook_workflow.py`, `export_playbook_workflow.py` |
| 7 | LLM-guided context exploration | `activities/exploration.py` |
| 8 | Error-aware retries: validation errors fed back to the LLM on retry | `activities/validate.py`, `activities/context.py` |
| 9 | Prompt caching (Anthropic `cache_control`) | `llm_client.py`, `llm_providers/anthropic.py` |
| 10 | Fuzzy edit matching: four-level fallback chain | `activities/output.py` (`difflib.SequenceMatcher`) |
| 11 | Model routing by capability tier | `models.py` (`CapabilityTier`), `llm_providers/models.py` (`resolve_model`) |
| 12 | Extended thinking for planning | `activities/planner.py` (`thinking_budget`) |
| 14 | Batch processing via the Anthropic/Mistral Batch API | `batch_poller_workflow.py`, `activities/batch_submit.py`, `activities/batch_poll.py`, `activities/batch_parse.py` |

Phase 13 is intentionally out of sequence — it is deferred (below).

## Remaining

### Phase 13 — Tree-Sitter Multi-Language Support (deferred to Release 2)

**Not implemented.** Context assembly parses **Python only**, via the standard-library `ast` module (`code_intel/parser.py`, `ast.parse()`) — a deliberate choice recorded in DECISIONS (D30, "Python `ast` over tree-sitter for Phase 4"). Phase 13 replaces `ast` with tree-sitter to analyze multiple languages, with graceful degradation for unsupported ones (design recorded in D64–D67).

Detailed design: [planning/PHASE13.md](planning/PHASE13.md).

**Related deferred work** — LSP-based context generation, also a future enhancement (D38, "Defer LSP to a Future Phase"): [planning/LSP_INTEGRATION_PLAN.md](planning/LSP_INTEGRATION_PLAN.md).

> Beyond the numbered roadmap, a richer multi-transform planner is sketched in [planning/task-management/DECOMPOSITION.md](planning/task-management/DECOMPOSITION.md) (a **draft, not implemented** — the shipped planner is the single-pass `activities/planner.py`). It is tracked as future direction in [OVERVIEW.md](OVERVIEW.md), not as a numbered phase.
