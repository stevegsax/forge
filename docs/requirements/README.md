# Forge Requirements

Gherkin-format behavioral requirements for the Forge orchestrator. These serve as living
documentation and a foundation for future BDD test automation with
[pytest-bdd](https://pytest-bdd.readthedocs.io/).

Going forward, Gherkin is only one half of the requirement package. Each reviewed requirement
should pair a behavioral example file (`<requirement-id>.feature`) with a structured requirement
core (`<requirement-id>.core.md`). The core carries typed contracts, functional-core /
imperative-shell boundaries, non-functional constraints, and review metadata. See:

- [STANDARD.md](STANDARD.md) — autonomous-agent requirements standard
- [TEMPLATE.md](TEMPLATE.md) — starting point for new requirement core files
- [REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md) — approval gate before implementation handoff
- [examples/README.md](examples/README.md) — worked examples of a complete reviewed requirement package

## Target Directory Structure

```text
docs/requirements/
├── README.md
├── STANDARD.md
├── TEMPLATE.md
├── REVIEW_CHECKLIST.md
├── examples/
│   ├── README.md
│   ├── inspira.feature
│   ├── inspira.core.md
│   └── inspira.review.md
├── task_execution.feature
├── task_execution.core.md
├── planning.feature
├── planning.core.md
├── fan_out.feature
├── fan_out.core.md
├── human_in_the_loop.feature
├── context_assembly.feature
├── context_assembly.core.md
├── exploration.feature
├── exploration.core.md
├── validation.feature
├── validation.core.md
├── output_processing.feature
├── output_processing.core.md
├── batch_processing.feature
├── batch_processing.core.md
├── git_operations.feature
├── git_operations.core.md
├── model_routing.feature
├── model_routing.core.md
├── llm_providers.feature
├── llm_providers.core.md
├── knowledge_management.feature
├── knowledge_management.core.md
├── ocr_pipeline.feature
├── ocr_pipeline.core.md
├── ocr_cli.feature
├── ocr_web_api.feature
├── cli.feature
├── cli.core.md
├── observability.feature
└── observability.core.md
```

The `.core.md` files are the source of truth for reviewed requirements. The `.feature` files remain
focused on examples and externally visible behavior. This repository does not yet contain paired
core files for every feature; the standard above defines the target format for new or revised
requirements.

## Tag Taxonomy

### Phase Tags

Map scenarios to implementation phases:

- `@phase-1` through `@phase-14`

### Priority Tags

Every scenario has exactly one priority tag:

- `@critical` — Core behaviors; failure means the system is broken
- `@standard` — Expected behaviors; important but not catastrophic if missing
- `@edge-case` — Boundary conditions and unusual inputs

### Capability Tags

Map scenarios to system capabilities:

- `@task-execution` — Universal workflow step, transition logic, retry loops
- `@planning` — Planner decomposition, step execution, sanity checks
- `@fan-out` — Parallel child workflows, conflict detection/resolution
- `@context` — Import graph, PageRank, token budget, priority ordering
- `@exploration` — LLM-guided context gathering, 12 providers
- `@validation` — Fix-then-check pipeline, ruff, tests, error-aware retries
- `@output` — File writing, 4-level edit matching, path traversal protection
- `@batch` — Batch submission, polling, multi-provider support
- `@git` — Worktree lifecycle, branch naming, commits
- `@model-routing` — Capability tiers, default models, CLI overrides
- `@llm-providers` — Provider abstraction, Anthropic/Mistral, feature degradation
- `@knowledge` — Extraction, playbooks, tag inference, context injection
- `@ocr` — Document OCR, PDF chunking, blob storage, image extraction, chunk reassembly
- `@cli` — CLI commands, flags, exit codes
- `@observability` — SQLite store, 7 tables, XDG paths, best-effort writes

### Cross-Cutting Tags

- `@retry` — Retry-related scenarios
- `@error-handling` — Error handling and fault tolerance
- `@temporal` — Temporal workflow/activity integration
- `@deterministic` — Pure, deterministic behaviors

## File-to-Capability Mapping

| File | Capabilities | Key Source Files |
|------|-------------|-----------------|
| `task_execution.feature` | Core workflow step, transitions, domains | `workflows.py`, `activities/transition.py`, `domains.py` |
| `planning.feature` | Planner, step execution, sanity checks | `activities/planner.py`, `activities/sanity_check.py` |
| `fan_out.feature` | Parallel sub-tasks, conflict resolution | `workflows.py`, `activities/conflict_resolution.py` |
| `human_in_the_loop.feature` | Workflow pause/resume, structured human input *(not implemented)* | planned — only batch/OCR signals exist (`workflows.py`) |
| `context_assembly.feature` | Import graph, PageRank, token budget | `activities/context.py`, `code_intel/` |
| `exploration.feature` | LLM-guided exploration, 12 providers | `activities/exploration.py`, `providers.py` |
| `validation.feature` | Fix-then-check, error-aware retries | `activities/validate.py`, `activities/context.py` |
| `output_processing.feature` | File writing, edit matching fallback | `activities/output.py` |
| `batch_processing.feature` | Batch submission, polling, multi-provider | `batch_poller_workflow.py`, `activities/batch_poll.py` |
| `git_operations.feature` | Worktree lifecycle, branch naming | `git.py`, `activities/git_activities.py` |
| `model_routing.feature` | Capability tiers, model resolution | `models.py` |
| `llm_providers.feature` | Provider abstraction, degradation | `llm_providers/` |
| `knowledge_management.feature` | Extraction, playbooks, injection | `extraction_workflow.py`, `activities/extraction.py` |
| `ocr_pipeline.feature` | OCR workflows, PDF chunking, blobs | `ocr/` |
| `ocr_cli.feature` | OCR job listing/status CLI | `cli.py` (`ocr-jobs`), `ocr/workflow_list_jobs.py` |
| `ocr_web_api.feature` | OCR OpenAPI web service, pagination *(not implemented)* | none — no web framework in repo |
| `cli.feature` | CLI commands, flags, exit codes | `cli.py` |
| `observability.feature` | SQLite store, 7 tables, migrations | `store.py` |

> **Implementation status:** 16 of these 18 specs are implemented. `ocr_web_api.feature` and `human_in_the_loop.feature` describe capabilities that are **not yet built** — see [../OVERVIEW.md](../OVERVIEW.md).

## Conventions

- **Domain language** in step text, not implementation class names
- **One When per scenario**
- **Quoted strings** for significant values: `Given a task with id "my-task"`
- **Angle brackets** for Scenario Outline parameters: `<status>`
- **Background** sections only for universal preconditions within a file
- **Comment headers** (`# --- Section ---`) group related scenarios within a file

## Paired Requirement Rules

- Every reviewed requirement package uses the same requirement ID in both filenames.
- The `.feature` file contains examples, not authoritative contracts.
- The `.core.md` file contains the contract map, domain algebra, capability boundaries, and review
  metadata.
- Every mandatory section in the `.core.md` file must be present, even if the value is
  `N/A — none, because ...`.
- A requirement package is not ready for autonomous implementation until the review checklist is
  completed and the core status is `approved`.

## Syntax Validation

```bash
pip install gherkin-official
python -c "
from gherkin.parser import Parser
from glob import glob
p = Parser()
for f in sorted(glob('docs/requirements/*.feature')):
    p.parse(open(f).read())
    print(f'OK: {f}')
print('All files valid.')
"
```

## Future: pytest-bdd Integration

These feature files are designed for compatibility with pytest-bdd's `parsers.parse()` step matcher. To wire up tests:

1. Install: `pip install pytest-bdd`
2. Create step definitions in `tests/bdd/` matching the step text
3. Run by tag: `pytest -m "critical" --bdd`
4. Run by capability: `pytest -k "model_routing" --bdd`
