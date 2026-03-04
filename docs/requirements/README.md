# Forge Requirements

Gherkin-format behavioral requirements for the Forge orchestrator. These serve as living documentation and a foundation for future BDD test automation with [pytest-bdd](https://pytest-bdd.readthedocs.io/).

## Directory Structure

```
docs/requirements/
├── README.md
├── task_execution.feature
├── planning.feature
├── fan_out.feature
├── context_assembly.feature
├── exploration.feature
├── validation.feature
├── output_processing.feature
├── batch_processing.feature
├── git_operations.feature
├── model_routing.feature
├── llm_providers.feature
├── knowledge_management.feature
├── ocr_pipeline.feature
├── cli.feature
└── observability.feature
```

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
- `@ocr` — Document OCR, PDF chunking, blob storage, chunk reassembly
- `@cli` — CLI commands, flags, exit codes
- `@observability` — SQLite store, 6 tables, XDG paths, best-effort writes

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
| `cli.feature` | CLI commands, flags, exit codes | `cli.py` |
| `observability.feature` | SQLite store, 6 tables, migrations | `store.py` |

## Conventions

- **Domain language** in step text, not implementation class names
- **One When per scenario**
- **Quoted strings** for significant values: `Given a task with id "my-task"`
- **Angle brackets** for Scenario Outline parameters: `<status>`
- **Background** sections only for universal preconditions within a file
- **Comment headers** (`# --- Section ---`) group related scenarios within a file

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
