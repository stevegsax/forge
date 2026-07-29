# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with pbook.

pbook is a workspace member of the forge monorepo at `apps/pbook` (D98 in the
root `docs/DECISIONS.md`), absorbed with full history from the standalone
`stevegsax/pbook` repo (now archived). The workspace has one root `uv.lock`
and one venv. This member declares no `[tool.uv.sources]` of its own — sibling
sources are declared once at the workspace root.

## Commands

Run all pbook commands from this directory (`apps/pbook/`) so pbook's own
pytest and coverage configuration applies. Never run `pytest apps/pbook/tests`
from the workspace root — the root's addopts (forge's coverage gate) would
apply and the podman-backed conftest would run under the wrong config.

All Python invocations go through `uv` — never call `python` or `pytest` directly.

```bash
uv run pytest                         # full suite (coverage gate: 85%)
uv run pytest tests/test_store.py     # one file
uv run pytest tests/test_store.py::test_save_entries_persists  # one test
uv run pytest -k "duplicate"          # match by substring
uv run ruff check src tests           # lint
uv run ruff format src tests          # format
uv run pbook <subcommand>             # CLI

uv run pbook migrate                  # apply Alembic migrations to the Postgres schema
uv run pbook worker                   # start the Temporal worker on pbook-task-queue
```

The store is PostgreSQL only — since T0.9/D102 (2026-07-22) it is the `pbook` database in the local `forge-postgres` instance (Supabase retired; the `FORGE_ENV` guard and env profiles in the root CLAUDE.md's "Running the System" govern which instance a process reaches). Set `PBOOK_DATABASE_URL` to a `postgresql://` (or `postgresql+psycopg://`) connection string before running `migrate`, the worker, or the CLI; bare `postgresql://` URLs are normalized to the psycopg v3 driver. The worker runs migrations once at startup.

The worker requires **both** LLM API keys in its environment: `ANTHROPIC_API_KEY` (the `sax_platform.llm` client used for extraction, review, and consolidation) and `OPENAI_API_KEY` (embeddings via `text-embedding-3-small`). A missing key no longer hangs — the LLM activities fail fast and non-retryably (see `src/pbook/workflow_steps/retry.py` and `_errors.py`), so an unset key surfaces as a failed workflow / `error` ingestion session rather than one stuck at `running`.

`pytest` runs with `asyncio_mode = "auto"` and a session-scoped event loop. `tests/conftest.py` provisions a real Postgres for the session: it uses `PBOOK_TEST_DATABASE_URL` if set, otherwise starts a `pgvector/pgvector:pg17` container via **podman** (so the test run needs a running podman machine, or that env var). Per-test isolation is a `TRUNCATE ... RESTART IDENTITY` of the `pbk_` tables, so entry ids restart at 1 each test — tests never touch the developer's real database.

`sax_platform` resolves through the workspace root's `[tool.uv.sources]` as a
fellow workspace member (`libs/sax-platform`). Local edits to it are picked up
directly; no tag pin, no re-lock for source changes. (`libs/sax-llm`, pbook's
former LLM provider source, was deleted at T3.5.)

## Architecture

pbook is a knowledge playbook service: it stores curated advice and LLM-extracted "pitfalls" tagged for retrieval into other agents' contexts. Three things define the shape of the codebase.

**Temporal worker on its own queue.** All orchestration runs as Temporal workflows on `pbook-task-queue` (see `src/pbook/worker.py`). Workflows live in `src/pbook/workflows/`, activities in `src/pbook/activities/`. Clients call pbook via cross-queue workflow execution — they never share the queue. The `TranscriptIngestionWorkflow` is the exception: it runs on `forge-task-queue` (forge-side) and calls back into `pbook-task-queue` for extraction. When adding a new workflow, register it in `worker.py` alongside its activities, or it won't be reachable.

**Function Core / Imperative Shell, enforced.** Every module separates pure logic from I/O. Examples: `store.build_entry_dict` (pure) vs `store.save_entries` (I/O); `activities/retrieval.rank_and_pack` (pure) vs `activities/retrieval.fetch_candidates` (I/O). Tests exercise the pure functions directly — they don't mock the database. Keep this split when adding code: a pure function the test can import and call beats a method that requires fixture setup.

**Injected LLM provider via `pbook.roots`.** Since T3.6 there is no provider global: the worker main builds one `sax_platform.llm.AnthropicLLM` provider at startup and injects it into `LlmActivities(provider)` (in `pbook/roots.py`), whose bound `llm_chat` method uses it. The provider is typed by the local `SupportsComplete` Protocol in `pbook/llm.py` (the one `complete` method `llm_chat` needs) — `set_provider`/`get_provider` are gone. Activities are split across three injected classes — `StoreActivities(engine)`, `LlmActivities(provider)`, `EmbeddingActivities(embedder)` — so activities that don't need an LLM (list, get, approve, prune candidate detection) live on `StoreActivities` and stay testable without a provider. Tests construct `LlmActivities` with a fake (e.g. `sax_platform.testing.FakeLLM`).

**Generic LLM/embedding workflow steps via `pbook.workflow_steps`.** Every LLM call goes through `llm_chat` (structured-output chat) or `llm_embed` (text-to-vector) — see `src/pbook/workflow_steps/`. Workflows resolve their model via `pbook.models.resolve_model()` in workflow body, build prompts (pure functions in `src/pbook/prompts/`) via `workflow.unsafe.imports_passed_through()`, call `llm_chat` with an `output_type_name` that keys into the frozen `OUTPUT_TYPES` mapping in `pbook/workflow_steps/output_types.py`, and receive the provider-validated structured output. When adding a new structured output type, add it to that `OUTPUT_TYPES` mapping (or `resolve_output_type` raises `KeyError`) — there is no worker-startup registration step anymore.

### Data model essentials

All tables live in the `pbook` schema and are prefixed `pbk_`. One `pbk_entries` table holds both `pitfall` (extracted) and `curated` (human-submitted) entries — `entry_type` is the discriminator. Tags are namespaced with a controlled vocabulary (`lang:`, `lib:`, `domain:`, `project:`, `pattern:`); see `src/pbook/tags.py` for valid values, and they are normalized into the `pbk_entry_tags` child table (matching is a JOIN, not SQLite `json_each`). Tag validation is enforced on the CLI write path; LLM-extracted tags are tolerated even if imperfect. Store read helpers re-assemble a `tags` list onto each entry dict, so consumers never see a raw `tags_json` column. Each entry stores a pgvector `vector(1536)` embedding (`pbook.store.EMBEDDING_DIM`, for text-embedding-3-small) used for semantic dedup and the `MaintenanceWorkflow`'s consolidation pass; similarity is computed in the database via the `<=>` cosine-distance operator, never row-by-row in Python.

`needs_review=True` is "optimistic review": LLM-extracted entries are visible by default; consumers who don't want them pass `approved_only=True` to retrieval. There is no separate staging table.

The store is configured by `PBOOK_DATABASE_URL` (a PostgreSQL connection string). Setting it empty — or leaving it unset — disables the store entirely: `pbook.store.build_engine(settings)` returns `None`, activities no-op, and `pbook migrate` exits with an error. Alembic uses a custom `version_table = pbk_alembic_version` so pbook's migration chain never collides with another tenant of the same database.

### Retrieval modes

`RetrievalInput.mode` is `CREATE` (boost general knowledge: `lang:`, `lib:`, `domain:`) or `FIX` (boost project-specific pitfalls: `project:`, `pattern:`). Mode reweights ranking only — it never filters. The retrieval workflow packs ranked candidates within a token budget (default 5,000) and records which entries were served so feedback (`pbook feedback`) can later boost or sink them.

### Quality bar (load-bearing)

The extraction prompt is built around: **better to extract nothing than to extract a misleading entry.** Generic advice ("use proper error handling") is rejected; only the unexpected-and-actionable signal counts. When changing extraction or review prompts (`src/pbook/activities/extraction.py`, `src/pbook/activities/review.py`, `src/pbook/ingestion_prompts.py`), preserve this constraint — relaxing it for any one case will degrade the playbook globally.

## Authoritative documentation

Source-of-truth design notes live in `design/` (OVERVIEW, DECISIONS, DATA_MODEL, WORKFLOWS, CLI, INTEGRATION). Read them before changing architecture; update them in the same change as the code.
