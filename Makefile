# Root targets for Forge: workspace-wide gates (T2.2) and the podman-managed
# local stack in deploy/local-stack — Postgres + MinIO + Temporal (the
# production workflow engine per D99; the app stores' production homes stay
# Supabase/S3). The default test suites run without the stack, except pbook's
# (its conftest needs a podman machine or PBOOK_TEST_DATABASE_URL).
#
# Typical first run:
#   make stack-up     # start Postgres + Temporal + UI + MinIO (needs podman machine)
#   make db-migrate   # apply Forge's Alembic migrations (needs FORGE_DB_URL set)
#   make gates        # everything CI runs: lint, typecheck, lint-imports, test
#
# Per-package suites run from each package's own directory so its own config
# applies (workspace command discipline — see CLAUDE.md).

.PHONY: help lint typecheck lint-imports test gates \
	stack-up stack-down stack-logs stack-psql db-migrate \
	workers-restart workers-status

# Bare `make` prints the target list instead of running the first target.
.DEFAULT_GOAL := help

lint:
	uv run ruff check .
	uv run ruff format --check .

# Per-package mypy, from each package's own directory (T2.3b-d append
# theirs as each strictness flip lands).
typecheck:
	uv run mypy
	cd libs/forge-contracts && uv run mypy
	cd libs/sax-llm && uv run mypy
	cd apps/ocr && uv run mypy
	cd apps/pbook && uv run mypy
	cd libs/sax-platform && uv run mypy

lint-imports:
	uv run lint-imports

test:
	uv run pytest
	cd apps/pbook && uv run pytest
	cd apps/ocr && uv run pytest
	cd libs/sax-llm && uv run pytest
	cd libs/forge-contracts && uv run pytest
	cd libs/sax-platform && uv run pytest

gates: lint typecheck lint-imports test

COMPOSE_DIR := deploy/local-stack

# Host directories for the Postgres + MinIO data volumes. XDG-conformant by
# default (spec: $XDG_DATA_HOME, falling back to ~/.local/share); override by
# exporting FORGE_PG_DATA / FORGE_MINIO_DATA. Exported so `podman compose` in
# $(COMPOSE_DIR) picks them up. Temporal has no volume — its state lives in
# Postgres.
XDG_DATA ?= $(if $(XDG_DATA_HOME),$(XDG_DATA_HOME),$(HOME)/.local/share)
PG_DATA_DIR ?= $(XDG_DATA)/forge/postgres
MINIO_DATA_DIR ?= $(XDG_DATA)/forge/minio
export FORGE_PG_DATA = $(PG_DATA_DIR)
export FORGE_MINIO_DATA = $(MINIO_DATA_DIR)

help:
	@echo "Gates (T2.2): what CI runs"
	@echo "  make gates        lint + typecheck + lint-imports + test"
	@echo "  make lint         ruff check + format --check (workspace-wide)"
	@echo "  make typecheck    mypy strict across all six packages"
	@echo "  make lint-imports import-linter DAG contracts (root pyproject)"
	@echo "  make test         all five package suites, each from its own directory"
	@echo ""
	@echo "Local stack (deploy/local-stack): Postgres + MinIO + Temporal"
	@echo "  data dirs: $(PG_DATA_DIR) | $(MINIO_DATA_DIR)"
	@echo "  make stack-up     start the stack (Postgres + Temporal + UI + MinIO + bucket init)"
	@echo "  make stack-down   stop and remove the stack containers"
	@echo "  make stack-logs   tail all containers' logs"
	@echo "  make stack-psql   open a psql shell against the running Postgres"
	@echo "  make db-migrate   apply Forge's Alembic migrations (needs FORGE_DB_URL)"
	@echo "  Temporal UI: http://localhost:$${FORGE_TEMPORAL_UI_PORT:-8233}"
	@echo "  MinIO console: http://localhost:$${FORGE_MINIO_CONSOLE_PORT:-9003} (user/pass: forge / forge-minio-secret)"
	@echo ""
	@echo "Workers (launchd-supervised; SIGTERM drains gracefully, KeepAlive restarts)"
	@echo "  make workers-restart   signal forge/ocr/pbook workers; launchd relaunches from disk"
	@echo "  make workers-status    list running worker processes"

stack-up:
	@podman machine inspect --format '{{.State}}' 2>/dev/null | grep -q running \
		|| { echo "podman machine is not running. Run 'podman machine start' (or 'podman machine init' if first use)."; exit 1; }
	@mkdir -p "$(PG_DATA_DIR)" "$(MINIO_DATA_DIR)"
	cd $(COMPOSE_DIR) && podman compose up -d

stack-down:
	cd $(COMPOSE_DIR) && podman compose down

stack-logs:
	cd $(COMPOSE_DIR) && podman compose logs -f

stack-psql:
	cd $(COMPOSE_DIR) && podman compose exec postgres psql -U forge -d forge_dev

# No `forge migrate` CLI exists and alembic.ini hardcodes a sqlite URL that
# run_migrations() overrides at runtime, so drive migrations through the same
# entry point the worker uses (forge.store.run_migrations against FORGE_DB_URL).
db-migrate:
	uv run python -c "from forge.store import run_migrations, get_store_url; run_migrations(get_store_url())"

# Every launchd-supervised worker now drains gracefully on SIGTERM (exit 0)
# instead of dying on Python's default handler — launchd's unconditional
# KeepAlive then relaunches it from whatever code is on disk. Leading `-`
# so an absent worker (e.g. pbook, which is opt-in) doesn't fail the target.
workers-restart:  # signal all workers; launchd KeepAlive restarts them on current on-disk code
	-pkill -TERM -f "uv run forge worker"
	-pkill -TERM -f "uv run --package ocr ocr worker"
	-pkill -TERM -f "uv run pbook worker"
workers-status:
	-pgrep -fl "uv run forge worker|uv run --package ocr ocr worker|uv run pbook worker"
