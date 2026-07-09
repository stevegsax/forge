# Local development database targets for Forge.
#
# Forge itself is driven with `uv run ...` (see CLAUDE.md); this Makefile only
# wraps the podman-managed local Postgres stack in deploy/local-database. Production
# runs on Supabase Postgres and the default test suite runs on SQLite — this
# stack is for running the worker/CLI against a real Postgres locally.
#
# Typical first run:
#   make db-up        # start Postgres (needs a running podman machine)
#   make db-migrate   # apply Forge's Alembic migrations (needs FORGE_DB_URL set)

.PHONY: help db-up db-down db-logs db-psql db-migrate

COMPOSE_DIR := deploy/local-database

# Host directories for the Postgres + MinIO data volumes. XDG-conformant by
# default (spec: $XDG_DATA_HOME, falling back to ~/.local/share); override by
# exporting FORGE_PG_DATA / FORGE_MINIO_DATA. Exported so `podman compose` in
# $(COMPOSE_DIR) picks them up.
XDG_DATA ?= $(if $(XDG_DATA_HOME),$(XDG_DATA_HOME),$(HOME)/.local/share)
PG_DATA_DIR ?= $(XDG_DATA)/forge/postgres
MINIO_DATA_DIR ?= $(XDG_DATA)/forge/minio
export FORGE_PG_DATA = $(PG_DATA_DIR)
export FORGE_MINIO_DATA = $(MINIO_DATA_DIR)

help:
	@echo "Local dev stack (deploy/local-database): Postgres + MinIO"
	@echo "  data dirs: $(PG_DATA_DIR) | $(MINIO_DATA_DIR)"
	@echo "  make db-up        start the stack (Postgres + MinIO + bucket init)"
	@echo "  make db-down      stop and remove the stack containers"
	@echo "  make db-logs      tail the Postgres container logs"
	@echo "  make db-psql      open a psql shell against the running container"
	@echo "  make db-migrate   apply Forge's Alembic migrations (needs FORGE_DB_URL)"
	@echo "  MinIO console: http://localhost:$${FORGE_MINIO_CONSOLE_PORT:-9003} (user/pass: forge / forge-minio-secret)"

db-up:
	@podman machine inspect --format '{{.State}}' 2>/dev/null | grep -q running \
		|| { echo "podman machine is not running. Run 'podman machine start' (or 'podman machine init' if first use)."; exit 1; }
	@mkdir -p "$(PG_DATA_DIR)" "$(MINIO_DATA_DIR)"
	cd $(COMPOSE_DIR) && podman compose up -d

db-down:
	cd $(COMPOSE_DIR) && podman compose down

db-logs:
	cd $(COMPOSE_DIR) && podman compose logs -f postgres

db-psql:
	cd $(COMPOSE_DIR) && podman compose exec postgres psql -U forge -d forge_dev

# No `forge migrate` CLI exists and alembic.ini hardcodes a sqlite URL that
# run_migrations() overrides at runtime, so drive migrations through the same
# entry point the worker uses (forge.store.run_migrations against FORGE_DB_URL).
db-migrate:
	uv run python -c "from forge.store import run_migrations, get_store_url; run_migrations(get_store_url())"
