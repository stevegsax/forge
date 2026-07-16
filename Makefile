# Local stack targets for Forge.
#
# Forge itself is driven with `uv run ...` (see CLAUDE.md); this Makefile only
# wraps the podman-managed local stack in deploy/local-stack — Postgres +
# MinIO + Temporal (the production workflow engine per D99; the app stores'
# production homes stay Supabase/S3). The default test suite runs on SQLite
# and needs none of this.
#
# Typical first run:
#   make stack-up     # start Postgres + Temporal + UI + MinIO (needs podman machine)
#   make db-migrate   # apply Forge's Alembic migrations (needs FORGE_DB_URL set)

.PHONY: help stack-up stack-down stack-logs stack-psql db-migrate

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
	@echo "Local stack (deploy/local-stack): Postgres + MinIO + Temporal"
	@echo "  data dirs: $(PG_DATA_DIR) | $(MINIO_DATA_DIR)"
	@echo "  make stack-up     start the stack (Postgres + Temporal + UI + MinIO + bucket init)"
	@echo "  make stack-down   stop and remove the stack containers"
	@echo "  make stack-logs   tail all containers' logs"
	@echo "  make stack-psql   open a psql shell against the running Postgres"
	@echo "  make db-migrate   apply Forge's Alembic migrations (needs FORGE_DB_URL)"
	@echo "  Temporal UI: http://localhost:$${FORGE_TEMPORAL_UI_PORT:-8233}"
	@echo "  MinIO console: http://localhost:$${FORGE_MINIO_CONSOLE_PORT:-9003} (user/pass: forge / forge-minio-secret)"

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
