# Root targets for Forge: workspace-wide gates (T2.2) and the podman-managed
# local stack in deploy/local-stack — Postgres + MinIO + Temporal (production
# per D99; since T0.9/D102 the app stores' production homes are this stack's
# forge/pbook databases too, with S3 for blobs and nightly dump backups). The
# default test suites run without the stack, except pbook's (its conftest
# needs a podman machine or PBOOK_TEST_DATABASE_URL).
#
# Typical first run:
#   make stack-up     # start Postgres + Temporal + UI + MinIO (needs podman machine)
#   make db-migrate   # apply Forge's Alembic migrations (needs FORGE_DB_URL set)
#   make gates        # everything CI runs: lint, typecheck, lint-imports, test
#
# Per-package suites run from each package's own directory so its own config
# applies (workspace command discipline — see CLAUDE.md).

.PHONY: help lint typecheck lint-imports test gates replay-histories \
	stack-up stack-down stack-logs stack-psql db-migrate backup-app-dbs \
	workers-restart workers-status dev-worker dev-worker-restart

# Bare `make` prints the target list instead of running the first target.
.DEFAULT_GOAL := help

lint:
	uv run ruff check .
	uv run ruff format --check .

# Per-package mypy, from each package's own directory (T2.3b-d append
# theirs as each strictness flip lands).
typecheck:
	uv run mypy
	cd apps/ocr && uv run mypy
	cd apps/pbook && uv run mypy
	cd libs/sax-platform && uv run mypy

lint-imports:
	uv run lint-imports

test:
	uv run pytest
	cd apps/pbook && uv run pytest
	cd apps/ocr && uv run pytest
	cd libs/sax-platform && uv run pytest

gates: lint typecheck lint-imports test

# Regenerate the committed workflow histories tests/test_replay.py replays
# (T4.1 ST4). Runs on the time-skipping test server with mocked activities — no
# real Temporal/DB/S3 — so it is safe against the production-pointing ambient env.
# Regenerate only after a deliberate change to workflow logic.
replay-histories:
	uv run python -m tests.replay.regenerate

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
	@echo "  make typecheck    mypy strict across all four packages"
	@echo "  make lint-imports import-linter DAG contracts (root pyproject)"
	@echo "  make test         all four package suites, each from its own directory"
	@echo "  make replay-histories  regenerate tests/replay/histories/*.json (workflow replay fixtures)"
	@echo ""
	@echo "Local stack (deploy/local-stack): Postgres + MinIO + Temporal"
	@echo "  data dirs: $(PG_DATA_DIR) | $(MINIO_DATA_DIR)"
	@echo "  make stack-up     start the stack (Postgres + Temporal + UI + MinIO + bucket init)"
	@echo "  make stack-down   stop and remove the stack containers"
	@echo "  make stack-logs   tail all containers' logs"
	@echo "  make stack-psql   open a psql shell against the running Postgres"
	@echo "  make db-migrate   apply Forge's Alembic migrations (needs FORGE_DB_URL)"
	@echo "  make backup-app-dbs  pg_dump forge+pbook -> S3 (needs FORGE_ENV set)"
	@echo "  Temporal UI: http://localhost:$${FORGE_TEMPORAL_UI_PORT:-8233}"
	@echo "  MinIO console: http://localhost:$${FORGE_MINIO_CONSOLE_PORT:-9003} (user/pass: forge / forge-minio-secret)"
	@echo ""
	@echo "Workers (launchd-supervised; SIGTERM drains gracefully, KeepAlive restarts)"
	@echo "  make workers-restart   signal forge/ocr/pbook workers; launchd relaunches from disk"
	@echo "  make workers-status    list running worker processes"
	@echo "  make dev-worker        start a staging-lane worker (forge-dev namespace) in detached tmux [WORKER=ocr|forge|pbook]"

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

# Manual run of the nightly offsite backup (the launchd db-backup agent runs it
# at 03:30). Dumps the forge + pbook databases from the forge-postgres container
# to s3://$$FORGE_BACKUP_S3_BUCKET/db-backups/. FORGE_ENV must be set in the
# shell (the script loads the matching profile for the bucket + AWS creds).
backup-app-dbs:
	deploy/local-stack/backup-app-dbs.sh

# Restart the PRODUCTION workers only, resolving each launchd label to its
# pid and signalling that pid — never a command-line pattern. The dev tmux
# workers run byte-identical command lines (the env split lives in
# environment variables, invisible to pkill), so the old pkill-by-pattern
# restart took the staging lane down with production (observed 2026-07-24).
# SIGTERM is deliberate: workers drain gracefully (stop polling, finish
# in-flight activities, exit 0) and launchd's unconditional KeepAlive
# relaunches them from whatever code is on disk. Restart the dev lane
# independently with dev-worker-restart.
WORKER_LABELS = com.saxcapital.forge-worker-1 com.saxcapital.forge-worker-2 \
	com.saxcapital.ocr-worker com.saxcapital.pbook-worker
workers-restart:
	@for label in $(WORKER_LABELS); do \
		info="$$(launchctl print "gui/$$(id -u)/$$label" 2>/dev/null)" \
			|| { echo "$$label: not installed — skipped"; continue; }; \
		pid="$$(printf '%s\n' "$$info" | awk '/[[:space:]]pid = /{print $$3; exit}')"; \
		if [ -n "$$pid" ]; then \
			kill -TERM "$$pid" \
				&& echo "$$label (pid $$pid): SIGTERM — drains, launchd relaunches on current code"; \
		else \
			echo "$$label: installed but not running — launchd relaunch pending"; \
		fi; \
	done
workers-status:
	@echo "prod (launchd):"
	@launchctl list | grep com.saxcapital || echo "  none installed"
	@echo "dev (tmux):"
	@found=0; for s in $$(tmux ls -F '#{session_name}' 2>/dev/null | grep '^dev-' || true); do \
		found=1; \
		if tmux list-panes -t "$$s" -F '#{pane_dead}' | grep -q 1; then \
			echo "  $$s: CRASHED (dead pane holds final output)"; \
		else \
			echo "  $$s: running"; \
		fi; \
	done; [ "$$found" = 1 ] || echo "  none running"

# Staging-lane worker (T0.9 dev namespace) in a detached tmux session, so it
# doesn't hold the terminal. Sources the dev profile with `set -a` (works for
# both plain and export-prefixed profile styles), declares FORGE_ENV=dev, and
# runs the chosen worker. Crash-safe: the session is created with
# remain-on-exit on (a crashed worker leaves a dead pane holding the final
# output instead of vaporizing the session and its scrollback) and the pane
# output is tee'd to a persistent log, so post-mortems survive both crashes
# and kill-session. The FORGE_ENV guard + namespace coherence check abort a
# misconfigured profile (exit 78); the recipe distinguishes running / crashed
# / died-at-startup instead of leaving a silently vanished session.
# Default worker: ocr (make dev-worker WORKER=forge).
WORKER ?= ocr
DEV_PROFILE = $${XDG_CONFIG_HOME:-$$HOME/.config}/forge/envs/dev.env
DEV_WORKER_LOG = $${XDG_STATE_HOME:-$$HOME/.local/state}/forge/logs/dev-$(WORKER)-worker.log
DEV_WORKER_CMD = $(if $(filter ocr,$(WORKER)),uv run --package ocr ocr worker,uv run $(WORKER) worker)
dev-worker:
	@test -f "$(DEV_PROFILE)" || { echo "no dev profile at $(DEV_PROFILE) — copy deploy/launchd/envs/dev.env.example there"; exit 1; }
	@if tmux has-session -t dev-$(WORKER)-worker 2>/dev/null; then \
		if tmux list-panes -t dev-$(WORKER)-worker -F '#{pane_dead}' | grep -q 1; then \
			echo "dev-$(WORKER)-worker CRASHED earlier — the dead pane holds the final output:"; \
			echo "  inspect: tmux attach -t dev-$(WORKER)-worker   (log: $(DEV_WORKER_LOG))"; \
			echo "  then:    tmux kill-session -t dev-$(WORKER)-worker && make dev-worker WORKER=$(WORKER)"; \
			exit 1; \
		else \
			echo "dev-$(WORKER)-worker already running — attach: tmux attach -t dev-$(WORKER)-worker"; \
		fi; \
	else \
		mkdir -p "$$(dirname "$(DEV_WORKER_LOG)")"; \
		tmux new-session -d -s dev-$(WORKER)-worker \
			'set -a; . "$${XDG_CONFIG_HOME:-$$HOME/.config}/forge/envs/dev.env"; set +a; export FORGE_ENV=dev; $(DEV_WORKER_CMD) 2>&1 | tee -a "$(DEV_WORKER_LOG)"' \; \
			set-option -t dev-$(WORKER)-worker remain-on-exit on; \
		sleep 3; \
		if ! tmux has-session -t dev-$(WORKER)-worker 2>/dev/null; then \
			echo "dev-$(WORKER)-worker died before crash-capture engaged — reproduce in the foreground:"; \
			echo "  set -a; source $(DEV_PROFILE); set +a; export FORGE_ENV=dev; $(DEV_WORKER_CMD)"; \
			exit 1; \
		elif tmux list-panes -t dev-$(WORKER)-worker -F '#{pane_dead}' | grep -q 1; then \
			echo "dev-$(WORKER)-worker exited immediately (guard/coherence failure?) — last output:"; \
			tmux capture-pane -t dev-$(WORKER)-worker -pS - | rg -v '^$$' | tail -5; \
			echo "  full log: $(DEV_WORKER_LOG) | clear: tmux kill-session -t dev-$(WORKER)-worker"; \
			exit 1; \
		else \
			echo "dev-$(WORKER)-worker started — attach: tmux attach -t dev-$(WORKER)-worker | stop: tmux kill-session -t dev-$(WORKER)-worker | log: $(DEV_WORKER_LOG)"; \
		fi; \
	fi

# Restart one staging-lane worker (make dev-worker-restart WORKER=forge).
# Kills the tmux session — including a dead crash-capture pane — and starts
# fresh. Complements dev-worker, which refuses to clobber a crashed session
# so its forensics survive. Dev-lane counterpart of workers-restart.
dev-worker-restart:
	@tmux kill-session -t dev-$(WORKER)-worker 2>/dev/null || true
	@$(MAKE) dev-worker WORKER=$(WORKER)
