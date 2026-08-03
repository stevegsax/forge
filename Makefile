# Root targets for Forge: workspace-wide gates (T2.2), migrations, and the two
# worker lanes. Forge owns no infrastructure (T10.1/D104): Postgres comes from
# the shared sax-datastores stacks (dev :5432, prod :5442) and Temporal from
# sax-temporal (dev :7236, prod :7243), both of which boot themselves. The
# default test suites need neither, except pbook's (its conftest needs a podman
# machine or PBOOK_TEST_DATABASE_URL).
#
# Typical first run:
#   make db-migrate   # apply Forge's Alembic migrations (needs FORGE_DB_URL set)
#   make gates        # everything CI runs: lint, typecheck, lint-imports, test
#
# Per-package suites run from each package's own directory so its own config
# applies (workspace command discipline — see CLAUDE.md).

.PHONY: help lint typecheck lint-imports lint-sql test gates replay-histories \
	db-migrate db-change workers-restart workers-status dev-worker dev-worker-restart \
	prod-deploy

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

# Lint every committed schema-change artifact with the pinned Squawk and the
# vendored .squawk.toml (a byte-copy of the operator's canonical config, so a
# "clean lint" claim in a request means clean against the rules the operator
# reviews with). Part of `gates`, so an artifact can never be committed
# unlinted.
#
# The version pin lives in ONE place — SQUAWK_VERSION in
# sax_platform.db.change_request, which `make db-change` also uses — and is
# read out of Python here rather than copied, because a request records the
# Squawk version it was linted with and two drifting pins would make that
# record a lie.
#
# Squawk itself exits 1 on "no files matched a pattern", so the file list is
# built here and an empty list short-circuits: a repo with no artifacts yet
# passes rather than failing on the linter's own argument handling.
SQL_ARTIFACT_GLOBS = datastore-changes/*/change-*.sql apps/pbook/datastore-changes/*/change-*.sql
lint-sql:
	@files=$$(ls -1 $(SQL_ARTIFACT_GLOBS) 2>/dev/null); \
	if [ -z "$$files" ]; then \
		echo "lint-sql: no datastore-changes artifacts — nothing to lint"; exit 0; \
	fi; \
	ver=$$(uv run python -c 'from sax_platform.db.change_request import SQUAWK_VERSION; print(SQUAWK_VERSION)') \
		|| { echo "lint-sql: could not read SQUAWK_VERSION from sax_platform.db.change_request"; exit 1; }; \
	echo "lint-sql: squawk-cli@$$ver over"; printf '  %s\n' $$files; \
	npx --yes squawk-cli@$$ver --config .squawk.toml $$files

test:
	uv run pytest
	cd apps/pbook && uv run pytest
	cd apps/ocr && uv run pytest
	cd libs/sax-platform && uv run pytest

gates: lint typecheck lint-imports lint-sql test

# Regenerate the committed workflow histories tests/test_replay.py replays
# (T4.1 ST4). Runs on the time-skipping test server with mocked activities — no
# real Temporal/DB/S3 — so it is safe against the production-pointing ambient env.
# Regenerate only after a deliberate change to workflow logic.
replay-histories:
	uv run python -m tests.replay.regenerate

help:
	@echo "Gates (T2.2): what CI runs"
	@echo "  make gates        lint + typecheck + lint-imports + lint-sql + test"
	@echo "  make lint         ruff check + format --check (workspace-wide)"
	@echo "  make typecheck    mypy strict across all four packages"
	@echo "  make lint-imports import-linter DAG contracts (root pyproject)"
	@echo "  make lint-sql     squawk over datastore-changes/**/change-*.sql (pinned version)"
	@echo "  make test         all four package suites, each from its own directory"
	@echo "  make replay-histories  regenerate tests/replay/histories/*.json (workflow replay fixtures)"
	@echo ""
	@echo "Store (shared sax-datastores Postgres; forge starts no stack of its own)"
	@echo "  make db-migrate   apply Forge's Alembic migrations (needs FORGE_DB_URL)"
	@echo "  make db-change CHAIN=forge|ocr|pbook FROM=<rev> [TO=<rev>] TITLE=<kebab-title>"
	@echo "                    generate a sax-datastores change request (offline SQL + request.md)"
	@echo ""
	@echo "Workers (launchd-supervised; SIGTERM drains gracefully, KeepAlive restarts)"
	@echo "  make prod-deploy REF=<ref>  pin the prod worktree to a commit and restart (D103)"
	@echo "  make workers-restart   signal forge/ocr/pbook workers; launchd relaunches from disk"
	@echo "  make workers-status    list running worker processes"
	@echo "  make dev-worker        start a staging-lane worker (forge-dev namespace) in detached tmux [WORKER=ocr|forge|pbook]"

# Thin wrapper over `forge migrate` (the CLI command; it applies the forge
# chain to FORGE_DB_URL and prints a credential-free target line). The old
# `python -c` form called forge.store.get_store_url, which T3.6 deleted with the
# module-global seams, so this target had been broken since then.
#
# Workers do NOT migrate: since the 2026-08-02 schema-change agreement they
# verify their chain at startup and refuse to start when the database is behind.
# This target is the dev self-service apply path. Whichever database
# FORGE_DB_URL names is what it touches — since T10.1/D104 that is a
# sax-datastores instance — so declare the environment (FORGE_ENV / a sourced
# profile, or `forge migrate --env dev`) before running it. Production schema
# changes never come through here: they go through the sax-datastores
# change-request process and the administrator applies them.
db-migrate:
	uv run forge migrate

# Generate a sax-datastores change request from one of the three Alembic
# chains: offline SQL per phase (transaction wrappers stripped, each phase's
# version-table stamp kept) plus a prefilled request.md, under
# datastore-changes/ (forge and ocr, one shared id sequence — both are product
# `forge`) or apps/pbook/datastore-changes/ (pbook, its own sequence).
#
# It touches no database and needs no FORGE_ENV: it reads the chain off disk
# and writes files. Production DDL is never applied from here — the generated
# artifacts are committed (the commit is the request) and the sax-datastores
# administrator applies them (sax-datastores/docs/schema-changes.md).
DB_CHANGE_TO = $(if $(TO),--to $(TO),)
DB_CHANGE_USAGE = usage: make db-change CHAIN=forge|ocr|pbook FROM=<rev> [TO=<rev>] TITLE=<kebab-title>
db-change:
	@test -n "$(CHAIN)" && test -n "$(FROM)" && test -n "$(TITLE)" \
		|| { echo "$(DB_CHANGE_USAGE)"; exit 64; }
	@case "$(CHAIN)" in \
		forge) cmd="uv run forge db-change";; \
		ocr)   cmd="uv run --package ocr ocr db-change";; \
		pbook) cmd="uv run --package pbook pbook db-change";; \
		*) echo "unknown CHAIN=$(CHAIN)"; echo "$(DB_CHANGE_USAGE)"; exit 64;; \
	esac; \
	$$cmd --from "$(FROM)" $(DB_CHANGE_TO) --title "$(TITLE)"

# Deploy production from a pinned commit (D103): checks REF out into the
# forge-prod worktree, syncs it, and restarts the launchd workers — the only
# sanctioned way to change what production runs. Thin wrapper; the mechanism and
# its guards live in deploy/prod-deploy.sh.
prod-deploy:
	@test -n "$(REF)" || { echo "usage: make prod-deploy REF=<ref>   (e.g. REF=main)"; exit 64; }
	deploy/prod-deploy.sh "$(REF)"

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
# both plain and export-prefixed profile styles), declares FORGE_ENV=dev plus the
# lane's base Temporal identity (FORGE_WORKER_IDENTITY=dev-<worker>-worker, which
# the worker version-stamps to dev-ocr-worker@<sha> — same string as the tmux
# session name here and as the row `make workers-status` prints, so the lane a
# poller belongs to reads the same everywhere), and runs the chosen worker.
# Both exports land AFTER the profile source, so the lane can never be
# contradicted by a stale value inside dev.env. Crash-safe: the session is created with
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
			'set -a; . "$${XDG_CONFIG_HOME:-$$HOME/.config}/forge/envs/dev.env"; set +a; export FORGE_ENV=dev FORGE_WORKER_IDENTITY=dev-$(WORKER)-worker; $(DEV_WORKER_CMD) 2>&1 | tee -a "$(DEV_WORKER_LOG)"' \; \
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
