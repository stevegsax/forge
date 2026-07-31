#!/usr/bin/env bash
#
# run-worker.sh <forge|pbook|ocr> [worker-identity]
#
# launchd entry point for a Forge, pbook, or ocr worker. FORGE_ENV comes from
# the agent's launchd plist (EnvironmentVariables: FORGE_ENV=prod, and for the
# workers FORGE_PROD_ACK=yes). load-env.sh then selects and loads the matching
# per-env profile $XDG_CONFIG_HOME/forge/envs/$FORGE_ENV.env (chmod 600), and
# this script execs the worker from the repo root so uv resolves the workspace
# venv.
#
# The profile is parsed line-by-line and NEVER shell-evaluated (G35, T0.7):
# values containing `&`, `;`, `$(...)` etc. are inert. This file stays bash
# (matching install.sh) to minimize churn; the shared load-env.sh avoids
# bashisms, a habit left from when the retired backup job (zsh) sourced it too.
set -euo pipefail

# launchd starts agents with a minimal PATH; uv and podman live in the usual
# user-install locations.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Require FORGE_ENV (from the plist), load + validate the per-env profile, and
# verify FORGE_ENV_TAG matches. Never sets FORGE_PROD_ACK (plist-only).
# shellcheck source=deploy/launchd/load-env.sh
. "$SCRIPT_DIR/load-env.sh"

worker="${1:?usage: run-worker.sh <forge|pbook|ocr> [worker-identity]}"
cd "$REPO_ROOT"

# Optional base identity, for every worker (install.sh passes prod-<agent-name>).
# Each CLI reads FORGE_WORKER_IDENTITY through its own --worker-identity option,
# and the worker appends the launch-time git version, so a poller reports e.g.
# prod-ocr-worker@<sha>. Exported AFTER load-env.sh so the plist argument — the
# thing that names this specific agent — wins over any profile value.
[[ -n "${2:-}" ]] && export FORGE_WORKER_IDENTITY="$2"

case "$worker" in
  forge)
    exec uv run forge worker
    ;;
  pbook)
    exec uv run pbook worker
    ;;
  ocr)
    # ocr is a workspace member, not a forge dependency — --package syncs
    # it into the shared venv (inexactly: nothing else is removed).
    exec uv run --package ocr ocr worker
    ;;
  *)
    echo "run-worker.sh: unknown worker '$worker' (expected forge|pbook|ocr)" >&2
    exit 64  # EX_USAGE
    ;;
esac
