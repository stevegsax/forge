#!/usr/bin/env bash
#
# start-stack.sh — launchd RunAtLoad entry point: make sure the podman machine
# is up, then bring up the local stack (Postgres + Temporal + UI + MinIO).
# Containers carry restart policies, so this only needs to run once per boot.
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! podman machine inspect --format '{{.State}}' 2>/dev/null | grep -q running; then
  podman machine start
fi

exec make -C "$REPO_ROOT" stack-up
