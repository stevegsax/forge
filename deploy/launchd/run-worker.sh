#!/usr/bin/env bash
#
# run-worker.sh <forge|pbook> [worker-identity]
#
# launchd entry point for a Forge or pbook worker. Loads secrets/config from
# $XDG_CONFIG_HOME/forge/forge.env (chmod 600), then execs the worker from the
# repo root so uv resolves the workspace venv.
#
# The env file is parsed line-by-line and NEVER shell-evaluated (G35, T0.7):
# values containing `&`, `;`, `$(...)` etc. are inert. Lines are KEY=VALUE;
# blank lines and #-comments are skipped; anything else aborts loudly.
set -euo pipefail

# launchd starts agents with a minimal PATH; uv and podman live in the usual
# user-install locations.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/forge/forge.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "run-worker.sh: $ENV_FILE not found — copy deploy/launchd/forge.env.example there (chmod 600)" >&2
  exit 78  # EX_CONFIG
fi
perms="$(stat -f '%Lp' "$ENV_FILE")"
if [[ "$perms" != "600" ]]; then
  echo "run-worker.sh: $ENV_FILE must be chmod 600 (is $perms) — it holds API keys" >&2
  exit 78
fi

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  key="${line%%=*}"
  value="${line#*=}"
  if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "run-worker.sh: malformed line in $ENV_FILE (not KEY=VALUE): ${line%%=*}" >&2
    exit 78
  fi
  export "$key=$value"
done < "$ENV_FILE"

worker="${1:?usage: run-worker.sh <forge|pbook> [worker-identity]}"
cd "$REPO_ROOT"

case "$worker" in
  forge)
    [[ -n "${2:-}" ]] && export FORGE_WORKER_IDENTITY="$2"
    exec uv run forge worker
    ;;
  pbook)
    exec uv run pbook worker
    ;;
  *)
    echo "run-worker.sh: unknown worker '$worker' (expected forge|pbook)" >&2
    exit 64  # EX_USAGE
    ;;
esac
