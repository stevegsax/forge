#!/usr/bin/env bash
#
# prod-deploy.sh <ref>
#
# Deploy production by pinning the prod checkout to a commit (D103). This is the
# only sanctioned way to change what production runs.
#
# Why a separate checkout at all: a worker execs `uv run` inside the checkout it
# was launched from, so that tree's contents at launch ARE the running code.
# While production ran the live working tree, an ordinary edit — or an installer
# run mid-edit, which is how it actually went wrong on 2026-07-25 — shipped
# uncommitted code that no commit described. Production therefore runs a
# dedicated git worktree ($PROD_ROOT) checked out at a *detached* commit:
#
#   - a linked worktree shares the main repo's object store, so a deploy needs no
#     network and can only ever land an object that already exists locally;
#   - detached HEAD makes the pin explicit — the deployed thing is a commit, not
#     a moving branch — and makes an accidental commit in the prod tree obvious;
#   - the workers refuse to start on a dirty or unverifiable checkout
#     (sax_platform.temporal.identity.require_clean_prod_code, exit 78), so the
#     pin is enforced rather than merely intended.
#
# The launchd plists are what bind production to a checkout: run-worker.sh
# computes the repo root from its own location, and install.sh renders every path
# from its own location, so the plists' program paths decide which tree runs.
# This script therefore verifies those paths point at $PROD_ROOT before it
# restarts anything, and refuses (with the exact installer command) when they do
# not — restarting workers whose plists still point at the live tree would
# "deploy" the wrong checkout silently, which is the failure this script exists
# to end.
#
# Usage:
#   deploy/prod-deploy.sh <ref>            # e.g. main, v1.0, a commit sha
#   PROD_ROOT=/tmp/x deploy/prod-deploy.sh <ref>   # override the target checkout
#
# Run it from the MAIN checkout: it rewrites $PROD_ROOT's working tree, and bash
# reads a script incrementally, so a copy running out of $PROD_ROOT could be
# swapped underneath itself mid-run. The script refuses that case.
set -euo pipefail

PROD_ROOT="${PROD_ROOT:-$HOME/repos-sax/forge-prod}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STACK_ENV_REL="deploy/local-stack/.env"
WORKER_PLIST="$HOME/Library/LaunchAgents/com.saxcapital.forge-worker-1.plist"
INSTALLER_FLAGS="--with-pbook --with-ocr --with-backup"

die() { echo "prod-deploy: $*" >&2; exit 1; }
step() { echo "==> $*"; }

[[ $# -eq 1 ]] || { echo "usage: deploy/prod-deploy.sh <ref>" >&2; exit 64; }  # EX_USAGE
REF="$1"

# --- 0. Refuse to deploy from inside the tree being rewritten ----------------
if [[ "$SOURCE_ROOT" == "$PROD_ROOT" ]]; then
  die "run the MAIN checkout's copy of this script — it rewrites $PROD_ROOT, and
    a running bash script whose file changes underneath it can misbehave."
fi

# --- 1. Resolve the ref to a commit in the main repo -------------------------
# ^{commit} rejects anything that is not (or does not peel to) a commit, so a
# typo or a tree/blob name fails here rather than producing a surprising checkout.
COMMIT="$(git -C "$SOURCE_ROOT" rev-parse --verify --quiet "${REF}^{commit}")" \
  || die "'$REF' does not resolve to a commit in $SOURCE_ROOT."
SUBJECT="$(git -C "$SOURCE_ROOT" log -1 --format='%h %s' "$COMMIT")"
step "deploying $REF -> $SUBJECT"

# --- 2. Pin $PROD_ROOT to that commit ----------------------------------------
if [[ -d "$PROD_ROOT" ]]; then
  git -C "$PROD_ROOT" rev-parse --git-dir >/dev/null 2>&1 \
    || die "$PROD_ROOT exists but is not a git checkout — inspect it by hand."
  # Never clobber: local modifications in the prod tree are either an emergency
  # hand-edit or a failed deploy, and both need a human to look before anything
  # is overwritten.
  if [[ -n "$(git -C "$PROD_ROOT" status --porcelain)" ]]; then
    git -C "$PROD_ROOT" status --short >&2
    die "$PROD_ROOT has local modifications (shown above). Inspect and clean it
    (git -C $PROD_ROOT status), then re-run. Nothing was changed."
  fi
  step "checking out $COMMIT in $PROD_ROOT"
  git -C "$PROD_ROOT" checkout --detach --quiet "$COMMIT"
else
  step "creating prod worktree $PROD_ROOT at $COMMIT"
  mkdir -p "$(dirname "$PROD_ROOT")"
  git -C "$SOURCE_ROOT" worktree add --detach "$PROD_ROOT" "$COMMIT"
fi

# --- 3. Untracked state the checkout needs -----------------------------------
# The local stack's port override (FORGE_PG_PORT=5434 on this machine) is
# deliberately gitignored, so a fresh worktree has no copy and would fall back to
# the default port.
if [[ -f "$SOURCE_ROOT/$STACK_ENV_REL" ]]; then
  mkdir -p "$(dirname "$PROD_ROOT/$STACK_ENV_REL")"
  cp "$SOURCE_ROOT/$STACK_ENV_REL" "$PROD_ROOT/$STACK_ENV_REL"
  step "copied $STACK_ENV_REL into the prod checkout"
else
  echo "prod-deploy: WARNING — $SOURCE_ROOT/$STACK_ENV_REL not found, so the prod
    checkout has no local-stack overrides. If this machine runs Postgres on a
    non-default port, a stack brought up from $PROD_ROOT would target the
    default port instead." >&2
fi

# --- 4. Sync the environment the workers will exec ---------------------------
step "uv sync --all-packages in $PROD_ROOT"
(cd "$PROD_ROOT" && uv sync --all-packages)

# --- 5. Verify the plists point at this checkout BEFORE restarting -----------
# A code-only deploy is safe to restart; a plist still pointing at another tree
# means launchd would relaunch the wrong checkout, and no restart can fix that.
installed_program=""
if [[ -f "$WORKER_PLIST" ]]; then
  installed_program="$(plutil -extract ProgramArguments.0 raw -o - "$WORKER_PLIST" 2>/dev/null || true)"
fi

if [[ "$installed_program" == "$PROD_ROOT/"* ]]; then
  step "plists point at $PROD_ROOT — restarting workers (graceful drain, KeepAlive relaunch)"
  make -C "$SOURCE_ROOT" workers-restart
  echo
  step "deployed $SUBJECT"
  echo "    verify: temporal task-queue describe --task-queue forge-task-queue"
  echo "    (identities should read <base>@$(git -C "$PROD_ROOT" rev-parse --short HEAD), with no -dirty)"
else
  echo
  echo "prod-deploy: the checkout is ready at $COMMIT, but the launchd agents still"
  echo "  point at: ${installed_program:-<no com.saxcapital.forge-worker-1 plist installed>}"
  echo "  Workers were NOT restarted — a restart would relaunch that other checkout."
  echo
  echo "  Run the ONE-TIME re-install from the prod checkout (it renders every plist"
  echo "  path from its own location, workers and the forge-stack agent alike):"
  echo
  echo "    $PROD_ROOT/deploy/launchd/install.sh $INSTALLER_FLAGS"
  echo
  echo "  Then this script's restart path takes over for every later deploy."
  exit 3
fi
