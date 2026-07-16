#!/usr/bin/env bash
#
# bootstrap-instance.sh — bring a fresh Amazon Linux 2023 instance to a running
# Forge deployment. Invoked by Terraform user-data AFTER the forge repo is
# cloned to $APP_DIR/forge (so this script, and deploy/, are already present).
#
# It: installs runtime deps, pulls secrets+certs from SSM, clones the sibling
# repos, builds the venv, starts the Temporal stack (incl. the mTLS gateway),
# and installs+starts the worker units.
#
# Inputs (env, set by user-data):
#   APP_DIR              default /srv/forge-app
#   SSM_PREFIX           default /forge
#   FORGE_OCR_S3_BUCKET  required for OCR
#   SAXLLM_REPO_URL, CONTRACTS_REPO_URL, FORGE_REF, SAXLLM_REF, CONTRACTS_REF
#   WITH_PBOOK           "true" to enable the pbook worker unit + migrations
#                        (default false; pbook code ships in the forge checkout
#                        as the apps/pbook workspace member — D98)
#   DATA_DEVICE          optional EBS device to mount at /data (e.g. /dev/nvme1n1)
set -euo pipefail

APP_DIR="${APP_DIR:-/srv/forge-app}"
export APP_DIR
export SSM_PREFIX="${SSM_PREFIX:-/forge}"
export FORGE_OCR_S3_BUCKET="${FORGE_OCR_S3_BUCKET:-}"
WITH_PBOOK="${WITH_PBOOK:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing runtime dependencies"
# ripgrep: the search_code exploration provider shells out to `rg` (T1.4).
dnf install -y git docker ripgrep
systemctl enable --now docker
# docker compose plugin
mkdir -p /usr/libexec/docker/cli-plugins
if ! docker compose version >/dev/null 2>&1; then
  curl -fsSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
    -o /usr/libexec/docker/cli-plugins/docker-compose
  chmod +x /usr/libexec/docker/cli-plugins/docker-compose
fi
# uv (system-wide, brings managed Python 3.12)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi

# Unprivileged service account for the worker units (T1.7). HOME is the app dir
# so uv's managed Python + cache land on a forge-owned path (chown below) that
# survives the units' ProtectHome=true. The dir already exists (repo cloned
# here), so do not --create-home.
getent group forge >/dev/null 2>&1 || groupadd --system forge
getent passwd forge >/dev/null 2>&1 || \
  useradd --system --gid forge --home-dir "$APP_DIR" --shell /sbin/nologin forge

# Optional: mount the data EBS volume for worktrees/repos/logs.
if [[ -n "${DATA_DEVICE:-}" && -b "${DATA_DEVICE}" ]]; then
  if ! blkid "${DATA_DEVICE}" >/dev/null 2>&1; then
    mkfs.xfs "${DATA_DEVICE}"
  fi
  mkdir -p /data
  grep -q "${DATA_DEVICE}" /etc/fstab || echo "${DATA_DEVICE} /data xfs defaults,nofail 0 2" >> /etc/fstab
  mount -a
fi
install -d -m 755 /data/forge/logs /data/repos

echo "==> Fetching secrets + TLS material from SSM"
bash "${SCRIPT_DIR}/fetch-secrets.sh"

# Make the GitHub token available for cloning private sibling repos.
# shellcheck disable=SC1091
set -a; source /etc/forge/forge.env; set +a
GH="https://x-access-token:${SAX_GITHUB_TOKEN}@github.com/stevegsax"

clone_or_pull() {  # clone_or_pull <url> <dir> <ref>
  local url="$1" dir="$2" ref="$3"
  if [[ -d "$dir/.git" ]]; then
    git -C "$dir" fetch --depth 1 origin "$ref" && git -C "$dir" checkout -f "$ref"
  else
    git clone "$url" "$dir" && git -C "$dir" checkout -f "$ref"
  fi
}

echo "==> Cloning sibling repositories"
# The root pyproject's editable path sources resolve as ../sax-llm and
# ../forge-contracts relative to $APP_DIR/forge, so both siblings must be
# present for uv sync. pbook is no longer a sibling: it ships inside the
# forge checkout as the apps/pbook workspace member (D98).
clone_or_pull "${SAXLLM_REPO_URL:-$GH/sax-llm.git}" "$APP_DIR/sax-llm" "${SAXLLM_REF:-main}"
clone_or_pull "${CONTRACTS_REPO_URL:-$GH/forge-contracts.git}" "$APP_DIR/forge-contracts" "${CONTRACTS_REF:-main}"

echo "==> Building the Forge venv"
# HOME=$APP_DIR so uv's managed Python + cache install under the (soon
# forge-owned) app dir, where the hardened units can reach them at runtime.
( cd "$APP_DIR/forge" && HOME="$APP_DIR" uv sync --frozen )

if [[ "$WITH_PBOOK" == "true" ]]; then
  echo "==> Running pbook migrations (one-time)"
  # PBOOK_DATABASE_URL comes from forge.env (fetch-secrets.sh emits it from
  # SSM SUPABASE_PBOOK_DB_URL); the store is Postgres-only.
  : "${PBOOK_DATABASE_URL:?PBOOK_DATABASE_URL must be set when WITH_PBOOK=true (create SSM ${SSM_PREFIX}/SUPABASE_PBOOK_DB_URL)}"
  ( cd "$APP_DIR/forge" && HOME="$APP_DIR" uv run pbook migrate )
fi

echo "==> Starting the Temporal stack (frontend + UI + mTLS gateway)"
( cd "$APP_DIR/forge/deploy/temporal" && docker compose up -d )

echo "==> Handing worker-owned paths to the forge account"
# The hardened units run as forge with ProtectSystem=strict + ReadWritePaths on
# these two trees. Everything root built above (venv, managed Python, cache,
# repos, logs) must be forge-owned so the unprivileged worker can read and
# write it.
chown -R forge:forge "$APP_DIR" /data

echo "==> Installing systemd worker units"
sed "s/forge-ocr-blobs-CHANGEME/${FORGE_OCR_S3_BUCKET}/" \
  "$APP_DIR/forge/deploy/systemd/forge-worker@.service" \
  > /etc/systemd/system/forge-worker@.service
if [[ "$WITH_PBOOK" == "true" ]]; then
  cp "$APP_DIR/forge/deploy/systemd/pbook-worker.service" /etc/systemd/system/
fi
systemctl daemon-reload
systemctl enable --now forge-worker@1 forge-worker@2
[[ "$WITH_PBOOK" == "true" ]] && systemctl enable --now pbook-worker

echo "==> Bootstrap complete."
