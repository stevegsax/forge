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
#   SAXLLM_REPO_URL, PBOOK_REPO_URL, FORGE_REF, SAXLLM_REF, PBOOK_REF
#   WITH_PBOOK           "true" to deploy pbook + ingestion (default false)
#   DATA_DEVICE          optional EBS device to mount at /data (e.g. /dev/nvme1n1)
set -euo pipefail

APP_DIR="${APP_DIR:-/srv/forge-app}"
export APP_DIR
export SSM_PREFIX="${SSM_PREFIX:-/forge}"
export FORGE_OCR_S3_BUCKET="${FORGE_OCR_S3_BUCKET:-}"
WITH_PBOOK="${WITH_PBOOK:-false}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing runtime dependencies"
dnf install -y git docker
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

# Optional: mount the data EBS volume for worktrees/repos/logs.
if [[ -n "${DATA_DEVICE:-}" && -b "${DATA_DEVICE}" ]]; then
  if ! blkid "${DATA_DEVICE}" >/dev/null 2>&1; then
    mkfs.xfs "${DATA_DEVICE}"
  fi
  mkdir -p /data
  grep -q "${DATA_DEVICE}" /etc/fstab || echo "${DATA_DEVICE} /data xfs defaults,nofail 0 2" >> /etc/fstab
  mount -a
fi
install -d -m 755 /data/forge/logs /data/repos /data/pbook

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
clone_or_pull "${SAXLLM_REPO_URL:-$GH/sax-llm.git}" "$APP_DIR/sax-llm" "${SAXLLM_REF:-main}"
if [[ "$WITH_PBOOK" == "true" ]]; then
  clone_or_pull "${PBOOK_REPO_URL:-$GH/pbook.git}" "$APP_DIR/pbook" "${PBOOK_REF:-main}"
fi

echo "==> Building the Forge venv"
( cd "$APP_DIR/forge" && uv sync --frozen )

if [[ "$WITH_PBOOK" == "true" ]]; then
  echo "==> Running pbook migrations (one-time)"
  ( cd "$APP_DIR/pbook" && PBOOK_DB_PATH=/data/pbook/pbook.db uv run pbook migrate )
fi

echo "==> Starting the Temporal stack (frontend + UI + mTLS gateway)"
( cd "$APP_DIR/forge/deploy/temporal" && docker compose up -d )

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
