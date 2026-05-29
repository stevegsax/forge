#!/usr/bin/env bash
#
# smoke-test.sh — quick post-deploy checks. Run on the instance (via SSM).
set -euo pipefail
APP_DIR="${APP_DIR:-/srv/forge-app}"

echo "== docker services =="
( cd "$APP_DIR/forge/deploy/temporal" && docker compose ps )

echo "== Temporal namespace (via loopback) =="
docker run --rm --network host temporalio/admin-tools:1.25.2 \
  temporal operator namespace list --address 127.0.0.1:7233 || true

echo "== forge task queue =="
( cd "$APP_DIR/forge" && uv run forge status --limit 3 ) || true

echo "== gateway is listening on :443 =="
ss -tlnp | grep ':443' || echo "WARN: nothing listening on 443"

echo "== mTLS gate rejects a certless client (expect TLS handshake failure) =="
if echo | openssl s_client -connect 127.0.0.1:443 -servername temporal.forge.internal 2>&1 \
   | grep -qiE 'alert|handshake failure|certificate required|verify error'; then
  echo "OK: connection without a client cert was rejected."
else
  echo "WARN: certless connection was NOT clearly rejected — check ssl_verify_client."
fi
