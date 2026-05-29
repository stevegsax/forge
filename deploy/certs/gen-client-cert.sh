#!/usr/bin/env bash
#
# gen-client-cert.sh — issue one authorized user's client certificate, signed
# by the client CA. Possession of this cert (+ key) is what lets a user's
# forge / pbook / ocr CLI connect through the mTLS gateway.
#
# Issue one per person (CN = their username). To revoke a user, stop trusting
# their cert: rotate the client CA and re-issue everyone (simplest for a small
# team), or run a CRL/OCSP (see deploy/certs/README.md). Keep client certs
# short-lived (default 1 year) so lost certs age out.
#
# Usage:
#   ./gen-client-cert.sh <username>
# Env:
#   CA_DIR        directory holding client-ca.{crt,key} (default ./ca)
#   OUT_DIR       output directory (default ./clients)
#   CLIENT_DAYS   validity in days (default 365)
set -euo pipefail

CA_DIR="${CA_DIR:-./ca}"
OUT="${OUT_DIR:-./clients}"
CLIENT_DAYS="${CLIENT_DAYS:-365}"

USERNAME="${1:?usage: gen-client-cert.sh <username>}"
mkdir -p "$OUT"

EXT="$(mktemp)"
trap 'rm -f "$EXT"' EXIT
cat > "$EXT" <<EOF
basicConstraints=CA:FALSE
keyUsage=critical,digitalSignature
extendedKeyUsage=clientAuth
subjectAltName=DNS:${USERNAME}.client.forge
EOF

echo "==> Issuing client cert for user: ${USERNAME}"
openssl genrsa -out "$OUT/${USERNAME}.key" 2048
openssl req -new -key "$OUT/${USERNAME}.key" \
  -subj "/O=Forge/OU=Users/CN=${USERNAME}" \
  -out "$OUT/${USERNAME}.csr"
openssl x509 -req -in "$OUT/${USERNAME}.csr" \
  -CA "$CA_DIR/client-ca.crt" -CAkey "$CA_DIR/client-ca.key" -CAcreateserial \
  -days "$CLIENT_DAYS" -sha256 -extfile "$EXT" \
  -out "$OUT/${USERNAME}.crt"
rm -f "$OUT/${USERNAME}.csr"
chmod 600 "$OUT/${USERNAME}.key"
chmod 644 "$OUT/${USERNAME}.crt"

cat <<EOF

Client cert written to $OUT/
  ${USERNAME}.crt   ${USERNAME}.key

Deliver these two files to ${USERNAME} over a secure channel, together with
server-ca.crt. They configure (see deploy/client/ONBOARDING.md):
  FORGE_TEMPORAL_TLS=1
  FORGE_TEMPORAL_TLS_SERVER_CA=.../server-ca.crt
  FORGE_TEMPORAL_TLS_CLIENT_CERT=.../${USERNAME}.crt
  FORGE_TEMPORAL_TLS_CLIENT_KEY=.../${USERNAME}.key
EOF
