#!/usr/bin/env bash
#
# gen-ca.sh — create the two internal Certificate Authorities for the Forge
# Temporal mTLS gateway.
#
#   server-ca : signs the gateway's server certificate. Clients trust this CA
#               (FORGE_TEMPORAL_TLS_SERVER_CA) so they can verify the gateway.
#   client-ca : signs per-user client certificates. The gateway trusts this CA
#               (nginx ssl_client_certificate) so only users holding a cert it
#               signed can connect. THIS is the "only authorized users" gate.
#
# Run this ONCE on a trusted operator workstation (or in a vault). The two
# *.key files are the crown jewels: they never go on the EC2 instance and are
# never committed. Only the public certs leave this directory:
#   - server-ca.crt  -> distributed to every client
#   - client-ca.crt  -> uploaded to the instance (gateway verification)
#   - server.crt/key -> uploaded to the instance (see gen-server-cert.sh)
#
# Usage:
#   ./gen-ca.sh [output-dir]        # default: ./ca
# Env:
#   CA_DAYS   CA validity in days (default 3650 = ~10 years)
set -euo pipefail

OUT="${1:-./ca}"
CA_DAYS="${CA_DAYS:-3650}"

mkdir -p "$OUT"

if [[ -f "$OUT/server-ca.key" || -f "$OUT/client-ca.key" ]]; then
  echo "Refusing to overwrite existing CA in $OUT (delete it first if you really mean to)." >&2
  exit 1
fi

echo "==> Generating server CA (signs the gateway server cert)"
openssl genrsa -out "$OUT/server-ca.key" 4096
openssl req -x509 -new -nodes -key "$OUT/server-ca.key" -sha256 -days "$CA_DAYS" \
  -subj "/O=Forge/OU=Temporal/CN=Forge Server CA" \
  -out "$OUT/server-ca.crt"

echo "==> Generating client CA (signs per-user client certs)"
openssl genrsa -out "$OUT/client-ca.key" 4096
openssl req -x509 -new -nodes -key "$OUT/client-ca.key" -sha256 -days "$CA_DAYS" \
  -subj "/O=Forge/OU=Temporal/CN=Forge Client CA" \
  -out "$OUT/client-ca.crt"

chmod 600 "$OUT"/*.key
chmod 644 "$OUT"/*.crt

cat <<EOF

CAs created in $OUT/
  server-ca.crt / server-ca.key   (give server-ca.crt to clients)
  client-ca.crt / client-ca.key   (give client-ca.crt to the instance)

Next:
  ./gen-server-cert.sh <public-dns-or-ip>     # the gateway's server cert
  ./gen-client-cert.sh <username>             # one per authorized user

KEEP *-ca.key OFFLINE AND SECRET. Anyone with client-ca.key can mint a
certificate that the gateway will accept.
EOF
