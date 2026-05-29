#!/usr/bin/env bash
#
# gen-server-cert.sh — issue the Temporal gateway's server certificate,
# signed by the server CA created by gen-ca.sh.
#
# The Subject Alternative Names (SANs) must cover every name/address clients
# will dial. We always include localhost / 127.0.0.1 (for on-box checks) and a
# stable internal name (temporal.forge.internal) so clients can pin a fixed
# server name via FORGE_TEMPORAL_TLS_SERVER_NAME regardless of DNS churn.
#
# Usage:
#   ./gen-server-cert.sh <primary> [extra-SAN ...]
#     <primary>    public DNS name (e.g. temporal.example.com) OR public IP.
#     extra-SAN    additional fully-qualified SAN entries, e.g.
#                  "IP:203.0.113.10" or "DNS:temporal-alt.example.com".
# Env:
#   CA_DIR        directory holding server-ca.{crt,key} (default ./ca)
#   OUT_DIR       output directory (default ./server)
#   SERVER_DAYS   validity in days (default 825; keep <= 825 per CA/B rules)
set -euo pipefail

CA_DIR="${CA_DIR:-./ca}"
OUT="${OUT_DIR:-./server}"
SERVER_DAYS="${SERVER_DAYS:-825}"

PRIMARY="${1:?usage: gen-server-cert.sh <public-dns-or-ip> [extra-SAN ...]}"
shift || true

mkdir -p "$OUT"

# Decide whether $PRIMARY is an IPv4 literal or a DNS name.
if [[ "$PRIMARY" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  PRIMARY_SAN="IP:${PRIMARY}"
else
  PRIMARY_SAN="DNS:${PRIMARY}"
fi

# Base SANs always present, plus the primary, plus any extras passed verbatim.
SANS="${PRIMARY_SAN},DNS:temporal.forge.internal,DNS:localhost,IP:127.0.0.1"
for extra in "$@"; do
  SANS="${SANS},${extra}"
done

EXT="$(mktemp)"
trap 'rm -f "$EXT"' EXIT
cat > "$EXT" <<EOF
basicConstraints=CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=${SANS}
EOF

echo "==> Issuing server cert for: ${PRIMARY}"
echo "    SANs: ${SANS}"
openssl genrsa -out "$OUT/server.key" 2048
openssl req -new -key "$OUT/server.key" \
  -subj "/O=Forge/OU=Temporal/CN=${PRIMARY}" \
  -out "$OUT/server.csr"
openssl x509 -req -in "$OUT/server.csr" \
  -CA "$CA_DIR/server-ca.crt" -CAkey "$CA_DIR/server-ca.key" -CAcreateserial \
  -days "$SERVER_DAYS" -sha256 -extfile "$EXT" \
  -out "$OUT/server.crt"
rm -f "$OUT/server.csr"
chmod 600 "$OUT/server.key"
chmod 644 "$OUT/server.crt"

cat <<EOF

Server cert written to $OUT/
  server.crt  server.key

Upload BOTH to the instance's secret store (SSM SecureString) alongside
client-ca.crt; the gateway loads them into /etc/forge/certs/ at boot.
EOF
