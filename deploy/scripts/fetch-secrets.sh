#!/usr/bin/env bash
#
# fetch-secrets.sh — pull secrets + TLS material from SSM Parameter Store into
# the locations the workers and the Temporal gateway expect. Runs at boot (via
# bootstrap-instance.sh) using the EC2 instance role; no static AWS keys.
#
# Writes:
#   /etc/forge/forge.env            (chmod 600)  — worker env (API keys, DB URL)
#   /etc/forge/certs/server.crt     gateway server cert
#   /etc/forge/certs/server.key     (chmod 600)  gateway server key
#   /etc/forge/certs/client-ca.crt  CA the gateway verifies clients against
#   $APP_DIR/forge/deploy/temporal/.env  — compose vars (Supabase host/user/pwd)
set -euo pipefail

SSM_PREFIX="${SSM_PREFIX:-/forge}"
APP_DIR="${APP_DIR:-/srv/forge-app}"
OCR_BUCKET="${FORGE_OCR_S3_BUCKET:-}"

# Resolve region from IMDSv2 if not already set.
if [[ -z "${AWS_REGION:-}" ]]; then
  TOKEN="$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" || true)"
  AWS_REGION="$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/placement/region || echo us-east-1)"
fi
export AWS_REGION

ssm() {  # ssm <param-name-without-prefix> ; prints value, empty if absent
  aws ssm get-parameter --with-decryption --name "${SSM_PREFIX}/$1" \
    --query Parameter.Value --output text --region "$AWS_REGION" 2>/dev/null || true
}

install -d -m 700 /etc/forge /etc/forge/certs

# --- worker env ---
umask 077
{
  echo "# Generated at boot by fetch-secrets.sh — do not edit by hand."
  echo "FORGE_DB_URL=$(ssm SUPABASE_FORGE_DB_URL)"
  echo "ANTHROPIC_API_KEY=$(ssm ANTHROPIC_API_KEY)"
  echo "SAX_GITHUB_TOKEN=$(ssm SAX_GITHUB_TOKEN)"
  [[ -n "$OCR_BUCKET" ]] && echo "FORGE_OCR_S3_BUCKET=$OCR_BUCKET"
  m="$(ssm MISTRAL_API_KEY)";  [[ -n "$m" ]] && echo "MISTRAL_API_KEY=$m"
  o="$(ssm OPENAI_API_KEY)";   [[ -n "$o" ]] && echo "OPENAI_API_KEY=$o"
} > /etc/forge/forge.env
chmod 600 /etc/forge/forge.env

# --- compose env for the Temporal stack ---
COMPOSE_ENV="${APP_DIR}/forge/deploy/temporal/.env"
{
  echo "SUPABASE_HOST=$(ssm SUPABASE_HOST)"
  echo "SUPABASE_USER=$(ssm SUPABASE_USER)"
  echo "SUPABASE_TEMPORAL_PWD=$(ssm SUPABASE_TEMPORAL_PWD)"
} > "$COMPOSE_ENV"
chmod 600 "$COMPOSE_ENV"

# --- TLS material for the gateway ---
ssm TLS_SERVER_CERT > /etc/forge/certs/server.crt
ssm TLS_SERVER_KEY  > /etc/forge/certs/server.key
ssm TLS_CLIENT_CA   > /etc/forge/certs/client-ca.crt
chmod 600 /etc/forge/certs/server.key
chmod 644 /etc/forge/certs/server.crt /etc/forge/certs/client-ca.crt

echo "fetch-secrets.sh: wrote /etc/forge/forge.env, compose .env, and /etc/forge/certs/*"
