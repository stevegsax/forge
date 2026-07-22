#!/usr/bin/env zsh
#
# backup-app-dbs.sh — nightly offsite durability for the local application
# databases (T0.9). pg_dump -Fc of the `forge` and `pbook` databases out of the
# forge-postgres container, uploaded to s3://$FORGE_BACKUP_S3_BUCKET/db-backups/
# with a UTC timestamp. Local dump files are staged in a temp dir and pruned on
# exit; any failed step aborts the whole run loudly (non-zero exit).
#
# It resolves its environment the same way the workers do: FORGE_ENV comes from
# the launchd plist (prod, with FORGE_PROD_ACK=yes), and load-env.sh loads the
# prod profile (FORGE_BACKUP_S3_BUCKET + AWS creds) and enforces the tag check.
# Run manually with `make backup-app-dbs` (FORGE_ENV must be set in the shell).
set -euo pipefail

# launchd starts agents with a minimal PATH; podman and aws live in the usual
# user-install locations.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

script_dir="${0:A:h}"

# Load + validate the prod env profile (shared with run-worker.sh). Supplies
# FORGE_BACKUP_S3_BUCKET and the AWS_* credentials, and enforces FORGE_ENV /
# FORGE_ENV_TAG agreement. Never sets FORGE_PROD_ACK (that comes from the plist
# or the interactive shell).
. "$script_dir/../launchd/load-env.sh"

: "${FORGE_BACKUP_S3_BUCKET:?backup: FORGE_BACKUP_S3_BUCKET is unset — set it in the $FORGE_ENV profile (deploy/launchd/envs/$FORGE_ENV.env)}"

container="forge-postgres"
databases=(forge pbook)
stamp="$(date -u +%Y%m%dT%H%M%SZ)"

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

if ! podman container inspect "$container" >/dev/null 2>&1; then
  echo "backup: container '$container' not found — is the stack up (make stack-up)?" >&2
  exit 1
fi

echo "backup: starting app-database dump ($stamp) → s3://$FORGE_BACKUP_S3_BUCKET/db-backups/"

for db in $databases; do
  dump="$work_dir/${db}-${stamp}.dump"
  dest="s3://$FORGE_BACKUP_S3_BUCKET/db-backups/${db}-${stamp}.dump"

  echo "backup: dumping database '$db' (pg_dump -Fc)"
  # -Fc: PostgreSQL custom format (compressed, restorable with pg_restore).
  # Connects over the container's local socket as the forge superuser (trust).
  podman exec "$container" pg_dump -Fc -U forge -d "$db" > "$dump"

  size="$(wc -c < "$dump" | tr -d ' ')"
  echo "backup: uploading '$db' ($size bytes) → $dest"
  aws s3 cp "$dump" "$dest"
  echo "backup: '$db' done"
done

echo "backup: complete ($stamp)"
