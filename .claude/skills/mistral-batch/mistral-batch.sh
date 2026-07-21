#!/usr/bin/env bash
# Query the Mistral Batch API for job status, errors, and results.
#
# Usage:
#   mistral-batch.sh                  List recent batch jobs
#   mistral-batch.sh <batch_id>       Get detailed status for a specific job
#   mistral-batch.sh errors <file_id> Download and display error file contents
#   mistral-batch.sh output <file_id> Download and display output file contents
set -euo pipefail

# Prefer .envrc key (env may have a stale/wrong key from another session)
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
if [ -f "$REPO_ROOT/.envrc" ]; then
    envrc_key="$(sed -n 's/^export MISTRAL_API_KEY="\(.*\)"/\1/p' "$REPO_ROOT/.envrc")"
    if [ -n "$envrc_key" ]; then
        MISTRAL_API_KEY="$envrc_key"
        export MISTRAL_API_KEY
    fi
fi
if [ -z "${MISTRAL_API_KEY:-}" ]; then
    echo "ERROR: MISTRAL_API_KEY is not set and not found in .envrc" >&2
    exit 1
fi

BASE="https://api.mistral.ai/v1"
AUTH="Authorization: Bearer $MISTRAL_API_KEY"

cmd="${1:-list}"

case "$cmd" in
    list)
        response="$(curl -s -H "$AUTH" "$BASE/batch/jobs")"
        if ! echo "$response" | jq -e '.data' >/dev/null 2>&1; then
            echo "ERROR: API request failed:" >&2
            echo "$response" | jq . 2>/dev/null || echo "$response" >&2
            exit 1
        fi
        echo "$response" | jq '.data[:10] | .[] | {
                id, status, model,
                total_requests, succeeded_requests, failed_requests,
                created_at, completed_at,
                errors: (.errors // []),
                has_output_file: (.output_file != null and .output_file != ""),
                has_error_file: (.error_file != null and .error_file != "")
            }'
        ;;

    errors)
        file_id="${2:?Usage: mistral-batch.sh errors <file_id>}"
        curl -s -H "$AUTH" "$BASE/files/$file_id/content" \
            | while IFS= read -r line; do
                echo "$line" | jq -c '.' 2>/dev/null || echo "$line"
            done
        ;;

    output)
        file_id="${2:?Usage: mistral-batch.sh output <file_id>}"
        curl -s -H "$AUTH" "$BASE/files/$file_id/content" \
            | while IFS= read -r line; do
                echo "$line" | jq -c '{
                    custom_id,
                    status: .response.status_code,
                    has_pages: ((.response.body.pages // []) | length > 0)
                }' 2>/dev/null || echo "$line"
            done
        ;;

    *)
        # Treat argument as a batch ID
        batch_id="$cmd"
        curl -s -H "$AUTH" "$BASE/batch/jobs/$batch_id" | jq .
        ;;
esac
