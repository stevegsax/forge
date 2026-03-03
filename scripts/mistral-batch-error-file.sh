#!/usr/bin/env bash
# Download the error_file for a Mistral batch job.
#
# Usage:
#   ./scripts/mistral-batch-error-file.sh <batch-id>
#
# Fetches the batch job, extracts the error_file ID, and downloads it.
# Requires MISTRAL_API_KEY in the environment (or .envrc via direnv).
# Requires jq for JSON parsing.

set -euo pipefail

# Check for jq
if ! command -v jq &>/dev/null; then
    echo "Error: jq is required but not installed" >&2
    exit 1
fi

# Load from .envrc (direnv) if not already exported
if [[ -z "${MISTRAL_API_KEY:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ENVRC_FILE="${SCRIPT_DIR}/../.envrc"
    if [[ -f "$ENVRC_FILE" ]]; then
        eval "$(grep '^export MISTRAL_API_KEY=' "$ENVRC_FILE")"
    fi
fi

if [[ -z "${MISTRAL_API_KEY:-}" ]]; then
    echo "Error: MISTRAL_API_KEY is not set and no .envrc file found" >&2
    exit 1
fi

BATCH_ID="${1:-}"
if [[ -z "$BATCH_ID" ]]; then
    echo "Usage: $0 <batch-id>" >&2
    exit 1
fi

echo "Fetching batch job ${BATCH_ID}..." >&2

JOB_JSON=$(curl -sf \
    -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
    -H "Accept: application/json" \
    "https://api.mistral.ai/v1/batch/jobs/${BATCH_ID}") || {
    echo "Error: failed to fetch batch job ${BATCH_ID}" >&2
    exit 1
}

STATUS=$(echo "$JOB_JSON" | jq -r '.status // empty')
ERROR_FILE=$(echo "$JOB_JSON" | jq -r '.error_file // empty')
ERRORS=$(echo "$JOB_JSON" | jq -c '.errors // empty')
FAILED=$(echo "$JOB_JSON" | jq -r '.failed_requests // empty')
TOTAL=$(echo "$JOB_JSON" | jq -r '.total_requests // empty')

echo "Status:          ${STATUS}" >&2
echo "Failed requests: ${FAILED:-0} / ${TOTAL:-?}" >&2
echo "Errors:          ${ERRORS:-none}" >&2
echo "Error file:      ${ERROR_FILE:-none}" >&2

if [[ -z "$ERROR_FILE" || "$ERROR_FILE" == "null" ]]; then
    echo "No error_file available for this batch." >&2
    exit 0
fi

echo "" >&2
echo "Downloading error file ${ERROR_FILE}..." >&2

curl -sf \
    -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
    "https://api.mistral.ai/v1/files/${ERROR_FILE}/content" || {
    echo "Error: failed to download error file ${ERROR_FILE}" >&2
    exit 1
}
