#!/usr/bin/env bash
# List batch jobs from the Mistral API.
#
# Usage:
#   ./scripts/mistral-batch-jobs.sh              # list all jobs
#   ./scripts/mistral-batch-jobs.sh RUNNING       # filter by status
#   ./scripts/mistral-batch-jobs.sh SUCCESS       # filter by status
#
# Requires MISTRAL_API_KEY in the environment.
#
# Valid statuses: QUEUED, RUNNING, SUCCESS, FAILED,
#                 TIMEOUT_EXCEEDED, CANCELLATION_REQUESTED, CANCELLED

set -euo pipefail

if [[ -z "${MISTRAL_API_KEY:-}" ]]; then
    echo "Error: MISTRAL_API_KEY is not set" >&2
    exit 1
fi

BASE_URL="https://api.mistral.ai/v1/batch/jobs"
STATUS="${1:-}"

if [[ -n "$STATUS" ]]; then
    URL="${BASE_URL}?status=${STATUS}"
else
    URL="${BASE_URL}"
fi

curl -s \
    -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
    -H "Accept: application/json" \
    "$URL" | python -m json.tool
