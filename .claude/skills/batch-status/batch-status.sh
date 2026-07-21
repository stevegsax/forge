#!/usr/bin/env bash
# Query the Forge batch_jobs ledger and OCR projections for batch job and OCR
# result status. Thin wrapper: cds to the repo root so `uv run python` resolves
# the workspace venv (SQLAlchemy is already a dependency there — nothing extra
# to install), then delegates to batch_status.py for the actual queries.
#
# Usage:
#   batch-status.sh                  Show recent batch jobs and OCR results
#   batch-status.sh summary          Batch job counts grouped by provider and status
#   batch-status.sh failed           Show failed/expired/missing batch jobs
#   batch-status.sh pending          Show submitted (+ legacy processing) batch jobs
#   batch-status.sh ocr <batch_id>   Show OCR results for a provider batch_id
#   batch-status.sh cross-ref        Cross-reference batch jobs with OCR job status
#
# Requires FORGE_DB_URL (see SKILL.md). Read-only against the Forge Postgres
# store — every query in batch_status.py is a SELECT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"
exec uv run python "$SCRIPT_DIR/batch_status.py" "$@"
