#!/usr/bin/env bash
# Query the Forge observability database for batch job and OCR result status.
#
# Usage:
#   batch-status.sh                  Show recent batch jobs and OCR results
#   batch-status.sh summary          Batch job counts grouped by provider and status
#   batch-status.sh failed           Show failed/errored batch jobs
#   batch-status.sh pending          Show submitted-but-not-completed batch jobs
#   batch-status.sh ocr <batch_id>   Show OCR results for a specific batch
#   batch-status.sh cross-ref        Cross-reference batch jobs with OCR results
set -euo pipefail

DB="${FORGE_DB_PATH:-${XDG_STATE_HOME:-$HOME/.local/state}/forge/forge.db}"

if [ ! -f "$DB" ]; then
    echo "ERROR: Forge database not found at $DB" >&2
    echo "Set FORGE_DB_PATH or check that Forge has been initialised." >&2
    exit 1
fi

cmd="${1:-default}"

case "$cmd" in
    default)
        echo "=== Recent Batch Jobs (last 20) ==="
        sqlite3 -header -column "$DB" "
            SELECT id, COALESCE(batch_id, '(none)') AS batch_id, provider, status, error_message,
                   datetime(created_at) AS created, datetime(updated_at) AS updated
            FROM batch_jobs
            ORDER BY updated_at DESC
            LIMIT 20;
        "

        echo ""
        echo "=== Recent OCR Results (last 20) ==="
        sqlite3 -header -column "$DB" "
            SELECT document_id, file_path, page_count, model_name,
                   input_tokens, output_tokens, batch_id,
                   datetime(created_at) AS created
            FROM ocr_results
            ORDER BY created_at DESC
            LIMIT 20;
        "
        ;;

    summary)
        sqlite3 -header -column "$DB" "
            SELECT provider, status, COUNT(*) AS count
            FROM batch_jobs
            GROUP BY provider, status
            ORDER BY provider, status;
        "
        ;;

    failed)
        sqlite3 -header -column "$DB" "
            SELECT id, COALESCE(batch_id, '(none)') AS batch_id, provider,
                   COALESCE(file_path, '(unknown)') AS file_path, error_message,
                   datetime(updated_at) AS updated
            FROM batch_jobs
            WHERE status NOT IN ('submitted', 'succeeded')
            ORDER BY updated_at DESC;
        "
        ;;

    pending)
        sqlite3 -header -column "$DB" "
            SELECT id, batch_id, provider,
                   datetime(created_at) AS created,
                   ROUND((julianday('now') - julianday(created_at)) * 24, 1) AS hours_ago
            FROM batch_jobs
            WHERE status = 'submitted'
            ORDER BY created_at;
        "
        ;;

    ocr)
        batch_id="${2:?Usage: batch-status.sh ocr <batch_id>}"
        sqlite3 -header -column "$DB" "
            SELECT document_id, file_path, page_count, length(text) AS text_chars,
                   input_tokens, output_tokens
            FROM ocr_results
            WHERE batch_id = '$batch_id';
        "
        ;;

    cross-ref)
        sqlite3 -header -column "$DB" "
            SELECT b.id, b.batch_id, b.provider, b.status,
                   CASE WHEN o.document_id IS NOT NULL THEN 'yes' ELSE 'no' END AS has_ocr_result,
                   o.document_id, o.file_path
            FROM batch_jobs b
            LEFT JOIN ocr_results o ON b.batch_id = o.batch_id
            ORDER BY b.updated_at DESC
            LIMIT 30;
        "
        ;;

    *)
        echo "Unknown command: $cmd" >&2
        echo "Usage: batch-status.sh [default|summary|failed|pending|cross-ref|ocr <batch_id>]" >&2
        exit 1
        ;;
esac
