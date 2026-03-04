#!/usr/bin/env bash
# ocr-results.sh — List OCR'd documents and view extracted text.
#
# Usage:
#   ocr-results.sh                          List all OCR'd documents
#   ocr-results.sh <document_id>            Show extracted text for a document
#   ocr-results.sh <document_id> --images   List images for a document
#   ocr-results.sh <document_id> --export <dir>  Export document with images
#   ocr-results.sh --search <term>          Search extracted text

set -euo pipefail

DB="${FORGE_DB_PATH:-${XDG_STATE_HOME:-$HOME/.local/state}/forge/forge.db}"

if [[ ! -f "$DB" ]]; then
    echo "error: database not found at $DB" >&2
    exit 1
fi

usage() {
    cat <<'EOF'
Usage:
  ocr-results.sh                              List all OCR'd documents
  ocr-results.sh <document_id>                Show extracted text for a document
  ocr-results.sh <document_id> --images       List images for a document
  ocr-results.sh <document_id> --export <dir> Export document with resolved images
  ocr-results.sh --search <term>              Search extracted text for a term
EOF
}

list_documents() {
    sqlite3 -header -column "$DB" "
        SELECT
            r.document_id,
            r.file_path,
            r.page_count,
            length(r.text) AS text_chars,
            r.input_tokens + r.output_tokens AS total_tokens,
            COALESCE(i.image_count, 0) AS images,
            r.created_at
        FROM ocr_results r
        LEFT JOIN (
            SELECT document_id, COUNT(*) AS image_count
            FROM ocr_images
            GROUP BY document_id
        ) i ON r.document_id = i.document_id
        ORDER BY r.created_at DESC;
    "
}

show_document() {
    local doc_id="$1"
    local meta
    meta=$(sqlite3 -json "$DB" "
        SELECT document_id, file_path, page_count, model_name,
               input_tokens, output_tokens, created_at
        FROM ocr_results
        WHERE document_id = '$doc_id';
    ")

    if [[ "$meta" == "[]" || -z "$meta" ]]; then
        echo "error: no OCR result found for document_id '$doc_id'" >&2
        echo >&2
        echo "Available documents:" >&2
        list_documents >&2
        exit 1
    fi

    # Print metadata header
    echo "=== Document Metadata ==="
    sqlite3 -header -column "$DB" "
        SELECT document_id, file_path, page_count, model_name,
               input_tokens, output_tokens, created_at
        FROM ocr_results
        WHERE document_id = '$doc_id';
    "
    echo

    # Show image count
    local img_count
    img_count=$(sqlite3 "$DB" "
        SELECT COUNT(*) FROM ocr_images WHERE document_id = '$doc_id';
    ")
    echo "Images: $img_count"
    echo

    echo "=== Extracted Text ==="
    sqlite3 "$DB" "
        SELECT text FROM ocr_results WHERE document_id = '$doc_id';
    "
}

list_images() {
    local doc_id="$1"
    sqlite3 -header -column "$DB" "
        SELECT
            id,
            original_image_id,
            page_index,
            mime_type,
            file_size_bytes,
            created_at
        FROM ocr_images
        WHERE document_id = '$doc_id'
        ORDER BY page_index, original_image_id;
    "
}

export_document() {
    local doc_id="$1"
    local output_dir="$2"

    # Verify document exists
    local text
    text=$(sqlite3 "$DB" "
        SELECT text FROM ocr_results WHERE document_id = '$doc_id';
    ")
    if [[ -z "$text" ]]; then
        echo "error: no OCR result found for document_id '$doc_id'" >&2
        exit 1
    fi

    mkdir -p "$output_dir/images"

    # Export images and build sed replacement script
    local sed_script=""
    while IFS='|' read -r img_id mime_type; do
        [[ -z "$img_id" ]] && continue
        # Determine file extension from mime type
        local ext="bin"
        case "$mime_type" in
            image/jpeg) ext="jpeg" ;;
            image/png) ext="png" ;;
            image/gif) ext="gif" ;;
            image/webp) ext="webp" ;;
            image/tiff) ext="tiff" ;;
        esac

        local img_file="images/${img_id}.${ext}"

        # Export image blob to file
        sqlite3 "$DB" "
            SELECT writefile('${output_dir}/${img_file}', data)
            FROM ocr_images WHERE id = '${img_id}';
        " > /dev/null

        # Build sed replacement: ocr-image://uuid -> images/uuid.ext
        sed_script="${sed_script}s|ocr-image://${img_id}|${img_file}|g;"
    done < <(sqlite3 -separator '|' "$DB" "
        SELECT id, mime_type FROM ocr_images
        WHERE document_id = '$doc_id'
        ORDER BY page_index;
    ")

    # Write markdown with resolved image references
    if [[ -n "$sed_script" ]]; then
        echo "$text" | sed "$sed_script" > "$output_dir/${doc_id}.md"
    else
        echo "$text" > "$output_dir/${doc_id}.md"
    fi

    echo "Exported to $output_dir/"
    echo "  ${doc_id}.md"
    local img_count
    img_count=$(find "$output_dir/images" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$img_count" -gt 0 ]]; then
        echo "  images/ ($img_count files)"
    else
        rmdir "$output_dir/images" 2>/dev/null || true
    fi
}

search_text() {
    local term="$1"
    sqlite3 -header -column "$DB" "
        SELECT
            document_id,
            file_path,
            page_count
        FROM ocr_results
        WHERE text LIKE '%${term//\'/\'\'}%'
        ORDER BY created_at DESC;
    "
}

if [[ $# -eq 0 ]]; then
    list_documents
elif [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
elif [[ "$1" == "--search" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "error: --search requires a term" >&2
        exit 1
    fi
    search_text "$2"
elif [[ $# -ge 2 && "$2" == "--images" ]]; then
    list_images "$1"
elif [[ $# -ge 3 && "$2" == "--export" ]]; then
    export_document "$1" "$3"
else
    show_document "$1"
fi
