#!/usr/bin/env bash
set -euo pipefail

# Submit a document for OCR via the platform batch service.
#
# Usage:
#   ./scripts/submit-ocr-task.sh [--skip-duplicate-detection] <file_path>

SKIP_DUPLICATE_DETECTION=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-duplicate-detection)
            SKIP_DUPLICATE_DETECTION=(--skip-duplicate-detection)
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--skip-duplicate-detection] <file_path>" >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [ $# -ne 1 ]; then
    echo "Usage: $0 [--skip-duplicate-detection] <file_path>" >&2
    exit 1
fi

FILE_PATH="$1"
# Strip leading/trailing whitespace
FILE_PATH="$(echo "$FILE_PATH" | xargs)"
if [ -z "$FILE_PATH" ]; then
    echo "Error: file_path is empty or whitespace-only" >&2
    exit 1
fi

ocr submit "${SKIP_DUPLICATE_DETECTION[@]}" "$FILE_PATH"
