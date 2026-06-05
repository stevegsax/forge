#!/usr/bin/env bash
set -euo pipefail

# Export OCR text + images for a document to the filesystem.
#
# Usage:
#   ./scripts/export-ocr-document.sh <document_id>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <document_id>" >&2
    exit 1
fi

ocr export "$1"
