#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <document_id>" >&2
    exit 1
fi

INPUTJSON=$(jo document_id="$1")
echo "${INPUTJSON}"

temporal workflow start \
  --type OcrExportWorkflow \
  --task-queue forge-task-queue \
  --input "${INPUTJSON}"
