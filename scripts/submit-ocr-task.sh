#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <file_path>" >&2
    exit 1
fi

FILEJSON=$(jo file_path="$1")
echo "${FILEJSON}"

temporal workflow start \
  --type OcrSubmitWorkflow \
  --task-queue forge-task-queue \
  --input "${FILEJSON}"

