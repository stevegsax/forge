#!/usr/bin/env bash
set -euo pipefail

FORCE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=true
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--force] <file_path>" >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [ $# -ne 1 ]; then
    echo "Usage: $0 [--force] <file_path>" >&2
    exit 1
fi

FILEJSON=$(jo file_path="$1" force="$FORCE")
echo "${FILEJSON}"

temporal workflow start \
  --type OcrSubmitWorkflow \
  --task-queue forge-task-queue \
  --input "${FILEJSON}"
