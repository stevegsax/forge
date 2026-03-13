#!/usr/bin/env bash
set -euo pipefail

SKIP_DUPLICATE_DETECTION=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-duplicate-detection)
            SKIP_DUPLICATE_DETECTION=true
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

FILEJSON=$(jo file_path="$1" skip_duplicate_detection="$SKIP_DUPLICATE_DETECTION")
echo "${FILEJSON}"

temporal workflow start \
  --type OcrSubmitWorkflow \
  --task-queue forge-task-queue \
  --input "${FILEJSON}" \
  --await-result
