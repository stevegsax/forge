#!/usr/bin/env nu

# Usage:
#
# ./scripts/submit-ocr-task.nu /path/to/document.pdf
# ./scripts/submit-ocr-task.nu --skip-duplicate-detection /path/to/document.pdf
#



# Submit a document for OCR via the Mistral batch API.
def main [
    file_path: string          # Path to the document file
    --skip-duplicate-detection  # Re-submit even if already OCR'd
] {
    let input = {
        file_path: $file_path
        skip_duplicate_detection: $skip_duplicate_detection
    }

    print ($input | to json)

    temporal workflow execute --type OcrSubmitWorkflow --task-queue forge-task-queue --input ($input | to json)
}
