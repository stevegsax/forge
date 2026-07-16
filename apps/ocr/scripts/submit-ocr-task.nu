#!/usr/bin/env nu

# Usage:
#
# ./scripts/submit-ocr-task.nu /path/to/document.pdf
# ./scripts/submit-ocr-task.nu --skip-duplicate-detection /path/to/document.pdf
#



# Submit a document for OCR via the platform batch service.
def main [
    file_path: string          # Path to the document file
    --skip-duplicate-detection  # Re-submit even if already OCR'd
] {
    if $skip_duplicate_detection {
        ocr submit --skip-duplicate-detection $file_path
    } else {
        ocr submit $file_path
    }
}
