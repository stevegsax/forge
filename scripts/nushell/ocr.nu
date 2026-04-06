# Forge OCR module — composable functions for document OCR via Temporal.
#
# Usage:
#
#   use ocr.nu
#
#   ocr submit ./doc.pdf
#   ocr submit --sync ./receipt.png
#   ocr list | where status == "succeeded" | each { |r| ocr export doc $r.document_id }
#   ocr list | where file_path =~ "signed" | sort-by file_path
#   ls *.pdf | par-each { |f| ocr submit $f.name }
#   ocr mark a1b2c3d4-...
#   ocr unmark a1b2c3d4-...

const TASK_QUEUE = "forge-task-queue"

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Execute a Temporal workflow and return the parsed result record.
def temporal-execute [workflow: string, input: record]: nothing -> record {
    let json_input = ($input | to json --raw)
    let response = (
        ^temporal workflow execute
            --type $workflow
            --task-queue $TASK_QUEUE
            --input $json_input
            --output json
        | complete
    )
    if $response.exit_code != 0 {
        error make { msg: $"temporal workflow execute failed: ($response.stderr | str trim)" }
    }
    $response.stdout | from json | get result
}

# Conditional flag builder: returns [--flag value] or [].
def with-flag [flag: string]: any -> list {
    if ($in | is-empty) { [] } else { [$flag $in] }
}

# ---------------------------------------------------------------------------
# Completions
# ---------------------------------------------------------------------------

def "nu-complete ocr status" [] {
    ["processing", "succeeded", "errored"]
}

# ---------------------------------------------------------------------------
# Exported commands
# ---------------------------------------------------------------------------

# Submit a document for OCR processing.
# Returns a record with document_id. Use --sync to block until complete.
export def submit [
    file_path: path                # Document to OCR
    --sync                         # Block until OCR completes (default: batch)
    --skip-duplicate-detection     # Re-submit even if already OCR'd
]: nothing -> record {
    let workflow = if $sync { "OcrSyncWorkflow" } else { "OcrSubmitWorkflow" }
    temporal-execute $workflow {
        file_path: ($file_path | path expand)
        skip_duplicate_detection: $skip_duplicate_detection
    }
}

# List OCR job submissions as a table.
# Compose with where, sort-by, first for further filtering.
export def list [
    --limit (-l): int = 50                                 # Max results (server-side)
    --status (-s): string@"nu-complete ocr status"         # Filter: processing, succeeded, errored
]: nothing -> table {
    temporal-execute "OcrListJobsWorkflow" {
        limit: $limit
        status_filter: ($status | default "")
    }
    | get jobs
    | update file_path { path basename }
    | update created_at { into datetime }
}

# Export OCR results (text and images) to disk.
# Returns a record with export_dir, markdown_path, image_count.
export def "export doc" [
    document_id: string            # Document to export
    --output-dir (-o): directory   # Override export directory
]: nothing -> record {
    mut input = { document_id: $document_id }
    if ($output_dir != null) {
        $input = ($input | merge { output_dir: $output_dir })
    }
    temporal-execute "OcrExportWorkflow" $input
}

# Mark a document for removal (soft-delete).
export def mark [
    document_id: string            # Document to mark
]: nothing -> record {
    temporal-execute "OcrMarkForRemovalWorkflow" { document_id: $document_id }
}

# Clear a removal mark on a document.
export def unmark [
    document_id: string            # Document to unmark
]: nothing -> record {
    temporal-execute "OcrClearRemovalMarkWorkflow" { document_id: $document_id }
}
