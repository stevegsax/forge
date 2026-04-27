#!/usr/bin/env nu
# Forge OCR module — composable functions for document OCR via Temporal.
#
# Usage:
#
#   use ocr.nu
#
#   ocr submit ./doc.pdf
#   ocr submit --sync ./receipt.png
#   ocr list | where status == "succeeded" | get document_id | each {|id| ocr export $id }
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

# Start a Temporal workflow without waiting and return the workflow ID.
def temporal-start [workflow: string, input: record]: nothing -> record {
    let json_input = ($input | to json --raw)
    let workflow_id = $"($workflow | str downcase)-(random chars --length 8)"
    let response = (
        ^temporal workflow start
            --type $workflow
            --task-queue $TASK_QUEUE
            --workflow-id $workflow_id
            --input $json_input
        | complete
    )
    if $response.exit_code != 0 {
        error make { msg: $"temporal workflow start failed: ($response.stderr | str trim)" }
    }
    { workflow_id: $workflow_id }
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
    let input = {
        file_path: ($file_path | path expand)
        skip_duplicate_detection: $skip_duplicate_detection
    }
    if $sync {
        temporal-execute "OcrSyncWorkflow" $input
    } else {
        temporal-start "OcrSubmitWorkflow" $input
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
export def export [
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

# ---------------------------------------------------------------------------
# Script entry point (when run via `./ocr.nu <subcommand> ...`)
# ---------------------------------------------------------------------------

def "main submit" [
    file_path: path                # Document to OCR
    --sync                         # Block until OCR completes
    --skip-duplicate-detection     # Re-submit even if already OCR'd
] {
    submit $file_path --sync=$sync --skip-duplicate-detection=$skip_duplicate_detection
}

def "main list" [
    --limit (-l): int = 50                          # Max results (server-side)
    --status (-s): string                           # Filter: processing, succeeded, errored
] {
    if ($status | is-empty) {
        list --limit $limit
    } else {
        list --limit $limit --status $status
    }
}

def "main export" [
    document_id: string            # Document to export
    --output-dir (-o): directory   # Override export directory
] {
    mut input = { document_id: $document_id }
    if ($output_dir != null) {
        $input = ($input | merge { output_dir: $output_dir })
    }
    temporal-execute "OcrExportWorkflow" $input
}

def "main mark" [document_id: string] {
    mark $document_id
}

def "main unmark" [document_id: string] {
    unmark $document_id
}

def main [] {
    print "usage: ocr.nu <submit|list|export|mark|unmark> [args...]"
    print "  or:  use ocr.nu  (to load as a module)"
}
