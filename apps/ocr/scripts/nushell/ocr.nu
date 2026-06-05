#!/usr/bin/env nu
# OCR module — composable functions for document OCR via Temporal.
#
# Usage:
#
#   use ocr.nu
#
# Per-command examples live on each exported command; run
# `ocr <subcommand> --help` to see them.
#
# Workflows are started on the OCR worker's queue (ocr-task-queue); the
# platform worker on forge-task-queue services the batch submit SPI and poller.

const TASK_QUEUE = "ocr-task-queue"

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Indent each line of `text` by four spaces. Empty input renders as
# "(empty)" so error sections never collapse to a blank line.
def indent-block [text: string]: nothing -> string {
    if ($text | is-empty) {
        "    (empty)"
    } else {
        $text | lines | each {|line| $"    ($line)" } | str join "\n"
    }
}

# Raise a structured error for a failed `^temporal` invocation.
#
# Surfaces:
#   - the workflow type and the CLI exit code in the headline;
#   - the task queue and the JSON input we sent (so the request is
#     reproducible without re-running the script);
#   - stderr verbatim (the CLI's terse one-liner, e.g.
#     "Error: workflow failed");
#   - stdout — parsed as JSON when temporal emits its `--output json`
#     failure envelope and rendered as YAML for readability; raw text
#     otherwise.
#
# `help` points at the worker log (the Python traceback lives there)
# and names the failure modes that surface as a generic
# "workflow failed".
# No return-type annotation: this command always raises via `error make`,
# and Nushell has no "never" type to express that — `nothing -> nothing`
# is rejected by the parser because `error make`'s output type is `error`.
def temporal-fail [
    command: string                # "execute" or "start"
    workflow: string               # workflow type name (e.g. "OcrListJobsWorkflow")
    json_input: string             # serialized input record sent to the CLI
    response: record               # output of `^temporal ... | complete`
] {
    let stderr = ($response.stderr | str trim)
    let stdout = ($response.stdout | str trim)
    let parsed = (try { $stdout | from json } catch { null })
    let stdout_block = if ($parsed != null) { $parsed | to yaml } else { $stdout }
    error make {
        msg: ([
            $"temporal workflow ($command) failed: ($workflow) — exit code ($response.exit_code)"
            ""
            $"  task queue: ($TASK_QUEUE)"
            $"  input:      ($json_input)"
            ""
            "  stderr:"
            (indent-block $stderr)
            ""
            "  stdout:"
            (indent-block $stdout_block)
        ] | str join "\n")
        help: ([
            "Inspect the OCR worker's logs for the workflow's Python traceback."
            "Common causes of a generic 'workflow failed':"
            "  • activity result payload >2 MiB (reduce --limit or paginate)"
            "  • an unregistered activity or workflow (restart the worker)"
            "  • an activity timeout"
        ] | str join "\n")
    }
}

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
        temporal-fail "execute" $workflow $json_input $response
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
        temporal-fail "start" $workflow $json_input $response
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
# Returns a record with the started workflow_id. OCR runs via the batch service.
@example "submit a document for batch OCR" { ocr submit ./doc.pdf }
@example "re-submit a document even if already OCR'd" { ocr submit --skip-duplicate-detection ./receipt.png }
@example "submit every PDF in the current directory in parallel" { ls *.pdf | par-each {|f| ocr submit $f.name } }
export def submit [
    file_path: path                # Document to OCR
    --skip-duplicate-detection     # Re-submit even if already OCR'd
]: nothing -> record {
    let input = {
        file_path: ($file_path | path expand)
        skip_duplicate_detection: $skip_duplicate_detection
    }
    temporal-start "OcrSubmitWorkflow" $input
}

# List OCR job submissions as a table.
# Compose with where, sort-by, first for further filtering.
@example "export every succeeded job" { ocr list | where status == "succeeded" | get document_id | each {|id| ocr export $id } }
@example "find signed documents, newest first" { ocr list | where file_path =~ "signed" | sort-by file_path }
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
@example "mark a document for removal" { ocr mark a1b2c3d4-5678-90ab-cdef-1234567890ab }
export def mark [
    document_id: string            # Document to mark
]: nothing -> record {
    temporal-execute "OcrMarkForRemovalWorkflow" { document_id: $document_id }
}

# Clear a removal mark on a document.
@example "clear a document's removal mark" { ocr unmark a1b2c3d4-5678-90ab-cdef-1234567890ab }
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
    --skip-duplicate-detection     # Re-submit even if already OCR'd
] {
    submit $file_path --skip-duplicate-detection=$skip_duplicate_detection
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

# OCR — submit documents for OCR and manage results.
#
# Subcommands: submit, list, export, mark, unmark.
# Run `ocr <subcommand> --help` for details on a specific subcommand.
export def main []: nothing -> nothing {
    print "usage: ocr <submit|list|export|mark|unmark> [args...]"
    print ""
    print "Run `ocr <subcommand> --help` for details on a specific subcommand."
}
