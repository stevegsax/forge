# Nushell scripts

- The scripts in this directory are written using [Nushell](https://www.nushell.sh/)
- A Nushell command list is available at [Nushell command list](https://www.nushell.sh/commands/)
- Nushell documentation is available at [The Nushell Book](https://www.nushell.sh/book/)
- A Nushell cookbook is available at [Nushell Cookbook](https://www.nushell.sh/cookbook/)
- The Nushell language reference is available at [Nushell Language Reference](https://www.nushell.sh/lang-guide/)
- An example collection of nushell scripts is available on this computer in directory `/Users/stevengreenberg/repos-other/nu_scripts`

## Loading modules

Modules are loaded with `use`. Nushell resolves module paths at parse time, so paths must be known before execution. Set `NU_LIB_DIRS` to include this directory:

```bash
NU_LIB_DIRS="scripts/nushell" nu
```

Or from within nushell:

```nu
$env.NU_LIB_DIRS = ($env.NU_LIB_DIRS | append "scripts/nushell")
use ocr.nu
```

To make this permanent, add the path to `NU_LIB_DIRS` in `~/.config/nushell/env.nu`.

## Writing modules

### Structure

- Single file modules (e.g. `ocr.nu`) for focused functionality
- Directory modules (`mymodule/mod.nu` with submodules) when a module exceeds ~500 lines
- `export def` for public commands, `def` for private helpers
- `const` for module-level configuration, `let` for runtime values

### Calling external commands

Use the `^` prefix to invoke external CLI tools:

```nu
^temporal workflow execute --type $workflow --output json
```

Capture stdout, stderr, and exit code with `complete`:

```nu
let response = (
    ^temporal workflow execute
        --type $workflow
        --input $json_input
        --output json
    | complete
)
if $response.exit_code != 0 {
    error make { msg: $"Command failed: ($response.stderr | str trim)" }
}
$response.stdout | from json
```

### Composable design

Every exported function should return structured data (records or tables), not printed text. This enables pipeline composition:

```nu
# Filtering, sorting, and chaining happen via nushell operators
ocr list | where status == "succeeded" | sort-by file_path | first 10

# One function's output feeds another's input
ocr list | each { |r| ocr export doc $r.document_id }

# Parallel execution over structured data
ls *.pdf | par-each { |f| ocr submit $f.name }
```

### Tab completion

Define completion functions with `nu-complete` prefix and attach via `@`:

```nu
def "nu-complete ocr status" [] {
    ["processing", "succeeded", "errored"]
}

export def list [
    --status (-s): string@"nu-complete ocr status"
] { ... }
```

### Type annotations

Use pipeline signatures and typed parameters for documentation and validation:

```nu
export def submit [
    file_path: path          # path type enables path expansion
    --sync                   # bool flag
    --limit (-l): int = 50   # typed with default
]: nothing -> record {       # pipeline signature: takes nothing, returns record
    ...
}
```

Use `into datetime`, `into filesize`, etc. to convert string fields into proper types so nushell renders them idiomatically (e.g. "2 days ago" for timestamps).

### Conditional flags

Use a helper to conditionally include flags when calling external commands:

```nu
def with-flag [flag: string]: any -> list {
    if ($in | is-empty) { [] } else { [$flag $in] }
}

# Usage: expands to [--namespace myns] or []
$namespace | with-flag --namespace
```

## Available modules

- **`ocr.nu`** — Submit, list, export, and manage OCR documents via Temporal workflows
