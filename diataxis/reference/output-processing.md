# Output Processing Reference

Technical reference for LLM output models, edit matching algorithms, and file write
behavior. For design rationale and worked examples, see
[Output Processing](../explanation/output-processing.md). For guidance on using validation
to detect edit failures, see
[Validation and Retries Reference](validation-and-retries.md).

---

## Data Models

### LLMResponse

The top-level structured response produced by every LLM generation call.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `explanation` | `str` | Yes | Free-text description of what the LLM did and why. Used for observability only; not processed by the orchestrator. |
| `files` | `list[FileOutput]` | Yes | New files to create. May be empty. |
| `edits` | `list[FileEdit]` | Yes | Edits to existing files. May be empty. |

Constraint: a file path must not appear in both `files` and `edits`.

---

### FileOutput

A complete new file to be written to the worktree.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | `str` | Yes | File path relative to the worktree root. |
| `content` | `str` | Yes | Complete file content to write. |

The parent directory is created if it does not exist. If a file at `path` already exists,
it is overwritten.

---

### FileEdit

All edits to a single existing file, applied as a sequence.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | `str` | Yes | File path relative to the worktree root. Must exist. |
| `edits` | `list[EditOperation]` | Yes | Ordered list of search/replace operations. Applied sequentially. |

If the file at `path` does not exist, the activity fails immediately.

---

### EditOperation

A single search/replace operation within a file.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `search` | `str` | Yes | The string to find in the file. Must match exactly once at the applicable fallback level. |
| `replace` | `str` | Yes | The string to substitute in place of the matched content. |
| `description` | `str` | No | Optional human-readable note on what this edit does. Used in logs only. |

---

## Edit Matching Algorithm

The `apply_edits` function implements a four-level fallback chain. Each level is attempted
in order. If a level finds exactly one match, the edit is applied and no further levels are
tried. If a level finds multiple matches, the edit fails as ambiguous. If a level finds
zero matches, the next level is attempted.

### Level 1: Exact Match

```
content.count(edit.search) == 1
```

The search string is compared to the file content as-is, including all whitespace. This is
the most common path for well-formed edits.

Failure conditions:

- Count is zero: no match found; proceed to level 2.
- Count is greater than one: ambiguous; edit fails.

---

### Level 2: Whitespace-Normalized Match

Each line of both the search string and the file content has trailing whitespace stripped
(`rstrip()`) before comparison. The match is located in the normalized versions, then the
corresponding span is identified in the original file content. The replacement is applied
to the original content.

Failure conditions:

- Count in normalized content is zero: proceed to level 3.
- Count in normalized content is greater than one: ambiguous; edit fails.

---

### Level 3: Indentation-Normalized Match

The search string is fully dedented using `textwrap.dedent` to remove common leading
whitespace. For each indentation level found in the file (multiples of 4 spaces, from 0 up
to the maximum indentation level present), the dedented search string is re-indented at
that level and an exact match is attempted. If exactly one indentation level produces a
unique match, the edit proceeds.

Failure conditions:

- No indentation level produces any match: proceed to level 4.
- One indentation level produces a match, but that match appears more than once: ambiguous;
  edit fails.
- More than one indentation level each produce a match: ambiguous; edit fails.

---

### Level 4: Fuzzy Match (difflib)

A sliding window of `len(search_lines)` lines is passed over the file's lines. Each window
position is scored using `difflib.SequenceMatcher`. Two conditions must hold:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Similarity threshold | 0.60 | The best-scoring window must score above this value. |
| Uniqueness gap | 0.05 | The best-scoring window must exceed the second-best score by at least this margin. |

If both conditions hold, the edit is applied to the best-matching window. If either
condition fails, the edit fails.

Failure conditions:

- Best score is below 0.60: no adequate match; edit fails.
- Best score is above 0.60 but within 0.05 of the second-best score: ambiguous; edit
  fails.

---

## Edit Application Order and Error Conditions

| Behavior | Description |
|----------|-------------|
| Sequential application | Edits in a `FileEdit.edits` list are applied in index order. Each edit operates on the file content as modified by all preceding edits. |
| Atomic failure | If any edit in the list fails (no match or ambiguous match), the activity fails immediately. Preceding edits within the same `FileEdit` are not rolled back. The file is left in its partially-edited state, and the workflow retries the entire step. |
| Cross-file independence | Edits to different files are independent. A failure in one `FileEdit` does not affect whether other `FileEdit` objects have been applied. |

---

## File Write Behavior

| Condition | Behavior |
|-----------|----------|
| File in `files`, path does not exist | File is created; parent directories are created as needed. |
| File in `files`, path exists | File is overwritten with the new content. |
| File in `edits`, path does not exist | Activity fails immediately with a file-not-found error. |
| File in `edits`, path exists | Each `EditOperation` in the edit list is applied sequentially. |
| Path in both `files` and `edits` | Activity fails immediately with a constraint violation error. |

---

## Matching Fallback Level Log Messages

Each fallback level emits a warning log if it is reached, so operators can track match
quality across runs.

| Level | Log Message |
|-------|-------------|
| 2 | `"Whitespace-normalized match used for edit in {path}"` |
| 3 | `"Indentation-normalized match used for edit in {path}"` |
| 4 | `"Fuzzy match used for edit in {path} (score={score:.2f})"` |
| Any (ambiguous) | `"Ambiguous match for edit in {path} at level {level} — edit failed"` |

Level 1 (exact match) does not emit a warning.

---

## Related

- [Output Processing](../explanation/output-processing.md) — Design rationale and worked
  examples for the fallback chain.
- [Validation and Retries Reference](validation-and-retries.md) — What happens when an
  edit fails and the step retries.
- [How to Configure Validation](../howto/configure-validation.md) — CLI flags for
  controlling validation behavior.
