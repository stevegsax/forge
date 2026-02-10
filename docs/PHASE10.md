# Phase 10: Fuzzy Edit Matching

## Goal

Make search/replace edits resilient to minor LLM output discrepancies. Currently, `apply_edits` requires an exact match of the search string in the file content. If the LLM produces a search string with trivial differences — whitespace variation, indentation mismatch, or minor character differences — the edit fails and the task retries from scratch.

The deliverable: a fallback chain in `apply_edits` that tries progressively looser matching strategies when exact match fails, applying the edit to the best match with high confidence.

## Problem Statement

The D50 implementation of search/replace edits (`apply_edits` in `activities/output.py`) requires each search string to appear exactly once in the file content via `content.count(edit.search)`. This is correct for well-formed edits but fragile in practice:

1. **Whitespace discrepancies.** The LLM may produce trailing spaces, tab-vs-space differences, or inconsistent line endings that differ from the file on disk.
2. **Indentation mismatch.** When the LLM is editing a method inside a class, it may reproduce the code with the wrong indentation level (e.g., 4 spaces instead of 8).
3. **Minor content drift.** The LLM's search string may omit a trailing comma, use single quotes instead of double quotes, or have a minor character difference from the actual file content.

These failures are wasteful: the LLM produced the correct intent but a trivially wrong search string. Retrying burns tokens re-reading the same context and re-generating the same output. Aider's "fuzzy matching" and Claude Code's "similarity-based search" both address this by trying progressively looser matching when exact match fails.

## Prior Art

- **Aider**: Uses a multi-strategy approach for edit matching. When exact match fails, it tries: (1) stripping trailing whitespace from both sides, (2) normalizing indentation, (3) finding the most similar block using `difflib.SequenceMatcher` with a configurable similarity threshold (default 0.6). Reports that fuzzy matching recovers ~40% of otherwise-failed edits.
- **Claude Code**: Uses a search/replace format with built-in tolerance. When exact match fails, it searches for the closest matching block using similarity scoring. Also supports "perfect prefix" matching where the search string matches the beginning of a line.
- **Mentat**: Uses tree-sitter-based diff application that is inherently tolerant of whitespace and formatting differences because it operates on AST nodes rather than text.

## Scope

**In scope:**

- Add a fallback chain to `apply_edits` with progressively looser matching: exact → whitespace-normalized → indentation-normalized → fuzzy (difflib).
- Each fallback level logs a warning so operators can track match quality.
- Configurable similarity threshold for fuzzy matching (default: 0.6).
- Fuzzy matching must still require uniqueness — if multiple blocks match above the threshold, the edit fails (ambiguous).

**Out of scope (deferred):**

- Tree-sitter-based structural matching (requires Phase 13 for multi-language tree-sitter support).
- LLM-assisted edit repair (asking the LLM to fix its own failed edit).
- Automatic indentation adjustment of the replacement string to match the matched block's indentation level. This is a natural follow-on but adds complexity; Phase 10 focuses on matching only.

## Architecture

### Fallback Chain

The matching fallback chain in `apply_edits` tries strategies in order of strictness:

```
1. Exact match (current behavior)
   content.count(edit.search) == 1

2. Whitespace-normalized match
   Strip trailing whitespace from each line of both search and content.
   Match on the normalized versions. Apply to the original.

3. Indentation-normalized match
   Dedent the search string fully (textwrap.dedent). For each possible
   indentation level in the file (0, 4, 8, 12, ...), re-indent the
   dedented search string and check for exact match.

4. Fuzzy match (difflib.SequenceMatcher)
   Slide a window of len(search_lines) over the file's lines.
   Score each window. If the best score exceeds the threshold and is
   unique (no other window within 0.05 of it), use that window.
```

Each level only activates if the previous level found zero matches. If any level finds exactly one match, the edit proceeds. If any level finds multiple matches, the edit fails as ambiguous.

### Pure Functions

All matching strategies are pure functions in `activities/output.py`:

```
def _exact_match(content: str, search: str) -> int | None:
    """Return the start index if search appears exactly once, else None."""

def _whitespace_normalized_match(content: str, search: str) -> tuple[int, int] | None:
    """Match after stripping trailing whitespace from each line.
    Returns (start, end) indices in the original content, or None."""

def _indentation_normalized_match(content: str, search: str) -> tuple[int, int] | None:
    """Match after normalizing indentation.
    Returns (start, end) indices in the original content, or None."""

def _fuzzy_match(
    content: str,
    search: str,
    threshold: float = 0.6,
) -> tuple[int, int, float] | None:
    """Find the best fuzzy match above the threshold.
    Returns (start, end, score) in the original content, or None."""
```

The main `apply_edits` function calls these in order:

```
def apply_edits(
    original: str,
    edits: list[SearchReplaceEdit],
    *,
    similarity_threshold: float = 0.6,
) -> str:
```

### Match Quality Tracking

Each edit application records which matching level succeeded. A new model tracks this:

```
class EditMatchResult(BaseModel):
    edit_index: int
    match_level: MatchLevel  # exact, whitespace, indentation, fuzzy
    similarity_score: float | None = None  # Only for fuzzy matches
```

`apply_edits` returns a richer result:

```
class EditApplicationResult(BaseModel):
    content: str
    match_results: list[EditMatchResult]
```

The existing `apply_edits` signature is preserved as the primary interface; a new `apply_edits_detailed` function returns the full result. `apply_edits` delegates to it and returns just the content string.

## Data Models

New models in `models.py`:

```
class MatchLevel(StrEnum):
    EXACT = "exact"
    WHITESPACE = "whitespace"
    INDENTATION = "indentation"
    FUZZY = "fuzzy"
```

## Project Structure

Modified files:

```
src/forge/
├── models.py                   # Modified: add MatchLevel
└── activities/
    └── output.py               # Modified: fallback chain in apply_edits
```

## Dependencies

No new dependencies. Uses `difflib.SequenceMatcher` (stdlib) for fuzzy matching and `textwrap.dedent` (stdlib) for indentation normalization.

## Key Design Decisions

### D55: Fallback Chain Over Single Strategy

**Decision:** Use an ordered fallback chain (exact → whitespace → indentation → fuzzy) rather than jumping directly to fuzzy matching for all edits.

**Rationale:** Exact matching should remain the fast path — it's O(n) and unambiguous. Fuzzy matching is O(n*m) and introduces a confidence threshold. The fallback chain preserves the performance and correctness guarantees of exact matching for the common case while recovering gracefully when minor discrepancies occur. Each level is more expensive and less certain than the previous, so trying them in order minimizes cost and maximizes confidence. Logging which level matched enables monitoring match quality degradation over time.

### D56: Similarity Threshold at 0.6

**Decision:** Default fuzzy matching threshold is 0.6 (configurable).

**Rationale:** Aider uses 0.6 as its default and reports good results — it catches most whitespace and minor content differences while avoiding false matches. A threshold below 0.5 risks matching unrelated code blocks. The threshold is configurable per-task if needed, but the default should be conservative enough to avoid incorrect matches.

### D57: Uniqueness Required at All Levels

**Decision:** Even fuzzy matching requires a unique best match. If two blocks score within 0.05 of each other above the threshold, the edit fails as ambiguous.

**Rationale:** A non-unique match means the system cannot confidently determine which code block the LLM intended to modify. Applying the edit to the wrong block would silently corrupt the file — worse than failing and retrying. The 0.05 gap requirement ensures the best match is clearly distinguishable from alternatives.

## Implementation Order

1. Add `MatchLevel` enum to `models.py`.
2. Implement `_exact_match`, `_whitespace_normalized_match`, `_indentation_normalized_match`, `_fuzzy_match` as pure functions in `output.py`.
3. Implement `apply_edits_detailed` in `output.py` with the fallback chain.
4. Update `apply_edits` to delegate to `apply_edits_detailed`.
5. Add logging for non-exact matches.
6. Tests for each matching strategy and the fallback chain.

## Edge Cases

- **All strategies fail:** `EditApplicationError` raised as before. The error message indicates that even fuzzy matching found no candidate above the threshold.
- **Fuzzy match tie:** Two blocks score above the threshold within 0.05 of each other. Error raised with both candidates shown in the message for debugging.
- **Empty file:** Exact match of empty search already errors. Fuzzy matching on an empty file finds no windows. Normal error path.
- **Very long search strings:** `SequenceMatcher` performance degrades for very long strings. In practice, search strings are typically 1-20 lines. No special handling needed.
- **Binary content / encoding issues:** `apply_edits` operates on strings. Files with encoding issues are caught at read time, before matching.
- **Replacement indentation:** Phase 10 does NOT adjust the replacement string's indentation to match the matched block. If the search matched at indentation level 8 but the search string was at level 4, the replacement is applied as-is. This is a known limitation; automatic indentation adjustment is deferred.

## Definition of Done

Phase 10 is complete when:

- `apply_edits` recovers from whitespace, indentation, and minor content differences via the fallback chain.
- Exact matching remains the fast path and is tried first.
- Fuzzy matching requires a unique best match above the threshold.
- Each edit application logs which matching level succeeded.
- The similarity threshold is configurable.
- All existing tests pass (backward compatible — exact matching behavior is unchanged).
- New tests cover: whitespace normalization, indentation normalization, fuzzy matching, ambiguous match rejection, threshold configuration, fallback chain ordering.
