+++
title = "Output Processing"
weight = 71
description = "How the LLM's structured response is processed: the LLMResponse schema, edit application with fuzzy matching, and file writing."
topic = "output-processing"
covers = [
    "Why Forge uses tool-use for structured output instead of parsing free-form text",
    "The LLMResponse schema: explanation, files, and edits",
    "Why edits use search/replace instead of full-file replacement (the D50 decision)",
    "The four-level edit matching fallback chain and why each level exists",
    "How edits are applied sequentially — each edit sees the result of the previous one",
    "How ambiguity (multiple matches) is handled as an error",
]
detail = "Focus on the design decisions and tradeoffs. Why search/replace? Why four fallback levels? What goes wrong when the LLM outputs slightly different whitespace? Use concrete examples showing how a search string matches (or fails to match) at each fallback level."
+++
This document explains how Forge processes the LLM's response: why structured output is
enforced through tool use, why edits use a search/replace model rather than full-file
replacement, and how the four-level matching fallback chain tolerates minor discrepancies
in LLM output.

For the technical specification of each model field and the exact matching thresholds,
see [Output Processing Reference](../reference/output-processing/). For what happens
after edits are applied, see [Validation and Retries](validation-and-retries/).

---

## Why Structured Output Instead of Free-Form Text

Early LLM orchestrators parsed code out of markdown fenced blocks in free-form model
responses. This is fragile: the model might wrap the code differently, include commentary
between blocks, or omit a block entirely without any signal of failure. The parsing logic
becomes an ongoing maintenance problem, and parsing errors are hard to distinguish from
model errors.

Forge avoids this by forcing the response into a known shape using Anthropic's tool-use
feature. The LLM is instructed that it must call a specific tool — `llm_response` — and
the input schema for that tool is the `LLMResponse` Pydantic model. The API enforces this:
if the model's output cannot be serialized into the tool-call format, the API returns an
error rather than a partial or malformed response. Pydantic then validates the structured
JSON against the model's field types and constraints.

The consequence is that every successful LLM call produces a well-formed, validated
`LLMResponse`. The orchestrator never needs to handle partially parsed or structurally
ambiguous output. If the API call succeeds, the response is usable.

This also has a secondary benefit: the structure defines a contract between the LLM and
the orchestrator that is stable across model versions. When Anthropic releases a new
Claude model, the tool-use schema remains the same. The orchestrator does not need to
re-validate that its text parsing still works.

---

## The LLMResponse Schema

`LLMResponse` splits the model's output into three distinct concerns, each handled
differently by the orchestrator.

The first concern is reasoning: the model's free-text account of what it did and why.
This is captured separately so it can surface in observability output without ever
influencing downstream logic. Mixing reasoning and code output in a single string would
make it impossible to route them independently.

The second and third concerns are the two kinds of write operations: creating new files
and editing existing ones. Keeping them separate enforces an important invariant — a path
may appear in one list or the other, never both. This prevents a class of ambiguity where
the orchestrator would have to decide whether a "new file" operation on a path that was
also edited means replacement or conflict. The `write_output` activity enforces this
constraint at application time.

For the field-level specification, see [Output Processing Reference](../reference/output-processing/).

---

## Why Search/Replace Instead of Full-File Replacement

The most obvious way to let an LLM modify a file is to have it output the entire new
content. This is the approach used in some simple orchestrators: include the file in the
prompt, ask the LLM to produce the updated version, write the result.

The problem is one of scale and scope. The D50 decision identified two failure modes:

**Token waste.** When the LLM outputs a full file, it must reproduce every line that did
not change. For a 500-line file with a 10-line change, 490 tokens of output are consumed
reproducing unchanged content. At scale, this multiplies across many files and many steps.

**Silent destruction.** When the full file content is not present in the context — because
the token budget did not allow it, or because it was omitted from context assembly — the
LLM generates a new version of the file from partial knowledge. It may correctly produce
the changed section while silently omitting or misremembering other sections. The resulting
file passes structural validation but loses code that was not in the LLM's context. This
failure mode is particularly dangerous because it produces a plausible-looking result.

Search/replace edits avoid both problems. The LLM specifies an exact string to find and a
replacement string. Only the changed content appears in the output. The orchestrator
applies the replacement to the actual current file content, preserving everything that was
not explicitly modified. The LLM does not need the full file in its context to produce a
correct edit — it only needs the section it is changing.

The tradeoff is that the search string must be precise enough to identify the correct
location in the file. This is where the fallback chain becomes necessary.

---

## The Four-Level Fallback Chain

Requiring an exact match on the search string is correct in principle — a search string
that could match multiple locations is ambiguous, and applying the edit to the wrong match
would corrupt the file — but it is fragile in practice. LLMs produce minor discrepancies
with some frequency: trailing whitespace on lines, indentation at the wrong level, a
missing comma, or a single-vs-double quote difference. These are not errors in the LLM's
understanding of what to change; they are artifacts of how the model reconstructed the
search string from its memory of the file.

Forge uses a four-level fallback chain that tries progressively looser matching strategies
until one succeeds or all are exhausted. The levels are tried in order; each activates only
when the previous level found zero matches.

### Level 1: Exact Match

The search string is looked up in the file content directly. If it appears exactly once,
the edit proceeds. If it appears more than once, the edit fails as ambiguous — multiple
identical blocks exist and there is no way to determine which one the LLM intended.

### Level 2: Whitespace-Normalized Match

Trailing whitespace is stripped from each line of both the search string and the file
content before comparison. The match is then performed on the normalized versions. If a
match is found, the edit is applied to the corresponding span in the original (un-normalized)
file content, preserving the file's own whitespace.

### Level 3: Indentation-Normalized Match

The search string is fully dedented to remove its common leading whitespace. It is then
re-indented at each indentation level found in the file and an exact match is attempted at
each level. If exactly one level produces a match, the edit proceeds. This handles the
common case where the LLM reproduces a method body at 4-space indentation when the actual
content is at 8-space indentation (inside a class).

### Level 4: Fuzzy Match

`difflib.SequenceMatcher` compares a sliding window of lines from the file against the
search string. Two conditions must hold for the match to proceed: the best-scoring window
must exceed a minimum similarity threshold, and it must be clearly ahead of the
second-best window. The second condition — the uniqueness gap — is what prevents a fuzzy
match from applying an edit to the wrong location when two similar blocks exist. If two
windows score nearly equally, the match is ambiguous and the edit fails rather than guess.
For the exact threshold values, see [Output Processing Reference](../reference/output-processing/).

### Walking Through the Chain

Consider an edit targeting a two-line block in a Python file. The LLM reproduced the
search string with trailing spaces on one line — a common artifact of how models
reconstruct content from context. Level 1 finds no exact match because the trailing
whitespace differs from the file on disk. Level 2 strips trailing whitespace from both
sides and tries again; the normalized versions match, the edit is applied to the original
span in the file (preserving the file's actual whitespace), and the chain stops there.

If instead the LLM had reproduced the block at the wrong indentation level — say, at
4-space indent when the file has it at 8 inside a class — level 2 would also fail.
Level 3 would then dedent both sides to a common baseline and try re-indenting the search
string at each depth present in the file. When the 8-space attempt matches, the edit
proceeds. Levels 1 through 3 are all deterministic and cheap; the system reaches level 4
only for genuine content drift, not formatting accidents.

---

## Sequential Application and Ambiguity as Error

Edits within a `FileEdit` are applied one at a time, in the order they appear in the
response. Each edit sees the file content as modified by all preceding edits. This matters
when a task makes multiple changes to the same file: a later edit's search string may only
be valid after an earlier edit has been applied, or a search string might match in the
original file but not after an earlier deletion.

Ambiguity at any fallback level — meaning more than one location matches the search string
at that level — causes the edit to fail immediately without attempting further fallback
levels. The rationale is that a fuzzy match against multiple candidates provides no basis
for choosing; applying the edit to an arbitrary candidate would risk data loss or corruption.
When an edit fails, the entire `write_output` activity fails, which triggers the retry
path with the edit failure reported as an error (see
[Validation and Retries](validation-and-retries/)).

---

## Connection to Context Assembly and Validation

The quality of search/replace edits depends on the quality of the context the LLM received.
When a target file's current content is present in the prompt, the LLM can reproduce the
search string with high fidelity. When it is absent — because the token budget was exhausted
or the file was not in context — the LLM must reconstruct the search string from memory,
which increases the likelihood of drift and the need for deeper fallback levels. Context
assembly ensures that target file contents are the highest-priority items in the token
budget. See [Context Assembly](context-assembly/) for the priority ordering.

After edits are applied and new files are written, the validation pipeline runs
deterministic checks (lint, format, tests) on the resulting worktree state. If validation
fails, the error output is fed back to the LLM on the next retry. See
[Validation and Retries](validation-and-retries/) for how the retry loop works.
