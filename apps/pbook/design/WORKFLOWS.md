# Workflows and Activities

pbook uses Temporal for workflow orchestration. All workflows run on the `pbook-task-queue`.

## RetrievalWorkflow

Fetches, ranks, and packs playbook entries within a token budget.

```
Input:  RetrievalInput(tags, mode, token_budget, project, approved_only)
Output: RetrievalResult(entries, token_count, total_candidates)

Step 1: fetch_candidates          [30s timeout]
        → query entries matching any tag (OR semantics, limit 100)
        → apply approved_only filter if set

Step 2: rank_and_pack             [in workflow thread, no activity]
        → score each entry by tag overlap + mode weighting
        → sort descending by score
        → pack within token_budget (~4 chars/token estimate)
```

### Scoring algorithm

Base score = number of overlapping tags between entry and query.

Mode-based adjustments:

| Mode | General tag bonus | Extracted tag bonus | API_DOC bonus | PITFALL bonus |
|------|-------------------|---------------------|---------------|---------------|
| `create` | +0.5 per tag | — | +1.0 | — |
| `fix` | — | +0.5 per tag | — | +1.0 |

Entries with zero tag overlap are excluded (score = 0).

## ExtractionWorkflow

Receives pushed experience data, extracts lessons via LLM, saves with `needs_review=True`.

```
Input:  JSON {"experiences": [...], "project": "..."}
Output: {"entries_created": int}

Step 1: extract_from_experience   [5m timeout, 60s heartbeat]
        → build system prompt with quality bar
        → call LLM (ExtractionResult structured output)
        → return JSON-serialized result

Step 2: save_extracted_entries    [30s timeout]
        → parse extracted entries
        → set entry_type=PITFALL, needs_review=True
        → bulk insert into database
```

### Extraction quality bar

The system prompt instructs the LLM to extract **only** lessons that are both:

1. **Unexpected** — the default approach did not work; multiple retries were needed; an API behaved differently than documented; a standard pattern failed in a specific context
2. **Actionable** — there is specific, concrete advice that would help next time

Explicitly excluded: generic advice, standard rules, expected behavior, vague or over-prescriptive guidance, entries without specific actionable content.

The prompt concludes: "It is better to extract NOTHING than to extract a misleading or overly generic entry. Quality over quantity."

## ManualEntryWorkflow

Validates, reviews, and saves a manually submitted playbook entry.

```
Input:  raw_json (PlaybookEntry as JSON string)
Output: {"approved": bool, "entry": dict, ...}

Step 1: validate_entry            [30s timeout]
        → parse raw JSON against PlaybookEntry schema
        → if invalid: return {approved: false, validation_error}

Step 2: fetch_existing_entries    [30s timeout]
        → query 50 most recent entries for duplication context

Step 3: review_entry              [2m timeout, 60s heartbeat]
        → build review prompt with quality bar + existing entries
        → call LLM (ReviewResult structured output)
        → if rejected: return {approved: false, rejection_reason}
        → apply suggestions (merge titles, content, tags)

Step 4: save_extracted_entries    [30s timeout]
        → save reviewed entry to database
        → return {approved: true, entry, entries_saved}
```

### Review quality bar

The review LLM evaluates entries on four criteria:

1. **Accuracy** — technically correct
2. **Specificity** — actionable, not vague
3. **Minimality** — concise without unnecessary prescriptions
4. **Duplication** — not substantially covered by an existing entry

The prompt states: "It is BETTER TO REJECT than to accept an entry that is misleading, over-prescriptive, too generic, or vague."

The LLM may suggest improvements to title, content, or tags. Suggestions are applied automatically: non-empty suggested values replace originals; suggested tags are merged (union, preserving order).

## ExportWorkflow

Exports matching entries with parallel fan-out.

```
Input:  JSON {"tags": [...], "limit": int}
Output: {"entries": [...], "count": int}

Step 1: fetch_entry_ids           [30s timeout]
        → query matching entry IDs by tags

Step 2: fan-out                   [30s per entry]
        → start one export_single_entry activity per ID
        → each converts DB row to PlaybookEntry dict

Step 3: gather
        → await all handles in order
        → return collected entries
```

## Activity summary

| Activity | Module | LLM Call | Database |
|----------|--------|----------|----------|
| `fetch_candidates` | retrieval | No | Read |
| `extract_from_experience` | extraction | Yes (Anthropic Haiku) | No |
| `save_extracted_entries` | extraction | No | Write |
| `validate_entry` | review | No | No |
| `fetch_existing_entries` | review | No | Read |
| `review_entry` | review | Yes (Anthropic Haiku) | No |
| `fetch_entry_ids` | export | No | Read |
| `export_single_entry` | export | No | Read |
