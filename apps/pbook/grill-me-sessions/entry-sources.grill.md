# Grill Session: entry-sources

Started: 2026-04-28
Last updated: 2026-04-28
Status: complete
Domain: database schema design + extraction pipeline (pbook + forge)

## Summary

Adds provenance tracking to playbook entries so the user can later
"discuss" a play and have the LLM refer back to the originating Claude
Code session(s). Centerpiece is a new `entry_sources` join table at
experience-level granularity (one row per situation, not per session),
with rich `source_context` text and an embedding for ad-hoc
debugging/review (NOT used in retrieval). forge produces situations
from the transcript; pbook persists them.

Scope expanded mid-session to also include match-or-attach behavior
in extraction: when the extraction LLM produces a candidate that
semantically matches an existing entry (≥0.85), no new entry is
created — instead, a new entry_sources row attaches to the existing
entry. A second source-level dedup threshold (0.92) prevents storing
duplicate justifications on the same entry. This makes the schema
deliver real value (no playbook pollution from cross-session
duplicates) and obsoletes the need for a separate `backfill-sources`
command — `pbook ingest --all --force` becomes the backfill path.

Discuss command design and routing all pbook LLM traffic through
forge are parked as separate efforts.

## Decision Log

### DECIDED: storage shape — join table over JSON column
- **Decision**: Use a new `entry_sources` join table rather than a
  JSON column on `entries`.
- **Rationale**: User preference. Lets us index/query "what entries came
  from session X" cheaply, and accommodates per-source context if we
  decide to put context on the join row.
- **Date**: 2026-04-28

### DECIDED: source_context location and embeddings
- **Decision**: `source_context` lives on `entry_sources` (one rich
  rationale per originating session). Add an `embedding` column on
  `entry_sources` for the situation text. Do NOT include situations
  or their embeddings in the automatic retrieval/LLM-query path.
- **Rationale**: Per-source preserves the structure of multiple
  situations producing the same play. Storing rich context is
  acceptable for now; can be trimmed later. The embedding is a
  debugging/review aid for the user — useful for ad-hoc analysis
  ("which plays came from situations like this?") without poisoning
  retrieval with situational specifics that would over-anchor the
  consuming LLM.
- **Date**: 2026-04-28

### REVISED: entry_sources granularity — experience-level, not session-level
- **Decision**: One row per (entry, experience), not per (entry, session).
  Surrogate `id INTEGER PRIMARY KEY AUTOINCREMENT`. Discriminator is
  `experience_hash = sha256(problem + resolution + context)`,
  nullable to allow future manual-attribution rows.
  UNIQUE `(entry_id, session_id, experience_hash)`. `ON DELETE CASCADE`
  on `entry_id`. No FK from `session_id` to `ingested_sessions`.
- **Rationale**: A single session can produce many distinct situations
  (potentially 100+); each that justifies a play is worth memorializing
  as its own origin. Session-level granularity would silently collapse
  multiple per-session origins onto one row. The hash buys idempotent
  re-ingest without requiring forge to mint UUIDs. Cascade preserved
  (no audit-log need). FK skipped (decouples from ingested_sessions).
- **Supersedes**: prior decision "session-level granularity, composite
  PK (entry_id, session_id)".
- **Date**: 2026-04-28

### DECIDED: extraction-stage quality bar gates entry_sources writes
- **Decision**: An `entry_sources` row exists only if the extraction
  LLM produced an entry for the experience. Situations forge wrote
  for experiences that didn't survive extraction are discarded.
- **Rationale**: The load-bearing quality filter is pbook's extraction
  prompt ("better to extract nothing than to extract a misleading
  entry"). Gating entry_sources on entry creation preserves that
  bar end-to-end. Two-pass extraction (forge writes situations only
  for survivors) was rejected as needlessly complex; the LLM tokens
  spent on discarded situations are marginal.
- **Date**: 2026-04-28

### DECIDED: source production — Option 1, forge writes the situation
- **Decision**: forge's `TranscriptIngestionWorkflow` analysis adds a
  `situation` field per experience — a rich rationale with quoted
  excerpts. Threaded through `PushExperienceInput.metadata` to pbook,
  which persists it as `entry_sources.source_context` unmodified.
- **Rationale**: forge has the transcript in hand; pbook does not.
  Quoted excerpts can only be selected at the analysis stage. Cheaper
  than a separate pbook LLM call (extends an existing call rather
  than adding one). Per-experience situation eliminates the
  same-session-collision question entirely.
- **Date**: 2026-04-28

### DECIDED: D — entry_sources columns
- **Decision**: `(id, entry_id, session_id, project_name,
  experience_hash, source_context, source_context_embedding,
  created_at)`. No transcript_path stored — derivable from
  project_name + session_id.
- **Date**: 2026-04-28

### DECIDED: E — manual entries get no entry_sources rows by default
- **Decision**: `pbook add` (curated entries) writes zero
  `entry_sources` rows. The nullable `experience_hash` leaves the
  door open for a future "attach manual source note" feature without
  schema change.
- **Date**: 2026-04-28

### DECIDED: G — consolidation re-parents entry_sources rows
- **Decision**: When `consolidate_entries_llm` merges N entries into
  one survivor, re-parent all `entry_sources` rows from the
  merged-away entries to the survivor before deleting them. Cascade
  handles any later prune of the survivor.
- **Date**: 2026-04-28

### REVISED: scope — extend to include match-or-attach in extraction
- **Decision**: Extending this round's scope beyond schema-only to
  include match-or-attach behavior in `save_extracted_entries`.
  Backfill of pre-existing entries handled by re-running
  `pbook ingest --all --force`, not by a separate
  `backfill-sources` command.
- **Rationale**: The schema only delivers value if extraction stops
  blindly inserting duplicates. Without match-or-attach, every new
  session creates new entries with one source each, even for lessons
  already in the playbook. The maintenance consolidation pass
  catches duplicates after the fact, but until it runs the playbook
  is polluted. Match-or-attach makes the schema *actually work* and
  also obsoletes the need for a backfill command — the existing
  `ingest --force` flow becomes the backfill mechanism, since
  re-ingestion now updates rather than duplicates.
- **Supersedes**: prior scope ("schema only this round") and Branch F
  recommendation (option 1 / empty backfill).
- **Date**: 2026-04-28

### DECIDED: H — match-or-attach behavior

- **Entry-match threshold**: 0.85 (matches existing
  `find_semantic_duplicates` default; consistent with manual-entry
  workflow). Tunable later.
- **Multiple matches**: attach to highest-scoring entry only. The
  maintenance consolidation pass must merge any near-duplicate
  entries (and re-parent their entry_sources rows; see Branch G).
- **No mutation of matched entry**: do not union tags, update
  title/content, or reset `needs_review`. Human approval remains
  mandatory.
- **Source-row dedup threshold**: 0.92 — before writing a new
  `entry_sources` row, compare its source_context embedding against
  existing source rows on the same entry. Skip insert if max
  similarity exceeds 0.92. Reasoning: don't store the same
  justification twice; only meaningfully different evidence
  earns a row.
- **UNIQUE conflict on `(entry_id, session_id, experience_hash)`**:
  ON CONFLICT DO NOTHING. Re-ingestion of identical experiences is
  a true no-op — no churn on stored situation text.
- **Match against entry_type**: both pitfall AND curated. A Claude
  Code session experiencing a curated lesson should attach as a
  source — empirical confirmation of the curated wisdom.
- **Source embedding location**: pbook side, alongside the existing
  `compute_embedding` activity. Follows the current (mixed)
  architectural pattern. Routing LLM/embedding traffic through
  forge is parked as a separate, broader refactor.
- **Date**: 2026-04-28

## Open Threads

(none — all branches resolved)

## Parking Lot

- **Discuss command design** — out of scope this session. Will need
  the schema and match-or-attach landed first; then design
  interaction model (one-shot question vs interactive, transcript
  loading, prompt design).
- **Route pbook LLM/embedding calls through forge** — current pattern
  is mixed: forge handles batch chat for analysis, pbook handles
  extraction chat directly via `pbook.llm` and embeddings via
  `pbook.embeddings`. Architecturally cleaner to have pbook do
  storage/orchestration only and route all LLM/embedding work
  through forge, but it's a meaningful refactor unrelated to this
  feature. Flagged for a future effort.
