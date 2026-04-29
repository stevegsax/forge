"""Editorial guidance for the pbook Claude Code skill.

`pbook skill-prompt --json` returns the contents of this module as a
structured payload that ``/skill-creator`` can ingest to populate
SKILL.md, and that the skill agent can re-fetch at runtime to refresh
its context. Each workflow's value is markdown-formatted.

When you add a new CLI command, update this module in the same change:
add a "command" entry in ``COMMANDS`` and (if relevant) a workflow
entry in ``WORKFLOWS``.
"""

from __future__ import annotations

from pbook.tags import EXTRACTED_NAMESPACES, GENERAL_NAMESPACES

_COMMANDS: dict[str, dict[str, str]] = {
    "list": {
        "description": "List entries with optional tag/type/project/needs-review filters.",
        "args": "--tag T --type T --project P --needs-review --include-rejected --limit N --json",
        "example": "pbook list --needs-review --json",
    },
    "get": {
        "description": "Fetch a single entry by id.",
        "args": "<id> --json",
        "example": "pbook get 151 --json",
    },
    "search": {
        "description": (
            "Free-text + tag search via RetrievalWorkflow. Requires the worker. "
            "Returns entries ranked by cosine similarity when QUERY is given."
        ),
        "args": (
            "<query> --tag T --threshold F --mode create|fix --limit N "
            "--include-rejected --include-unapproved --json"
        ),
        "example": "pbook search 'flaky pytest' --tag lang:python --json",
    },
    "sources": {
        "description": "List entry_sources rows for an entry — the situations that produced it.",
        "args": "<id> --json",
        "example": "pbook sources 151",
    },
    "session-text": {
        "description": (
            "Render a Claude Code session transcript by id (USER:/ASSISTANT: format). "
            "--raw emits the JSONL bytes; --path overrides path resolution."
        ),
        "args": "<session_id> --path P --raw --json",
        "example": "pbook session-text abc-def-…",
    },
    "tags": {
        "description": "Show tag namespaces (canonical) and values currently in use.",
        "args": "--json",
        "example": "pbook tags --json",
    },
    "review": {
        "description": "List entries flagged needs_review.",
        "args": "--limit N --json",
        "example": "pbook review --json",
    },
    "approve": {
        "description": "Clear the needs_review flag on an entry.",
        "args": "<id> --json",
        "example": "pbook approve 151 --json",
    },
    "reject": {
        "description": (
            "Soft-mark an entry as rejected with an optional reason. "
            "The row stays for audit; default queries hide it."
        ),
        "args": "<id> --reason 'why' --json",
        "example": "pbook reject 151 --reason 'too generic' --json",
    },
    "add": {
        "description": (
            "Add a new playbook entry. Reads JSON from stdin (or --file). "
            "--needs-review flags it for later approval. --schema prints the JSON schema."
        ),
        "args": "--file P --needs-review --schema --json",
        "example": (
            'echo \'{"title":"...","content":"...",'
            '"tags":["lang:python"]}\' | pbook add --json'
        ),
    },
    "feedback": {
        "description": "Record helpful/harmful feedback on a retrieved entry.",
        "args": "<id> --helpful | --harmful --context 'why'",
        "example": "pbook feedback 151 --helpful",
    },
}


_QUERY_WORKFLOW = """\
## Query workflow

When the user asks "find plays about X" or "what do we know about X?",
prefer free-text search over tag enumeration:

1. **Try semantic search first**:
   `pbook search "<their phrase>" --json`
   Output is ranked by cosine similarity. The skill should show the
   user the top few hits with their similarity scores. If similarity
   for the top hit is below ~0.6, treat it as a weak match — say so.

2. **If they mentioned a tag explicitly**, narrow with `--tag`:
   `pbook search "<phrase>" --tag lang:python --json`
   Tag and query are AND-merged.

3. **For "show me all <category>" queries**, prefer `pbook list`:
   `pbook list --tag lang:python --json`
   No similarity scores; just tag membership.

4. **The worker must be running** for `search`. If it's not, the
   command emits `{"error": ..., "code": "worker_unavailable"}`.
   Surface that clearly to the user — don't retry silently.
"""

_DISCUSS_WORKFLOW = """\
## Discuss workflow

When the user picks an entry and wants to talk about *why* it exists,
compose three commands:

1. **The entry itself**: `pbook get <id> --json`
2. **The originating sources**: `pbook sources <id> --json`
   Each row's `source_context` is the rich situation excerpt
   forge captured at extraction time. Show those first — they're
   usually enough.
3. **Only when needed: the original transcript**:
   `pbook session-text <session_id>` (resolves the JSONL via
   `~/.claude/projects/`). The transcript is large; quote
   specific spans rather than dumping the whole thing.

Reach for transcripts when:
- The user asks something the source_context doesn't answer
- Multiple sources agree on the play but the user wants to see
  one specific case
- The user is debugging the play itself (was extraction wrong?)
"""

_REVIEW_WORKFLOW = """\
## Review-queue workflow

When the user wants to triage `needs_review` entries:

1. **List the queue**: `pbook review --json`
2. **For each entry, examine in context**:
   - `pbook get <id> --json` — the entry itself
   - `pbook sources <id> --json` — what produced it
3. **Ask the user to decide** — don't auto-approve.
4. **Apply the decision**:
   - Approve: `pbook approve <id> --json`
   - Reject (with reason): `pbook reject <id> --reason "..." --json`
     Reject is *soft*; the row survives. `--reason` is optional but
     strongly encouraged — surfaced in `rejection_reason`.

Reject when the entry is misleading, over-prescriptive, generic,
or duplicated. Approve when it's specific, actionable, and hasn't
been captured already.
"""

_ADD_WORKFLOW = """\
## Add workflow

When the user wants to add a new play:

1. **Discover what tags are available**: `pbook tags --json`
   Use `namespaces` to validate, `values_in_use` to suggest familiar
   choices.
2. **Discover the JSON schema**: `pbook add --schema`
3. **Compose the JSON** with the user's help (title, content, tags).
   Keep content minimal and specific — the quality bar is "better
   to skip than to add a misleading or generic entry."
4. **Pipe to add**:
   `echo '<json>' | pbook add --json`
   Use `--needs-review` if you want the user to do a final approval
   pass via the review queue later (otherwise the entry is stored
   as approved).

The skill is the human-in-the-loop. Don't route through any LLM
review path — the skill (with the user) IS the review.
"""


_WORKFLOWS: dict[str, str] = {
    "query": _QUERY_WORKFLOW,
    "discuss": _DISCUSS_WORKFLOW,
    "review_queue": _REVIEW_WORKFLOW,
    "add": _ADD_WORKFLOW,
}


def _build_tag_section() -> dict:
    """Return a structured description of the tag system.

    Values-in-use is intentionally NOT included here — it's runtime
    state and lives behind ``pbook tags --json``. This section is
    static guidance the skill can bake into its prompt.
    """
    return {
        "namespaces": {
            "general": sorted(GENERAL_NAMESPACES),
            "extracted": sorted(EXTRACTED_NAMESPACES),
        },
        "notes": (
            "Tags are namespaced as `<namespace>:<value>`. The namespace set "
            "is closed (defined in pbook.tags); values are open. Use the "
            "`general` namespaces (lang/lib/domain) for cross-project knowledge "
            "and `extracted` namespaces (project/pattern) for project-specific "
            "or pattern-specific tags. Run `pbook tags --json` to see the "
            "values currently in use as suggestions."
        ),
    }


def build_skill_prompt(operation: str = "") -> dict:
    """Return the structured skill-prompt payload.

    If ``operation`` is non-empty, return only the matching workflow's
    markdown alongside its commands; otherwise return the full payload.
    """
    if operation:
        if operation not in _WORKFLOWS:
            msg = (
                f"Unknown operation {operation!r}. Available: "
                f"{sorted(_WORKFLOWS)}"
            )
            raise KeyError(msg)
        return {
            "workflow": _WORKFLOWS[operation],
            "commands": _COMMANDS,
            "tags": _build_tag_section(),
        }

    return {
        "commands": _COMMANDS,
        "workflows": _WORKFLOWS,
        "tags": _build_tag_section(),
    }
