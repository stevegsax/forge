# Forge Project Review — 2026-07-08

Independent review of the forge repository: architecture, code, tests, operations, the active 2026-06-10 migration plan, and a comparison against public projects. Six parallel review agents covered the Temporal workflow layer, the activities/execution layer, the LLM/context layers, CLI/persistence/operations, tests/docs/plan, and the external landscape; the highest-impact claims were re-verified first-hand. Read-only — no changes were made.

This review deliberately does not restate the 2026-06 architecture review's findings ([docs/reviews/2026-06-architecture-review.md](docs/reviews/2026-06-architecture-review.md)). Every spot-checked claim from that review verified accurately against the code — none was refuted. What follows is: the current state, what that review missed, what has decayed since, the process picture, and the external comparison.

## 1. Executive summary

1. **The June review was accurate; the code it described is unchanged.** All five criticals (C1–C5) are still live. The last commit in forge and every sibling repo is 2026-06-10 — the day the 47-task plan was approved. Zero tasks have been started, including hour-sized ones (T1.0).
2. **The plan's evidence base is gone.** The "durable" review artifacts (`merged-plan.md`, five wave JSONs, 164 verifier verdicts) that all 47 task files cite as design rationale no longer exist anywhere under `~/.claude`. Only the in-repo distillation survives. Verified first-hand.
3. **Two new criticals the June review missed**, both surviving the planned migration as specced: fan-out child workflows get 15–20-minute execution timeouts while their default-mode batch waits legally take up to 25 hours (§4.1); and the edit engine double-applies search/replace edits under ordinary Temporal activity retry, silently corrupting files (§4.2).
4. **The deletion plan has a booby trap.** T1.1 deletes the dead provider stack — which contains the only `max_retries=0` fix. The live sax-llm client stacks SDK retries (2) under Temporal retries (3): up to 9 provider attempts per failing call, with backoff hidden from Temporal. The task file doesn't know this.
5. **Operational posture undercuts the one safety fix the plan kept.** Workers run as root with zero systemd hardening (no `User=`, `NoNewPrivileges=`, `ProtectSystem=`), so C3's LLM-influenced subprocesses execute as root on the box holding all platform secrets. T1.7 scrubs the env but never drops privilege.
6. **The external comparison validates the architecture and challenges two specifics.** Owned control flow, Temporal-for-durability, worktree isolation, and repo-map context are all now industry practice or vendor-official. Batch-as-default-transport for *dependent multi-step* work is genuinely novel — no public art exists either way. The 0.6 fuzzy-edit threshold is below every published tool's floor (aider disabled fuzzy entirely; Roo Code defaults to exact), and the industry's peak-capability trend line runs opposite to the no-agentic-loop bet.
7. **The dominant risk is not architectural — it is that the repo reviews, plans, and formats instead of shipping.** Five months of history show two full review→plan cycles completing without the first cycle's findings being fixed; markdown outweighs source ~2:1; the final recorded acts before the 28-day silence are two markdown-lint sweeps. The plan's verified facts (model pins, API behavior, wheel matrix) decay while it waits.

## 2. Current state (verified 2026-07-08)

| Check | Result |
| --- | --- |
| `uv run pytest` (full default suite) | 1,358 passed, 4 deselected, 29s |
| `uv run mypy src` (strict) | Clean, 62 files |
| `uv run ruff check .` | **Fails — 3 errors** at HEAD with the locked ruff 0.15.0 |
| Migration tasks started | **0 of 47**; every task file's Development Notes empty; TASKS.md unchanged |
| Commits since plan approval (2026-06-10) | 0 in forge, sax-llm, pbook, forge-contracts |
| June criticals C1–C5 | All still live (C1 re-verified at `workflow_blocks.py:119-123`) |
| Review evidence artifacts | **Missing** — no `review-artifacts-2026-06-10/` or `merged-plan.md` under `~/.claude` |
| `ocr` sibling repo | **Not checked out** anywhere under `~/repos-sax` despite being one of the five repos T2.1 merges |
| Model pins | `claude-sonnet-4-5-20250929` ×17 in src; the plan's own target `claude-sonnet-4-6` is now also a generation behind |

Lint being red on a clean checkout is small but diagnostic: there is no CI (known, T2.2), so gates only hold when a human runs them, and this one has already rotted.

## 3. Process findings

This section matters more than any individual bug.

### 3.1 The execution pause is undocumented and contradicts the repo's own standards

Phase 1's framing is "stop the bleeding": two verified silent-data-loss criticals with ~20-line interim fixes (T1.2, T1.3). Four weeks later neither patch exists. Either the system is not being operated (then the urgency framing was rhetorical) or it is being operated with known data-loss bugs. No artifact records which — [PROCESS.md](development-plans/PROCESS.md) declares "accurate status documentation is as important as writing code," and the flagship work queue violates it. This is the second cycle of this pattern: the four February code reviews in `archive/to-merge/code-review/` were re-triaged in June with the concession that "none of the reviews' themes are cleanly closed" ([docs/OVERVIEW.md:52](docs/OVERVIEW.md)).

### 3.2 The plan's evidence is unrecoverable

Every task file points its design rationale at `~/.claude/plans/perform-a-thorough-adversarial-vectorized-barto.md` and the "durable copies" in `~/.claude/projects/.../review-artifacts-2026-06-10/`, which the handoff declares authoritative. Both are gone. The agreement matrix, the verifier verdicts' underlying evidence, and the full merged plan cannot be consulted; re-adjudicating anything (R1's history arithmetic, the 3.14 wheel matrix, the structured-outputs batch verification) means redoing it. A repo that version-controls 8.5k lines of *archived* markdown kept its load-bearing evidence outside version control. The 47 task files' dead pointers should be corrected to name the in-repo review doc as the surviving record.

### 3.3 The documentation apparatus is a structural tax

Counted: 206 markdown files, 32,495 lines, against 16,557 lines of source (tests: 25,223). Over half the markdown is self-declared non-authoritative (`diataxis/`, `archive/`) yet actively maintained — the repo's two most recent commits are markdown-lint sweeps across all of it. The plan itself exists in five synchronized representations (HANDOFF, HANDOFF-tasks, TASKS.md, 47 task files, D86–D97 + review doc), and they diverged within a day of creation (T8.2's task file claims a CLAUDE.md defect that commit `ea273e2` fixed the same day). Five of the 47 tasks exist solely to keep documents true. Meanwhile the status-of-record itself is wrong — [docs/OVERVIEW.md](docs/OVERVIEW.md) lines 26 and 31 still document the dead provider stack as a live capability, contradicting the repo's own review, and the correction is queued at T8.2, behind ~40 tasks. A two-line truth fix is held hostage to the last phase.

### 3.4 Verified facts are decaying while the plan waits

- ~11 call sites carry hardcoded `claude-sonnet-4-5-20250929` fallbacks that bypass the tier registry (`activities/llm.py:27`, `batch_submit.py:33`, `planner.py:202,324`, `sanity_check.py:169,245`, `conflict_resolution.py:230,312`, `exploration.py:39`, `eval/judge.py:26`, `cli.py:1617`). The pin is legacy-active today; pbook's equivalent already got retired once. A registry bump under T3.2 leaves ten stale fallbacks behind.
- The model bump is coupled to the thinking-shape migration: sax-llm emits `budget_tokens` for every model; that form is rejected with a 400 on Opus 4.7+/Sonnet 5. The REASONING pin cannot be modernized before T3.2's adaptive-thinking work, and no task file states this ordering constraint.
- Phase 3 rests on one-time (2026-06-10) verification of `output_config.format` batch behavior and `messages.parse`; Phase 2 on a point-in-time cp314 wheel matrix. All unrechecked, and the evidence behind them is gone (§3.2).

### 3.5 Weak arguments in the June review worth flagging

- The §8 rejection of `schema_version` on the S3 envelope leans on "blob life bounded by … bucket TTL" — **no lifecycle rule exists in the Terraform** (`deploy/terraform/s3.tf` enables versioning, no expiry). The rationale cites infrastructure that was never built.
- The refuted plaintext-prompt finding analyzed the store's trust boundary but not log files: `cli.py:489-490` sets the *root* logger to DEBUG when file logging attaches, so any SDK that logs payloads at DEBUG leaks prompts into `worker.log` outside the analyzed boundary.
- The sandbox rejection reasons from macOS (`sandbox-exec` deprecated) — but production is EC2 Linux, where systemd sandboxing and an egress allowlist are cheap and implementable (§4.5). The rejection generalized a dev-machine constraint to the deployment target.
- D1's founding rationale ("batch mode enables true parallelism") is falsified by D79 (one request per batch) and carries no supersession note, unlike the otherwise-disciplined amendment banners.

## 4. New technical findings

Ordered by severity. "Uncovered" means no task in [TASKS.md](development-plans/TASKS.md) addresses it; task IDs indicate where an amendment belongs.

### 4.1 CRITICAL — Fan-out is structurally incompatible with batch mode (both defaults)

`workflows.py:137-143` gives child workflows a 15-minute base execution timeout (+5 min/level), applied at `:1201` and `:1692`. `sync_mode` defaults to `False` (`models.py:788`, `:871`), so each child's generation call enters `batch_submit_and_wait` with a 25-hour wait (`workflow_blocks.py:59`) inside a 15–20-minute timeout. Any batch slower than ~15 minutes (the provider guarantees only ≤24h) kills the child, orphans the paid batch, and the poller's later signal to the dead child trips C2's FAILED path. The June review's history arithmetic never examined the child-timeout tree, and **T4.1's timer-loop keeps the wait inside the child, so the migration preserves the bug.** Uncovered — amend T4.1/T5.3.

### 4.2 CRITICAL — Edit application is non-idempotent under activity retry

`activities/output.py:459-474` writes files one-by-one inside the activity; `_WRITE_RETRY` (`workflows.py:128-131`) retries on transient `OSError`. On retry, already-edited files are re-read and re-edited — reproduced: an insert-style search/replace applied twice duplicates the anchor line instead of no-op'ing. Ordinary disk pressure converts a recoverable transient into silent file corruption that then flows into validation and commit. T1.6a covers persist-write idempotency only. Uncovered.

### 4.3 MAJOR — Temporal failure-propagation seams (three findings, one family)

- **No gather exception isolation.** `workflows.py:1230-1233`, `:1720-1724` bare-await each child. A child that *raises* (timeout per §4.1, batch error, 25h wait expiry) propagates `ChildWorkflowError`: no `TaskResult`, no run record, no worktree cleanup, and the default ParentClosePolicy (TERMINATE) kills in-flight siblings, orphaning their paid batches. `ingestion_workflow.py:264-299` demonstrates the correct per-child try/except pattern in the same codebase. Uncovered (T1.6b covers only the batch-wait record).
- **Timeout arithmetic doesn't close.** The 48h workflow execution timeout (`cli.py:66`) admits, at defaults, up to 22 batch waits × 25h per single-step task (exploration rounds sit inside the attempt loop, `workflows.py:562-618`). One slow batch blows the ceiling with the same ungraceful ending. Uncovered.
- **Zero `try`/`finally` in `workflows.py`.** Every exception path leaks the worktree *and* its `forge/<task_id>` branch; `remove_worktree`'s branch delete is best-effort (`git.py:210-218`), and `create_worktree` always uses `-b` (create-new), so the next run of the same task ID fails permanently on "branch already exists". Compounding: `commit_changes` (`git.py:255-274`) raises non-retryable `CommitError("Nothing to commit")` when a retry follows a commit that actually landed. Uncovered.

### 4.4 MAJOR — LLM lane fragility under the shipped defaults

- **No `stop_reason` handling anywhere.** Refusals, `max_tokens` truncation, and prose answers all collapse into misleading `ValidationError`s. Sync lane: 3 full-price identical retries. Batch lane: the parse activity fails *deterministically* on the same stored bytes after a paid up-to-24h round-trip. (`sax_llm/client.py:125-132`; `activities/llm.py:66`; `planner.py:221`.)
- **Thinking forces `tool_choice: auto`.** `sax_llm/anthropic.py:143-148` downgrades forced tool use whenever thinking is on — and `ForgeTaskInput.thinking` defaults to `budget_tokens=10_000`, so every default planner/sanity/conflict call runs in the mode where the model may legally answer in prose, which the harness then discards and crashes on.
- **`max_tokens=4096` hard default, no knob**, while domain prompts demand complete file contents (`domains.py:18-45`): one ~500-line file exceeds the cap and triggers the worst path above. T3.5/T3.1 replace the mechanism; their acceptance criteria should inherit stop_reason/refusal/truncation handling and an output-token budget explicitly. Uncovered as written.
- **Retry stacking (the T1.1 booby trap).** Dead `forge/llm_client.py:258-264` sets `max_retries=0` with a correct rationale comment; live `sax_llm/client.py:197-204` uses the SDK default (2). Combined with `_LLM_RETRY` (3): up to 9 provider attempts, 429/529 backoff invisible to Temporal. T1.1 must port the fix or T3.1 must own it — neither says so.

### 4.5 MAJOR — Operations

- **Root workers, zero sandboxing.** Neither `deploy/systemd/forge-worker@.service` nor the pbook unit sets `User=`, `NoNewPrivileges=`, `ProtectSystem=`, `ProtectHome=`, or `PrivateTmp=`. Model-influenced subprocesses (C3) run as root beside all platform secrets. Same cheap-fix class as T1.7; fold it in.
- **Scheduled workflows can wedge permanently** (live until Phase 4): the poller and extraction activities set no retry policy (unlimited default attempts, no non-retryable classes), the schedules use overlap-SKIP with no execution timeout, and `batch_poll.py:205-207` converts any single job error into whole-activity failure — one bad batch turns the poller into a 24h retry loop while the schedule skips every run.
- Secrets bootstrap fails open (`fetch-secrets.sh` writes empty values on missing SSM parameters); Temporal runs on `auto-setup` (documented not-for-production) with schema setup re-running per restart; Terraform has local state, a floating `>= 5.0` provider floor, and a latest-AMI data source that replaces the instance on any post-release `apply`.

### 4.6 MAJOR — Context/eval layer

- **Planner repo map is nondeterministic.** `planner.py:266-268` seeds PageRank personalization with `list(graph.modules)[:5]` — a hash-ordered set slice that differs across worker processes. The intended uniform fallback already exists for an empty list (`graph.py:179-182`); one-line fix. Uncovered by T7.x.
- **Prompt caching is plausibly net-negative.** Cache breakpoints sit on a sub-1024-token tool schema (never cacheable) and on a monolithic system block whose volatile tail busts reuse across attempts; under D79 single-request batches nothing shares a prefix within the 5-minute TTL, so nearly every call pays the 1.25× write premium for entries nothing reads. Checkable from the store: `cache_read_input_tokens` should be ~0. The exploration loop additionally renders `## Round N of M` at the *top* of the system prompt (`exploration.py:75`), guaranteeing prefix divergence in the one loop that could benefit. Uncovered (T7.4 caps accumulation; nothing owns breakpoint layout).
- **The eval judge is blind.** `judge_plan` accepts `repo_context` but its only caller passes none (`eval/runner.py:133`), so the completeness and context-quality criteria are scored without seeing the repo — noise dressed as measurement. Comparison math is a single judge sample against a ±0.5 band over a 3-case corpus. Harmless until someone treats an `EvalComparison` as a gate; T6.2 calibrates pbook's judge, not this one.
- **Plan preflight gate still has no task file.** Duplicate step IDs, overlapping fan-out targets, and cyclic references are constructible (`models.py:244-249`); the checks exist in `eval/deterministic.py` but run only in eval, and the review's disposition row itself notes "no dedicated task file." Exactly the kind of item that falls through a 47-item checklist — pin it to T5.1/T5.2.

### 4.7 MAJOR — Edit-engine correctness beyond idempotency

Sequential-edit invalidation feeds the fuzzy matcher: when edit 1 consumes text edit 2 searches for, levels 1–3 miss and difflib matches an unrelated adjacent window — reproduced at score 0.625, silently overwriting the wrong text. The D57 uniqueness margin only fires when two windows compete; a single weak match applies unchallenged. A two-line search where one line is entirely wrong already clears 0.6. See §6 for how far outside industry practice this threshold sits. T7.6's `allow_fuzzy` knob doesn't reduce the risk when fuzzy is on.

### 4.8 Selected minors

Full agent transcripts contain ~30 more; these change behavior or mislead operators.

| Finding | Location | Note |
| --- | --- | --- |
| False "adaptive thinking" claims; `ThinkingConfig.effort` dead | `llm_client.py:87-91`, `cli.py:598`, `models.py:115-118` | No code path builds adaptive; operators are misinformed |
| `--thinking-budget` silent no-op in default mode | `cli.py:594-600` | No planner/sanity/conflict in single-step mode |
| Exploration LLM calls never persisted | `workflows.py:380-400` | Up to 20 calls/task missing from the interactions table the §8 retention rejection relies on |
| `_submit_and_wait` helper defaults `sync_mode=True`, inverting batch-first | `cli.py:365` | Latent trap for future callers |
| ReDoS via LLM-supplied regex, on the event loop | `providers.py:57-99` | No timeout; blocks the worker; outside T1.4's subprocess fix |
| Prompt injection via unescaped code fences; playbooks are a self-poisoning channel | `context.py:242-247`, `:505-527` | LLM-extracted playbooks re-enter prompts unescaped |
| CRLF files get mixed line endings after fallback edits | `output.py:163-164`, `:429` | Reproduced |
| `validate.py` doesn't catch `TimeoutExpired`; 60s test cap | `validate.py:35-52` | Hard workflow failure instead of failed validation |
| `heartbeat_timeout=60s` on an activity that never heartbeats | `manual_playbook_workflow.py:73` | Any review >60s fails deterministically, retries unbounded |
| Playbook hot-path does full-table scans, no SQL LIMIT | `store.py:357-422` | Grows with exactly the pollution the review documented |
| `update_batch_status` is guardless last-writer-wins | `store.py:522-547` | Can regress the audit ledger Phase 4 keeps |
| Alembic: ORM/baseline `server_default` drift; no `render_as_batch`; boot-time migration race with two workers | `alembic/`, `worker.py:109-121` | First ALTER migration breaks SQLite; fresh deploys crash once |
| Coverage denominator includes 1,400 dead LOC; 2,515 LOC of tests cover sax-llm, not forge | `pyproject.toml:79` | 85% is arithmetically honest, semantically misleading |
| Zero replay, property-based, crash-recovery, or wire-contract tests | tests/ | Replay tests arrive only in T5.5, *after* the transport rewrite |
| `boto3` declared/unused; `mistralai` used/undeclared | `pyproject.toml:20` | Artifacts of the dead stack |
| `TOC.md` says "D1–D85"; DECISIONS is at D97 | `TOC.md:15` | Stale same-day |
| `scripts/basenames.py` queries a table that no longer exists | `scripts/` | Broken since the OCR split |

## 5. Migration plan assessment

The plan's content quality is high: single-PR task sizing, verified acceptance criteria, honest rejected-ideas ledger, deletion-first bias, and correctly ordered safeguards on the dangerous steps (T6.2 before T6.3; report-only sweep first). The critique is structural:

1. **It is a ~35-task serial march with no mid-course stable state.** After Phase 1, value lands only at whole-phase boundaries; between T2.1 and the end of Phase 5 there is no designed "stop here and the system is coherent" checkpoint. At observed calendar velocity (zero tasks in four weeks), that structure converts any pause into indefinite half-migrated limbo. Realistic effort: 3–4.5 months full-time; 6–12+ months at observed velocity, during which §3.4's facts rot further.
2. **T2.1 bundles three independent risks** — five-repo git-filter-repo history rewrite, uv-workspace conversion, and the Python 3.12→3.14 jump — at position ~11 of 47, with thin acceptance criteria, no rehearsal, no rollback plan, and an unanswered question: TASKS.md and the whole PROCESS.md apparatus live in forge, which T2.1 archives. Where does the work queue live for T2.2–T8.4? Split the Python bump out; rehearse filter-repo into a throwaway target; answer the tracking question first.
3. **The new criticals survive it.** T4.1's timer-loop keeps batch waits inside child workflows (§4.1); nothing owns gather exception isolation or timeout derivation (§4.3); T1.1 deletes the retry fix (§4.4); T5.x could faithfully consolidate the code around all of these defects without fixing any of them. The plan needs an amendment pass before execution, not after.
4. **T6.3 lacks operational safeguards** — one mega-migration including a halfvec conversion and HNSW rebuild against hosted Supabase with no transaction/downtime strategy (`CREATE INDEX CONCURRENTLY` can't run inside Alembic's transaction), and the pre-migration JSON dump has no tested restore path.
5. **The interim tasks are justified only if the system is operated.** T1.2/T1.3 are correct 20-line calls *for a live system*. Four idle weeks argue the opposite. Decide explicitly: if forge is running unattended anywhere, land them this week; if not, drop them and reorder to reach Phase 4 sooner.

## 6. External comparison

Full sourced survey in the review transcript; condensed here. Primary sources: Agentless (arXiv 2407.01489), aider's repo-map and edit-format publications, mini-SWE-agent, OpenHands, Anthropic's "Building Effective Agents" and SWE-bench scaffold posts, Temporal's OpenAI Agents SDK integration, 12-Factor Agents, Cognition's "Don't Build Multi-Agents," AWS Step Functions batch-orchestration patterns, and the 2026 scaffold-taxonomy paper (arXiv 2604.03515).

**Validated by public art:**

- *Owned control flow.* Anthropic's own engineering guidance ("workflows for predictability; find the simplest solution"), 12-Factor Agents (Factor 8 "own your control flow"), and Cognition's single-writer doctrine all converge on Forge's framing. Prefect's ControlFlow — the closest philosophical cousin — was archived, which cuts both ways: the ideas won, the standalone product didn't.
- *Temporal for LLM durability* is now vendor-official (OpenAI Agents SDK integration, 2025). Notably, all published Temporal AI patterns wrap *agentic* loops in durability; nobody publishes Forge's inversion. The beaten path is a year old.
- *Worktree-per-task isolation* is now first-party in Claude Code (`--worktree`) and Cursor 2.x (up to 8 parallel worktree agents).
- *Repo-map context packing* is aider's design; note aider ranks a **symbol-reference** graph, not an import graph — more signal per edge — and personalizes by conversation state. Forge's import-graph variant is coarser, and its single-language AST approach sits against a tree-sitter-standard ecosystem (consistent with deferred Phase 13).
- *Batch + Temporal mechanics*: the community-standard pattern (Kinney 2025; AWS Step Functions Bedrock-batch orchestration) is submit → durable poll → fetch — exactly D88/Phase 4's timer-loop. R1 is independently validated.

**Genuinely novel, therefore unproven:** batch APIs as the *default transport for dependent multi-step workflows*. Public batch art is embarrassingly parallel (evals, embeddings, moderation). Nobody publishes wall-clock data for chaining N sequential batch round-trips; §4.1/§4.3 are what that gap looks like from the inside. Forge is generating first-of-kind evidence here — worth writing up internally either way.

**Challenged by public evidence:**

- *The 0.6 fuzzy threshold is an outlier.* Aider ships a fallback chain but its difflib fuzzy stage is deliberately unreachable — it fails loudly and feeds the error back for a retry. Roo Code's fuzzy floor is 0.8 with a 1.0 default and requires a line-number anchor. Codex/Cursor never score below whitespace-trimming. Forge at 0.6 with no anchor is more permissive than every published tool; §4.7 shows the failure mode empirically. Aider's data does validate having a *structural* fallback chain (disabling flexible application: 9× more edit errors) — the issue is specifically the similarity-scored last level.
- *The capability trend line.* mini-SWE-agent — ~100 lines, bash-only, no scaffold intelligence — scores >74% on SWE-bench Verified; pipeline architectures haven't held SOTA since early 2025, and Agentless-style stages now serve as training priors and components inside agents (Kimi-Dev) rather than winning outright. Forge's bet trades peak per-task capability for cost, durability, and auditability — a defensible trade that should be stated as one, not assumed costless. The batch discount is the one advantage agentic loops structurally cannot access.
- *LLM merge conflict resolution is research-grade* (MergeGen, Rover, the AgenticFlict corpus — no production tool with reliability numbers). Cursor's best-of-N selection and Cognition's single-writer principle both suggest *avoiding* concurrent writes beats *resolving* them. Worth stress-testing whether fan-out needs LLM merge at all, or whether the planner should enforce disjoint file ownership (which `eval/deterministic.py` can already check — it just never runs, §4.6).

## 7. Recommendations, in order

1. **Land one task this week.** T1.0 or T1.7 — both under a day. The pattern of zero execution is the project's dominant risk; nothing else on this list matters if it holds. Record the pause's reason in TASKS.md either way.
2. **Pull the truth fixes out of Phase 8.** OVERVIEW's dead-stack claim and TOC's D85 line are two-line edits; the status-of-record should not be wrong for the duration of the migration.
3. **Amend the plan before executing it** (one sitting): add §4.1/§4.3 to T4.1/T5.3 acceptance criteria (child timeout derivation, gather try/except, worktree `finally`); add the `max_retries=0` port to T1.1; add stop_reason/truncation handling and an output-token budget to T3.5/T3.1; create the plan-preflight-gate task the review's own disposition admits is missing; fix the dead evidence pointers in all 47 task files.
4. **Decide the operating question explicitly.** If forge runs unattended anywhere: T1.2/T1.3/T1.7 plus systemd `User=`/hardening are urgent this week. If not: drop the interims, note that in TASKS.md, and reorder to shorten the path to Phase 4.
5. **Fix the three one-liners that don't need the migration:** planner PageRank seed `[]` (`planner.py:267`), fuzzy threshold to ≥0.8 or require-anchor (`output.py`), exploration round counter moved to the user prompt (`exploration.py:75`).
6. **De-risk T2.1**: split the 3.14 bump from the merge, rehearse filter-repo, and decide where the work queue lives post-archive.
7. **Measure before Phase 3**: pull `cache_read_input_tokens` from the store to confirm or refute the net-negative caching hypothesis (§4.6); re-verify structured-outputs batch behavior and the wheel matrix, since the original evidence is gone.
8. **Add per-file edit application or content-hash guards** to `write_output` (§4.2) — this is corruption-class and independent of everything else.

## 8. What holds up

For calibration, the things this review tried and failed to break: workflow-code determinism is clean (no time/random/env/I-O, sorted iteration, `workflow.uuid4()`); the path-safety boundary rejects traversal, absolute paths, and symlink escapes including for new files; the survivable-persist mechanism is genuinely idempotent for its covered writes; the mTLS design fails closed; no secrets are committed; DECISIONS.md's supersession discipline is unusually good; and the June review's factual claims survived every spot-check. The architecture's core bet is coherent and now partially vendor-validated — the gap between this system's design quality and its execution cadence is the finding.
