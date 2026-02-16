# Unified Batch Completion Best Practices (Merged Report)

This document consolidates the five source reports in `docs/to-merge/completion-documents` into a single, structured guide. It preserves shared recommendations, integrates unique details from each report, and explicitly calls out contradictory guidance.

## Table of Contents

1. [Scope and Shared Conclusions](#1-scope-and-shared-conclusions)
2. [Canonical Completion-Document Structure](#2-canonical-completion-document-structure)
3. [Model-Family Adaptations](#3-model-family-adaptations)
4. [Token-Budget Triage and Compression](#4-token-budget-triage-and-compression)
5. [Single-Call vs Multi-Call Orchestration](#5-single-call-vs-multi-call-orchestration)
6. [Prompting Patterns That Improve Output Quality](#6-prompting-patterns-that-improve-output-quality)
7. [Advanced Pipelines for Maximum Quality](#7-advanced-pipelines-for-maximum-quality)
8. [Reusable Template](#8-reusable-template)
9. [Contradictions Across Source Reports](#9-contradictions-across-source-reports)
10. [Source Material](#10-source-material)

---

## 1. Scope and Shared Conclusions

Across all reports, the main consensus is:

- Use a stable scaffold with explicit objective, constraints, output contract/schema, and quality checks.
- Front-load high-priority instructions and keep critical constraints explicit.
- Prefer plan-first generation (outline/key points) before full drafting.
- For high-stakes outputs, multi-pass workflows (plan -> draft -> critique -> revise -> validate) are generally better than one-shot generation.
- Under token pressure, cut stylistic and redundant context first; preserve objective/constraints/schema/rubric.
- Keep one cross-model macro structure, then tune prompt packaging per provider/model family.

Unique contribution included from the most detailed report:

- Long-context behavior should account for primacy/recency effects ("Lost in the Middle"), and structure should exploit beginning/end salience.
- Anthropic-specific cache breakpoint planning can reduce input costs substantially in iterative batch systems.

---

## 2. Canonical Completion-Document Structure

A merged canonical structure (compatible with all reports) is:

1. **Task Header**
   - Task ID/version, target model family, audience, depth, latency/cost tier.
2. **Objective / Success Definition**
   - One-sentence deliverable definition plus concrete success criteria.
3. **Hard Constraints / Guardrails**
   - Must-do and must-not-do rules, policy boundaries, banned claims, citation requirements, format limits.
4. **Context Pack (Facts First)**
   - Prioritized facts/snippets, assumptions, known unknowns, confidence/timestamp where available.
5. **Output Contract (Schema/Headings)**
   - Exact required headings/order or JSON schema, length limits, citation format, fallback behavior for missing data.
6. **Evaluation Rubric / Acceptance Tests**
   - Accuracy, completeness, structure compliance, evidence quality, actionability/risk handling.
7. **Generation Procedure**
   - Plan first, then draft by section, then self-check/revision.
8. **Verification Checklist**
   - Section presence, constraint compliance, unsupported-claim check, contradiction check, schema conformance.
9. **Optional Examples**
   - Keep concise and high-signal (0-2 examples unless model/task explicitly benefits from more).
10. **Critical Reminder Near Output Point**
   - Repeat non-negotiables near the end for long contexts.

### Placement guidance synthesized from all reports

- Default order above is robust for maintainability and cross-model usage.
- For long prompts, exploit both **primacy** (critical instructions early) and **recency** (short reminder near generation point).
- Put large reference/context blocks in the middle where possible.

---

## 3. Model-Family Adaptations

All sources agree the macro structure should mostly stay the same; adapt syntax/strictness per model family.

### Claude (Anthropic)

- Strong fit with XML-style segmentation (`<objective>`, `<constraints>`, `<format>`, etc.).
- Keep explicit instruction hierarchy and separate required behavior from optional style.
- For batch/caching workflows, place stable prompt prefix content first and volatile retry/error content last.
- Some reports recommend few-shot examples (3-5 for complex tasks), while others recommend minimal examples; see contradiction notes.

### OpenAI GPT models

- Concise task framing + strict output contract works well.
- Schema-driven outputs (JSON mode/function/tool schema) preferred when deterministic parsing is required.
- For long contexts, repeated constraints at beginning and end are recommended.
- One report notes few-shot can underperform zero-shot in some GPT-4.1 contexts; others permit minimal examples.

### Llama-family

- Use simpler, unambiguous wording and concrete templates.
- Prefer shorter chunks and iterative passes for long/high-fidelity output.
- Use model-correct chat/special-token templates where applicable.

### Granite (IBM)

- Emphasize enterprise compliance/governance constraints.
- Keep policy/provenance/validation fields explicit.
- Deterministic output formats and checklists are favored.

### DeepSeek

- Chat variants: standard structured prompting and schema constraints.
- Reasoning variants (per one source) may require minimal direct prompting and can degrade with explicit CoT/few-shot/system-heavy framing.
- In all cases, anchor long-form generation with section constraints + final checklist.

### Kimi (Moonshot)

- Strong long-context usage, but still benefit from explicit compact schema.
- Use concise scope boundaries, explicit formatting, and citation/source hierarchy instructions.

---

## 4. Token-Budget Triage and Compression

### Common cut order (merged)

Cut first:

1. Stylistic prose/flourishes and duplicated wording.
2. Redundant or non-essential background narrative.
3. Extra examples (retain at most one high-value example unless needed).
4. Nice-to-have secondary sections/extended alternatives.
5. Detailed rationale/historical context.

Preserve as long as possible:

- Objective/success condition.
- Hard constraints/policy boundaries.
- Output schema/required section order.
- Core factual inputs and acceptance rubric.

### Compression techniques found across reports

- Convert prose to bullet atoms.
- Replace narrative context with fact tables or key-value fields.
- Use representation downgrades (full content -> signatures -> names-only -> omitted).
- Keep a minimal core prompt block always present (objective, constraints, schema, tests).
- Shift optional/secondary context to retrieval or later refinement passes.

---

## 5. Single-Call vs Multi-Call Orchestration

### Single-call is generally enough when

- Task is short, narrow, low-risk, and well-scoped.
- Output is strongly templated and parsing-sensitive.
- Context is already sufficient.

### Multi-call is preferred when

- Output is long/complex/high-stakes.
- Cross-section consistency is important.
- You need explicit quality gates and cheaper targeted retries.

### Common multi-pass pattern (all reports aligned)

1. **Plan pass**: outline, claims inventory, assumptions, evidence gaps.
2. **Draft pass(es)**: section-by-section generation with local constraints.
3. **Synthesis/integration**: merge sections, normalize style/terminology.
4. **Critique/QA**: rubric-based defect finding.
5. **Repair/revision**: fix only diagnosed issues.
6. **Format lock/validation**: enforce schema and checklist compliance.

### Additional orchestration details preserved

- Parallelize independent section drafts; sequence dependent sections.
- Use running summaries (instead of full prior text) to maintain coherence in long docs.
- Limit iterative refinement cycles due to diminishing returns.
- Keep a decision log for later passes.

---

## 6. Prompting Patterns That Improve Output Quality

Consolidated high-signal patterns:

- **Contract-first prompting**: state required output format before broad context.
- **Short, imperative instructions** over narrative prose.
- **Explicit section contracts** (exact headings/order/count rules).
- **Plan-first prompting** (outline/key points/evidence map before full prose).
- **Uncertainty handling requirement** (`Supported/Inferred/Speculative`, confidence labels, unknowns).
- **Claim-to-evidence linkage** requirements for factual tasks.
- **Self-check checklist** embedded in prompt.
- **Targeted regeneration** of failing sections instead of full rewrites.
- **Tool/schema-constrained output** where possible for deterministic machine-readability.

---

## 7. Advanced Pipelines for Maximum Quality

When call budget is effectively unconstrained, merge of advanced recommendations:

1. Ingest/normalize source material and deduplicate facts.
2. Generate multiple candidate outlines (N-best planning) and select/merge best.
3. Draft sections in parallel with section-specific constraints.
4. Run evidence pass to verify support for each claim.
5. Run consistency pass for terminology, numbers, definitions, tone.
6. Run adversarial critique pass to surface missing counterpoints/overclaims.
7. Repair only diagnosed defects.
8. Enforce deterministic final format/schema.
9. Optional readability polish pass.
10. Optional multi-candidate full-document judging/synthesis for highest-stakes outputs.

Implementation notes gathered from reports:

- Planning quality often bounds downstream quality.
- Keep context fresh per pass; avoid infinitely growing histories.
- Conservative effective context windows usually perform better than maxing theoretical token limits.

---

## 8. Reusable Template

```markdown
# Task Header
- task_id:
- model_target:
- version:
- audience:
- depth:

## Objective
<one-sentence success definition>
Success criteria:
1) ...
2) ...

## Hard Constraints
- Must:
- Must not:
- Citation/policy rules:
- Length/format limits:

## Context (prioritized facts)
- Facts:
- Assumptions:
- Known unknowns:
- Confidence/timestamps (optional):

## Output Contract
- Format: Markdown | JSON
- Required headings/fields and order:
- Citation style:
- Missing-data fallback behavior:

## Evaluation Rubric
- Accuracy
- Completeness
- Structure compliance
- Evidence quality
- Risk/uncertainty handling

## Process Instructions
1) Produce concise outline + key points + evidence map.
2) Draft sections in required order.
3) Run rubric/checklist and revise gaps.
4) Return final output only.

## Verification Checklist
- [ ] All required sections present
- [ ] Constraints satisfied
- [ ] Unsupported claims flagged or removed
- [ ] Contradictions reconciled or called out
- [ ] Output contract satisfied
```

---

## 9. Contradictions Across Source Reports

The source files are broadly aligned, but these concrete contradictions/near-contradictions appear:

1. **Where the task instruction should appear**
   - Some reports place objective/task at the very beginning as the canonical default.
   - The Claude-authored report recommends putting target task/instruction near the **end** (recency zone), with role/format/constraints first.
   - Practical resolution: keep objective early, and add a short recency reminder of the task/constraints near the end for long contexts.

2. **Few-shot usage guidance (especially GPT/DeepSeek reasoning)**
   - Several reports recommend 1-2 concise examples (or up to 3-5 for complex Claude tasks).
   - Another report claims GPT-4.1 often does better zero-shot and that DeepSeek R1 degrades with few-shot.
   - Practical resolution: default to zero-shot for reasoning-heavy/modern instruction-following models; add minimal examples only when empirical eval shows gains.

3. **Chain-of-thought/"think step by step" instructions**
   - General reports encourage planning directives (plan first, outline first).
   - The Claude report specifically warns DeepSeek R1 may degrade with explicit CoT prompting.
   - Practical resolution: require explicit *observable artifacts* (outline, key points, checklist) instead of requesting hidden reasoning traces.

4. **Model-specific template breadth**
   - Most reports argue one canonical order should be reused across models.
   - The Claude report argues differences are large enough to maintain model-specific templates.
   - Practical resolution: keep one shared macro schema while maintaining provider-specific wrappers/delimiters and tuned defaults.

5. **Instruction repetition thresholds**
   - One report gives a concrete threshold (~8K+ tokens) for repeating critical instructions at both ends.
   - Others recommend front-loading without specific thresholding.
   - Practical resolution: apply repetition for long prompts by policy (e.g., >=8K) and evaluate model-specific breakpoints.

---

## 10. Source Material

Merged from:

- `codex-01-batch-completion-document-best-practices.md`
- `codex-02-completion-documents-batch-best-practices.md`
- `codex-03-batch-completion-document-best-practices.md`
- `codex-04-batch-completion-document-best-practices.md`
- `claude-01-batch-completion-guide.md`
