# Best Practices for Writing Completion Documents for Batch Mode

## Executive recommendations (TL;DR)

1. Use a **fixed, predictable scaffold**: objective, constraints, source facts, acceptance criteria, output schema, and style guardrails.
2. Put **high-importance instructions first** and repeat critical constraints near the output schema.
3. Ask for a **brief plan/outline first**, then full prose, either in one call (two-pass in a single response) or multiple calls for high-stakes work.
4. Force **grounding and traceability**: require explicit assumptions, unresolved questions, and confidence/uncertainty notes.
5. For token limits, cut style extras first; preserve objective, constraints, factual inputs, and acceptance tests.
6. Prefer **JSON or tightly constrained Markdown sections** for structural reliability in batch systems.

---

## 1) Recommended structure for completion documents

Use this order unless you have a strong reason to change it:

1. **Task objective**
   - One sentence defining the final deliverable and intended audience.
2. **Success criteria / rubric**
   - Bullet list of what a "good" answer must contain.
3. **Hard constraints**
   - Do-not-violate rules (format, length, banned content, required citations, policy boundaries).
4. **Input context and source facts**
   - Facts, references, data excerpts, and known assumptions.
5. **Required output schema**
   - Exact section headings and order (or JSON schema).
6. **Reasoning process instructions**
   - Ask for concise planning notes and explicit assumptions (without requesting hidden chain-of-thought).
7. **Quality checks before finalizing**
   - Self-check list aligned with the rubric.
8. **Fallback behavior**
   - What to do if data are missing, ambiguous, or conflicting.

### Why this order works

- It resolves ambiguity early (objective, rubric, constraints) before model generation starts drifting.
- It reduces format failures by pinning schema before prose.
- It improves determinism for batch workflows because validation criteria are explicit.

---

## 2) Model-family differences (Claude, OpenAI, Llama, Granite, DeepSeek, Kimi, others)

Most of the structure above is universal. Differences are usually in **format robustness**, **instruction precedence sensitivity**, and **context-window usage**.

### Practical guidance by family

- **Claude (Anthropic)**
  - Usually follows structured instructions well when given clear XML/Markdown section boundaries.
  - Works well with explicit constitutional constraints and "what to do when uncertain."
  - Keep instruction hierarchy explicit: role -> task -> constraints -> format.

- **OpenAI models**
  - Strong schema-following when given explicit output contracts and examples.
  - Benefit from concise, high-signal prompts with clear separators and acceptance tests.
  - Use strict JSON mode or function/tool schemas when downstream parsing is critical.

- **Llama-family (self-hosted/open weights variants)**
  - Behavior varies by fine-tune; often needs tighter formatting instructions and shorter, less ambiguous prompts.
  - Include concrete mini-examples of desired section style.
  - Prefer smaller chunks and iterative refinement for long, high-fidelity documents.

- **Granite (IBM)**
  - Enterprise settings often reward explicit governance constraints, citation requirements, and deterministic templates.
  - Keep policy, provenance, and validation fields explicit in the schema.

- **DeepSeek**
  - Strong long-form generation; can drift without strict structural anchors.
  - Use section-by-section constraints and re-anchor with a required checklist at the end.

- **Kimi (Moonshot)**
  - Strong long-context behavior in many settings; good for large source packs.
  - Still enforce compact section schema so long contexts do not produce diffuse outputs.

### Should section order change per model?

Usually **no**. Keep the same macro order. Adjust only:
- strictness of schema constraints,
- amount of examples,
- chunk size / number of iterative calls.

Consistency across models makes A/B testing and orchestration easier.

---

## 3) If token-constrained: what to cut first

Cut in this order (top = first to cut):

1. **Stylistic flourishes** (tone directives, rhetorical preferences, duplicated wording).
2. **Non-essential background narrative** (keep only facts required to answer).
3. **Few-shot examples** (keep one minimal exemplar if needed for format compliance).
4. **Secondary sections** (nice-to-have analysis, extended alternatives).
5. **Only then compress core constraints** (never remove objective, hard constraints, or acceptance criteria).

### Compression strategy

- Keep a "minimal core prompt" block always present:
  - objective,
  - hard constraints,
  - required schema,
  - acceptance tests.
- Move extra context to referenced attachments/chunks and retrieve only relevant slices.

---

## 4) One-call vs multi-call decomposition

### Is splitting into multiple calls helpful?

For complex tasks, **yes**, especially when quality matters more than latency/cost.

Recommended staged pattern:
1. **Planning call**: summary, assumptions, proposed headings, evidence gaps.
2. **Section drafting calls**: one or a few sections per call, each with local constraints.
3. **Synthesis call**: unify voice, resolve cross-section inconsistencies, remove duplication.
4. **Critique/QA call**: check against rubric and constraints; list defects.
5. **Repair call**: fix defects only.

### When one call is enough

A single call is often enough for:
- short, low-risk documents,
- highly templated outputs,
- cases with clear, narrow source material.

### Hybrid best practice

Ask the model to generate:
1) brief outline + key points, then 2) full draft in the same response.

This gives many benefits of decomposition with one API round trip. For highest reliability, split into multiple calls anyway.

---

## 5) How to request structure for best output quality

Use explicit, testable instructions:

- "Return exactly these H2 sections in this order: ..."
- "Under each section, provide 3-5 bullets, each starting with an action verb."
- "Mark assumptions as `Assumption:` and unknowns as `Unknown:`."
- "End with `Quality Checklist` and evaluate pass/fail for each criterion."

### Strong prompt patterns

1. **Contract-first prompting**
   - Give output contract before content body.
2. **Delimiters and tags**
   - Use fenced blocks or XML tags around inputs and required schema.
3. **Rubric anchoring**
   - Provide scoring dimensions; ask model to self-evaluate against them.
4. **Progressive expansion**
   - Outline -> section drafts -> integrated final.
5. **Targeted regeneration**
   - Regenerate only failing sections instead of full document rewrites.

### Should you ask for point listing first?

Yes. Asking for a short list of intended points/headings before full prose generally improves coherence and reduces omission.

If you can make many requests, do this explicitly as a separate planning step.

---

## 6) High-quality batch pipeline (best-answer objective)

If request count is effectively unconstrained, use this pipeline:

1. **Ingest + normalize**
   - Normalize source docs, deduplicate, tag by topic and confidence.
2. **Plan**
   - Generate candidate outlines and choose one via rubric scoring.
3. **Draft sections in parallel**
   - Produce section drafts with section-specific constraints.
4. **Evidence pass**
   - Ensure claims map to sources; flag unsupported assertions.
5. **Consistency pass**
   - Align terminology, numbers, definitions, and tense/voice.
6. **Adversarial critique pass**
   - Ask a separate call to identify weak logic, missing counterpoints, or overclaims.
7. **Repair pass**
   - Fix only diagnosed issues.
8. **Final format lock**
   - Enforce schema exactly; run parser/validator.
9. **Human-readability polish**
   - Optional final pass for clarity and brevity.

### Orchestration tips

- Keep per-call prompts narrowly scoped.
- Carry forward a compact "decision log" so later calls know why choices were made.
- Use deterministic settings where possible for formatting-critical stages.

---

## 7) Reusable template

```markdown
# Task
<one-sentence objective and audience>

# Success Criteria
- <criterion 1>
- <criterion 2>

# Hard Constraints
- <must include / must not include>
- <length, citation, style, policy>

# Inputs
<facts, references, assumptions>

# Required Output Format
## Section A
## Section B
## Section C

# Process Instructions
1. First provide a 5-bullet outline.
2. Then provide the full document.
3. Mark assumptions and unknowns explicitly.

# Quality Checklist (must pass)
- <check 1>
- <check 2>
```

---

## 8) Web references reviewed

- Anthropic documentation on prompt engineering and structure controls.
- Meta/Llama prompting guidance.
- IBM guidance on enterprise prompt engineering patterns.
- DeepSeek API prompt engineering guide.
- Moonshot/Kimi platform documentation.
- Wei et al. (2022) *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*.
- Yao et al. (2023) *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*.

(Notes: some vendor docs are JavaScript-rendered and may vary by region/session; principles above emphasize stable, cross-vendor practices.)
