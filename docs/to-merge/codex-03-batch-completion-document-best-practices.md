# Best Practices for Writing "Completion" Documents for Batch Mode

## Purpose
This report provides practical guidance for designing completion-style input documents that produce high-quality, reliable outputs in batch pipelines. It covers:
- Recommended document structure.
- Model-family differences (Claude, OpenAI GPT, Llama, Granite, DeepSeek, Kimi, and similar models).
- What to trim first under token constraints.
- Whether and when to split work across multiple calls.
- Prompting patterns that improve output quality.

---

## Executive Summary
1. **Use a stable, explicit template** with clear sections in this order: objective, constraints, context, required output schema, quality bar, and validation checklist.
2. **Front-load critical instructions** because all model families show some degradation on late instructions in long contexts.
3. **For best quality, use staged generation** for complex tasks (plan -> section drafts -> synthesis -> verification), not one-shot generation.
4. **Under token pressure, cut examples and background first**, never objective/constraints/schema/rubric.
5. **Model-specific tuning is real but modest**: keep the same high-level structure, then lightly adapt formatting, verbosity controls, and schema strictness per model family.

---

## Recommended Document Structure (Canonical Template)
Use this baseline structure for batch completion inputs.

### 1) Task Header (short)
- `Task-ID`
- `Audience`
- `Desired depth`
- `Deadline/latency budget`

### 2) Objective (non-negotiable)
- One sentence primary goal.
- 3-7 concrete success criteria.

### 3) Constraints and Guardrails (non-negotiable)
- Must include: prohibited content/actions, compliance requirements, citation requirements, formatting limits.
- Explicitly list assumptions allowed vs disallowed.

### 4) Context Pack (prioritized)
- Put **most relevant facts first**.
- Label each source snippet with confidence and timestamp.
- Separate facts from hypotheses.

### 5) Output Contract (non-negotiable)
- Required sections and order.
- Required style (bullets, tables, tone, length bands).
- Structured schema (JSON/YAML/Markdown heading contract).
- Explicitly say what to do if information is missing.

### 6) Quality Rubric
- Define scoring dimensions (accuracy, completeness, reasoning transparency, actionability, risk awareness).
- Include pass/fail thresholds.

### 7) Self-Check Instructions
- Ask model to run a short checklist before finalizing:
  - Did I satisfy every required section?
  - Did I violate any constraints?
  - Are uncertainties flagged?

### 8) Optional: Planning Scratchpad (hidden/internal if supported)
- If your platform supports hidden reasoning/planning channels, ask for a concise plan first and then final answer.
- If not, request a **brief public outline** before full generation.

---

## Should Structure Differ by Model Family?
**High-level order should stay mostly constant**. Most gains come from better constraints and decomposition, not radical reordering. That said, these adjustments are useful:

### Claude (Anthropic)
- Strong with long-context synthesis and constitutional-style constraints.
- Works well with explicit "what good looks like" rubric and XML-like delimiters.
- Keep instructions crisp; avoid contradictory style rules.

### OpenAI GPT models
- Strong instruction following and schema compliance when output contract is explicit.
- Prefer clear section contracts + short examples.
- For structured outputs, strict schema instructions generally improve reliability.

### Llama-family (open models, varying alignment)
- More variability across hosting/fine-tuning variants.
- Benefits from explicit step-by-step scaffolding and concise constraints.
- Often improved by shorter context and stronger redundancy on key constraints.

### Granite (IBM) and enterprise-tuned models
- Often optimized for enterprise policy/compliance and extraction tasks.
- Keep a compliance checklist and deterministic output format.
- Reduce ambiguity in role and task framing.

### DeepSeek / Kimi and other high-capability reasoning/chat models
- Typically perform well with decomposition and explicit evaluation criteria.
- Ask for uncertainty labeling and alternatives for ambiguous tasks.
- Keep output schema strict if downstream automation depends on parsing.

### Practical takeaway
- **Do not heavily reorder sections by model**.
- Instead tune:
  1. Delimiter style (XML/Markdown/JSON),
  2. Schema strictness,
  3. Redundancy of critical constraints,
  4. Verbosity caps.

---

## If Token-Constrained, What to Cut First?
Use this cut order (top = cut first):

1. Extra prose/background narrative.
2. Redundant examples (keep 1 best exemplar).
3. Nice-to-have style preferences.
4. Secondary analysis sections.
5. Historical context.

**Never cut these unless impossible:**
- Objective.
- Hard constraints.
- Output contract/schema.
- Minimum quality rubric.

Compression tactics:
- Replace paragraphs with bullet atoms.
- Convert repeated rules into compact checklist IDs.
- Move large references to short retrieved snippets rather than full dumps.

---

## One Call vs Multi-Call Decomposition
For simple tasks, one call is often enough if you request an initial outline and key points before detailed prose.

For complex/high-stakes tasks, **multi-call is usually better**:

### Recommended multi-call pipeline
1. **Plan call**: generate thesis, section list, key claims, evidence needed.
2. **Section calls**: draft each major section with local context windows.
3. **Synthesis call**: merge sections, normalize tone and terminology.
4. **Critique call**: independent reviewer prompt checks gaps/risks.
5. **Repair call**: apply critique and produce final document.

Why this works:
- Reduces long-context drift.
- Improves completeness and internal consistency.
- Allows targeted retrieval per section.
- Enables explicit QA gates.

When one-call can suffice:
- Short tasks, low risk, limited sources, weak latency budget.

---

## Prompting Guidelines to Maximize Output Quality

### A. Ask for a "content plan" first
Yes—requesting key points/outline first usually improves downstream coherence.
- Require: thesis, claims, evidence slots, risks, and open questions.
- Then ask model to draft using only that plan + provided sources.

### B. Use explicit section contracts
State exact headings and output order. Example:
- `## Executive Summary`
- `## Findings`
- `## Recommendations`
- `## Risks and Unknowns`

### C. Provide evaluation criteria in the prompt
Models optimize toward what you measure. Include a rubric and acceptance test.

### D. Force uncertainty handling
Require model to tag each claim as:
- `Supported`, `Inferred`, or `Speculative`.

### E. Require source linkage
For factual work, require per-claim citations or evidence IDs.

### F. Add a final compliance pass
Ask model to output a short checklist confirming:
- all sections present,
- constraints met,
- unresolved questions listed.

---

## Suggested Master Template (Copy/Paste)

```markdown
# Task
ID: <id>
Audience: <audience>
Depth: <brief|standard|deep>

## Objective
<one-sentence goal>
Success criteria:
1. ...
2. ...

## Hard Constraints
- Must ...
- Must not ...
- Length budget: ...
- Citation requirement: ...

## Context (ranked)
[Source A | confidence high | date]
...
[Source B | confidence medium | date]
...

## Required Output Schema
Return Markdown with exactly these sections in this order:
1. Executive Summary (max N bullets)
2. Detailed Analysis
3. Recommendations (prioritized)
4. Risks / Unknowns
5. Validation Checklist

## Quality Rubric (pass >= 4/5 each)
- Accuracy
- Completeness
- Actionability
- Constraint compliance
- Clarity

## Process
Step 1: Produce concise outline + key points only.
Step 2: Wait (or continue) to full draft following the approved outline.
Step 3: Run self-check and output final answer.
```

---

## Research and Documentation Signals (Web)
The recommendations above align with broadly consistent guidance from major model providers and prompt-engineering literature:
- Anthropic prompt engineering guidance emphasizes clear instructions, explicit structure, and iterative refinement.
- OpenAI prompt best practices similarly stress instruction clarity, format specification, and decomposition for complex tasks.
- Community and research patterns (chain-of-thought alternatives, plan-then-write, self-critique/revision loops) show gains on complex generation and reliability workflows.

Useful references:
- Anthropic Prompt Engineering Overview: https://docs.anthropic.com/
- OpenAI Prompt Engineering guides: https://platform.openai.com/docs/guides/prompt-engineering
- Prompting Guide (community compendium): https://www.promptingguide.ai/
- "Self-Refine" style iterative generation/editing (paper): https://arxiv.org/abs/2303.17651
- "Reflexion" style verbal feedback loops (paper): https://arxiv.org/abs/2303.11366

---

## Practical "Best Possible Answer" Strategy (Unlimited Calls)
If number of calls is unconstrained and quality is paramount:

1. **Scope call**: clarify objective, audience, constraints, and scoring rubric.
2. **Retrieval call(s)**: gather and rank evidence chunks.
3. **Planning call**: generate argument map + section skeleton.
4. **Parallel section drafting**: one call per section with focused context.
5. **Consistency merge**: unify style, terms, and references.
6. **Adversarial review**: separate call tasked to find flaws and missing counterpoints.
7. **Fact-check pass**: verify each claim against evidence IDs.
8. **Compression pass**: tighten language while preserving rubric scores.
9. **Final QA pass**: enforce schema and checklist compliance.

This pipeline generally beats single-pass generation for depth, factual discipline, and structural reliability.

---

## Follow-up Questions (to tailor implementation)
1. Are your completion jobs primarily **factual reports**, **policy/procedure docs**, or **creative/strategy outputs**?
2. Do you require strict machine-parseable output (JSON) or human-readable Markdown is enough?
3. Is latency/cost more important than quality, or is quality the only priority?
4. Do you already have a citation/evidence store we should integrate into the template?
