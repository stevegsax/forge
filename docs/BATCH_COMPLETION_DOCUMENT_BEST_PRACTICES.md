# Best Practices for Batch-Mode "Completion" Documents

This report describes how to structure completion-ready documents for batch LLM execution, how to tune that structure by model family, and how to design multi-pass pipelines for maximum answer quality.

## Executive Recommendations

1. **Use a strict, repeatable scaffold** with explicit sections, constraints, and acceptance criteria.
2. **Place high-priority instructions first** (goal, hard constraints, output schema), then context, then optional guidance.
3. **Request a short planning pass before long-form writing** (outline + key claims + evidence map), then generate section-by-section.
4. **Prefer multi-call orchestration for complex tasks**: planning, drafting, critique, revision, and final format validation.
5. **Token-trim in this order**: examples and style flourishes first, then secondary context, then low-priority appendices; preserve requirements and rubric.
6. **Use model-specific formatting affordances** (Claude XML tags, JSON schema for structured outputs, role clarity for chat-native models).

---

## 1) Recommended Structure for Completion Documents

Use this section order for most providers:

1. **Task Header**
   - Task ID, version, timestamp, model target.
2. **Primary Objective (1-3 lines)**
   - Single-sentence success definition.
3. **Hard Constraints / Non-Negotiables**
   - Must-do and must-not-do rules.
4. **Output Contract**
   - Exact schema or markdown outline required in final answer.
5. **Evaluation Rubric**
   - How quality is judged (accuracy, completeness, citations, tone, etc.).
6. **Essential Context**
   - Facts, source snippets, assumptions, scope boundaries.
7. **Reasoning/Process Instructions**
   - e.g., “first list key points, then draft,” “flag uncertainty,” “cite evidence.”
8. **Section-by-Section Requirements**
   - Bullet requirements for each document section.
9. **Few-shot Examples (optional, concise)**
   - 1-2 high-signal examples only.
10. **Final Verification Checklist**
    - Self-check before final output (schema, missing claims, contradictions).

### Why this order works

- It aligns with instruction prioritization behavior in modern LLMs: early high-salience constraints reduce drift.
- It gives the model a “decoder target” early (output contract), improving coherence in long generations.
- It reduces completion variance by making quality criteria explicit.

---

## 2) Provider-Specific Structure Differences

The **core order should usually remain the same**. Change only packaging details:

### Claude (Anthropic)

- Use XML-style structural tags for instruction regions (`<objective>`, `<constraints>`, `<format>`).
- Keep explicit section boundaries and short, concrete directives.
- Strongly separate required behavior vs optional style guidance.

### OpenAI models

- Keep concise role/task framing, then explicit output schema and acceptance criteria.
- For complex outputs, ask for an initial plan or headings first, then expand.
- Use JSON schema or strictly defined markdown headings when deterministic parsing matters.

### Llama-family models

- Prefer simpler, explicit instruction wording and concrete section templates.
- Reduce ambiguity and avoid overly abstract policy language.
- Reinforce output shape with literal heading stubs.

### Granite (IBM)

- Keep enterprise-style explicit constraints and verifiable criteria.
- Use tightly scoped prompts and clear “input → required output” mappings.
- Encourage factual grounding and concise response objectives.

### DeepSeek

- Works well with OpenAI-compatible chat formats plus explicit structure.
- If using reasoning/thinking variants, separate “thinking budget policy” from output format.
- Use JSON mode for strict machine-readable outputs where needed.

### Kimi (Moonshot)

- Favor explicit formatting instructions and concise scope boundaries.
- For long context tasks, specify source hierarchy and citation expectations.
- Use staged prompting (outline then expansion) for very long documents.

### Should section order change across providers?

- **Usually no**: objective/constraints/output contract should stay near the top for all.
- **What changes** is syntax and strictness:
  - Claude: XML wrappers help.
  - OpenAI/DeepSeek/Kimi: schema + bullet constraints.
  - Llama/Granite: simpler wording, strong templates, less ambiguity.

---

## 3) If Token-Constrained, What to Cut First

Cut in this priority order (top = safest to remove):

1. Redundant prose and style instructions.
2. Multiple examples (keep one best example).
3. Nice-to-have sections or optional edge-case handling.
4. Secondary background context.
5. Detailed rationale text.

Avoid cutting these unless absolutely necessary:

- Primary objective and hard constraints.
- Output contract/schema.
- Evaluation rubric.
- Critical source facts.

A practical compression pattern:

- Convert paragraphs to bullets.
- Replace long context with a “fact table” of atomic claims.
- Move optional guidance to a later refinement pass.

---

## 4) One Long Call vs Multiple Calls

For high-quality complex deliverables, **multi-call pipelines are usually superior**.

### Recommended pipeline

1. **Plan pass**
   - Output: thesis, section outline, key claims, evidence gaps.
2. **Draft pass(es)**
   - Generate one section at a time with local constraints.
3. **Critique pass**
   - Check for missing requirements, unsupported claims, weak logic.
4. **Revision pass**
   - Apply targeted fixes only.
5. **Final format validation pass**
   - Enforce schema, heading order, style, and completeness checklist.

### Why multi-pass wins

- Reduces context dilution across long generations.
- Improves consistency by anchoring on a stable outline.
- Enables deterministic quality gates between passes.
- Makes retries cheap (rerun only failed section).

### When a single call is enough

- Short/simple tasks with low consequence of omission.
- Strong, compact scaffold + explicit checklist + strict output format.

---

## 5) Should You Ask for Points First Before Full Draft?

**Yes, generally.** Asking for key points first is high-leverage because it:

- Commits the model to coverage before stylistic elaboration.
- Surfaces scope gaps early.
- Improves coherence in later sections.

Best pattern:

1. Ask for **outline + key points + evidence map**.
2. Approve or edit this intermediate artifact.
3. Expand section-by-section while preserving approved points.
4. Run final consistency and completeness checks.

---

## 6) Prompting Guidelines to Maximize Document Quality

1. **Define success like a test**
   - “A response is acceptable only if it includes X, Y, Z.”
2. **Specify output grammar**
   - JSON schema or fixed markdown headings.
3. **Constrain scope explicitly**
   - In-scope/out-of-scope lists.
4. **Force uncertainty handling**
   - Require confidence levels and explicit unknowns.
5. **Require evidence linkage**
   - Claim → source mapping when possible.
6. **Use decomposition directives**
   - “Plan first, then draft.”
7. **Add self-check checklist**
   - Missing section, contradiction, schema, constraint adherence.
8. **Prefer short imperative instructions**
   - Avoid narrative prompt text.

---

## 7) Suggested “Completion Document” Template

```markdown
# Task Header
- task_id:
- model_target:
- version:
- date:

## Objective
<one-sentence success definition>

## Hard Constraints
- Must:
- Must not:

## Output Contract
- Format: (Markdown | JSON)
- Required headings/fields:
- Citation rules:
- Length targets:

## Evaluation Rubric
- Accuracy:
- Completeness:
- Structure compliance:
- Evidence quality:

## Context
- Facts:
- Assumptions:
- Scope boundaries:

## Generation Procedure
1) Produce outline + key points.
2) Expand sections in order.
3) Run verification checklist.

## Section Requirements
### Section A
- ...
### Section B
- ...

## Verification Checklist
- [ ] All required sections present
- [ ] Constraints satisfied
- [ ] No unsupported claims
- [ ] Output contract satisfied
```

---

## 8) Advanced Techniques (Best-Answer Objective, Unlimited Calls)

If your objective is the best possible answer and you can make many calls:

1. **N-best planning**: generate 3 outlines, score them, merge the best.
2. **Parallel section drafting**: draft sections independently, then reconcile style and claims.
3. **Adversarial review pass**: use a separate call to find weaknesses/omissions.
4. **Source-grounded rewrite**: force each key claim to cite evidence.
5. **Deterministic formatting pass**: final call only for structure normalization.
6. **Majority-vote for contentious claims**: compare independent reasoning trajectories.
7. **Rubric scoring loop**: model scores against rubric, revises until threshold.

---

## 9) Practical Recommendation for Forge Batch Mode

Given Forge’s batch/state-machine design, the strongest pattern is:

- **Step 1 (planner)**: outline + key claims + dependency graph.
- **Step 2..N (section writers in parallel where possible)**: constrained section drafts.
- **Step N+1 (integrator)**: merge sections, normalize terms and style.
- **Step N+2 (validator)**: rubric + constraint checks, produce fix list.
- **Step N+3 (finalizer)**: apply fixes and emit final contract-compliant document.

This preserves determinism, supports retries per step, and improves final answer quality.

---

## Web Sources Reviewed

- Anthropic Claude prompt engineering overview and XML-tag guidance:  
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview  
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags
- Google Gemini prompt design strategies:  
  https://ai.google.dev/gemini-api/docs/prompting-strategies
- DeepSeek API docs (chat prefix completion, JSON output, thinking mode):  
  https://api-docs.deepseek.com/guides/chat_prefix_completion  
  https://api-docs.deepseek.com/guides/json_mode  
  https://api-docs.deepseek.com/guides/thinking_mode
- Moonshot/Kimi docs sitemap and prompt best-practice page:  
  https://platform.moonshot.cn/sitemap-0.xml  
  https://platform.moonshot.cn/docs/guide/prompt-best-practice
- Meta Llama prompt engineering how-to:  
  https://www.llama.com/docs/how-to-guides/prompting/
- IBM Granite prompt engineering docs:  
  https://www.ibm.com/granite/docs/use-cases/prompt-engineering/
- OpenAI-adjacent prompting references used due Cloudflare access constraints on platform/help pages in this environment:  
  https://cookbook.openai.com/examples/gpt4-1_prompting_guide  
  https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering
