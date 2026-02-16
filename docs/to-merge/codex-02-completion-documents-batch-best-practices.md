# Best Practices Report: Writing "Completion" Documents for Batch Mode

## Executive summary
- Use a **rigid, predictable document skeleton** (goal → constraints → evidence → output contract → quality checks).
- Prefer **structured prompts** (XML/markdown section tags, explicit fields, acceptance criteria) over prose-only instructions.
- For high-stakes/complex outputs, use a **multi-pass pipeline** (plan → draft → critique → revise → final format) rather than one long call.
- Keep model-specific adjustments lightweight: mostly formatting and control-token conventions, not a totally different structure.
- Under token pressure, preserve **task objective, constraints, and output schema** first; cut examples/background first.

---

## 1) Recommended structure for completion documents

Use this order by default:

1. **Task header (short)**
   - `task_id`, `audience`, `domain`, `target length`, `deadline/latency tier`.
2. **Objective (non-negotiable)**
   - One paragraph with exactly what “good” means.
3. **Constraints and policy boundaries**
   - Must/should/must-not bullets (compliance, citation rules, tone, forbidden claims, source restrictions).
4. **Context packet (facts only)**
   - Curated notes/data snippets the model can rely on.
   - Label unknowns explicitly.
5. **Output contract**
   - Exact required sections, order, and format (Markdown headings, JSON schema, table columns, citation style).
6. **Evaluation rubric**
   - 3–7 checks to self-verify before final answer.
7. **Generation procedure (optional but powerful)**
   - e.g., “First produce outline, then full draft, then run rubric and patch gaps.”
8. **Final answer instruction**
   - “Return only final document” (or “return JSON only”).

### Minimal template

```markdown
# TASK
Task ID: <id>
Audience: <who reads this>
Objective: <single-sentence success condition>

# CONSTRAINTS
- Must: ...
- Must not: ...
- Style: ...
- Citations: ...

# CONTEXT
<facts, references, snippets>
Known unknowns: ...

# OUTPUT CONTRACT
Return Markdown with sections in this order:
1) ...
2) ...
3) ...
Length: ...

# QUALITY RUBRIC
- [ ] Requirement A satisfied
- [ ] Requirement B satisfied
- [ ] No contradictions with context

# INSTRUCTIONS
1) Make a short outline.
2) Write full draft.
3) Check against rubric; revise if needed.
4) Output final draft only.
```

---

## 2) Model-family differences (Claude, OpenAI, Llama, Granite, DeepSeek, Kimi, others)

### What usually stays the same
- Clear objective + constraints + output schema are universally beneficial.
- Explicit sectioning improves adherence across vendors.
- Asking for self-check against a rubric generally improves completeness.

### What often changes by model family

| Family | Practical adjustments |
|---|---|
| Claude (Anthropic) | Strong gains from explicit structure and tagged sections (XML-style segmentation is documented by Anthropic). Keep instructions unambiguous and role-separated. |
| OpenAI GPT models | Strong response to concise system/developer instructions, strict output schemas, and explicit success criteria. For batch generation, deterministic formatting instructions are important. |
| Llama-family (incl. hosted variants) | Often benefits from simpler, shorter instructions and concrete examples of desired format; avoid over-nested constraints. |
| Granite-family (IBM) | Prefer enterprise-style explicitness: compliance constraints, deterministic template, and validation checklist. |
| DeepSeek | For reasoning-capable variants, separate planning expectations from final output formatting, and constrain verbosity to avoid drift. |
| Kimi / Moonshot and similar | Use explicit output shape + language/tone controls; include short in-context format examples when possible. |

### Should section order change by model?
Usually **no**. Keep one canonical order for maintainability. Only change if a model repeatedly fails a specific step (e.g., put output schema earlier if formatting non-compliance is frequent).

---

## 3) If token constrained, what to cut first

Cut in this order:

1. **Redundant narrative/background**.
2. **Extra examples** (keep one high-value example max).
3. **Nice-to-have style guidance**.
4. **Extended rationale/explanations**.

Do **not** cut unless absolutely necessary:
- Objective success condition.
- Hard constraints/policy boundaries.
- Output schema/required section order.
- Core source facts.

### Compression tactics
- Convert paragraphs to bullet constraints.
- Replace prose with compact key-value fields.
- Reference stable external context by ID (if your platform supports retrieval) instead of re-sending full text.

---

## 4) One call vs multi-call decomposition

### Recommendation
For simple tasks: single call is usually enough.
For complex/high-quality tasks: multi-call is better.

### High-quality multi-call pipeline
1. **Plan call**: produce outline + claims inventory + open questions.
2. **Section drafting calls**: draft each major section against the same constraints.
3. **Synthesis call**: merge sections, normalize voice and transitions.
4. **Critique call**: run rubric + identify unsupported claims and missing requirements.
5. **Revision call**: fix critique findings only.
6. **Formatting call (optional)**: enforce exact schema/markdown/JSON.

This typically improves coverage and factual consistency, especially when the final document is long or heavily constrained.

### Alternative lightweight approach
If call budget is limited but >1 call available:
- Call 1: summary + headings + key points + evidence map.
- Call 2: full draft following that plan.
- Call 3: targeted QA pass.

This is often a sweet spot.

---

## 5) Should you ask the model to list points first?

Yes—usually helpful.

Requesting a brief **plan/points-first step** often improves:
- completeness,
- reduced topic drift,
- better section balance,
- easier human review before expansion.

Use bounded instructions, e.g.:
- “List 6–10 key points only, max 120 words total.”
- “Then draft section 1 using only listed points and provided context.”

Avoid unconstrained chain-of-thought requests; ask for concise intermediate artifacts (outline, checklist, claim table) instead.

---

## 6) Output-structure request patterns that maximize quality

Use explicit contracts such as:

1. **Schema-first prompting**
   - “Return valid JSON matching this schema…” (or strict markdown heading list).
2. **Checklist-gated generation**
   - Require a self-check table before finalizing (can be internal if you only want final output).
3. **Claim-evidence pairing**
   - For each major claim, require source/evidence pointer.
4. **Length and granularity bounds**
   - Per-section token/word budgets prevent early-section over-expansion.
5. **Failure-mode instruction**
   - “If context is insufficient, state `INSUFFICIENT_EVIDENCE` and list missing inputs.”
6. **Deterministic style constraints**
   - Tense, voice, audience level, forbidden phrases, citation style.

---

## 7) Best-possible quality strategy when unlimited iterative calls are allowed

Use an editorial workflow:

1. **Scoping**: objective, audience, rubric, schema.
2. **Content plan**: outline + key claims + dependency map.
3. **Evidence assembly**: gather citations/snippets; mark confidence per claim.
4. **Progressive drafting**: section-by-section generation with shared style guide.
5. **Adversarial review**: separate critique pass for contradictions, omissions, weak evidence.
6. **Targeted remediation**: patch only failing rubric items.
7. **Consistency pass**: terminology, definitions, numbering, cross-references.
8. **Final lint pass**: formatting and schema validation.

### Useful orchestration tricks
- Maintain a persistent **document state object** (outline, glossary, claims, decisions).
- Track unresolved questions explicitly to prevent hallucinated closure.
- Use model diversity for critique (different model/reasoning mode for reviewer pass).
- Keep prompts modular and reusable as templates per task class.

---

## 8) Practical default blueprint (ready to implement)

- **Pass A (planner)**: “Generate outline + key points + evidence needs.”
- **Pass B..N (writers)**: “Draft each section with fixed contract + per-section budget.”
- **Pass N+1 (editor)**: “Unify style, remove repetition, ensure logical flow.”
- **Pass N+2 (QA)**: “Score against rubric, return only failures with line refs.”
- **Pass N+3 (fixer)**: “Apply only required fixes, preserve structure.”

This generally outperforms one-shot long completions for complex deliverables.

---

## 9) Sources consulted (web)

- Anthropic prompt engineering overview and structured prompting guidance (incl. XML tags).
- Google Gemini prompting strategies documentation.
- DeepSeek reasoning model guide.
- Moonshot/Kimi platform usage docs.
- Community and vendor guidance on staged drafting, rubric-based evaluation, and iterative refinement patterns.

> Note: OpenAI docs pages were intermittently protected by anti-bot challenge during this run, so equivalent public guidance from other vendor docs was weighted more heavily.


### Reference URLs
- https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags
- https://ai.google.dev/gemini-api/docs/prompting-strategies
- https://api-docs.deepseek.com/guides/reasoning_model
- https://platform.moonshot.ai/docs/guide/start-using-kimi
- https://platform.openai.com/docs/guides/prompt-engineering (access limited in this environment)
- https://platform.openai.com/docs/guides/batch (access limited in this environment)
