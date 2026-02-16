# Batch Completion Guide

Best practices for structuring documents submitted to LLMs via batch completion APIs. Written for Forge's batch-first architecture but applicable to any system that treats LLM calls as document completions.

**Scope.** This guide covers document structure, model-specific considerations, token-budget triage, multi-pass generation strategies, output structuring techniques, and advanced refinement patterns. The assumption throughout is that the caller can make as many batch requests as needed — the goal is to maximize output quality, not minimize API calls.

---

## Table of Contents

1. [Document Structure](#1-document-structure)
2. [Model-Specific Considerations](#2-model-specific-considerations)
3. [Token-Budget Triage](#3-token-budget-triage)
4. [Single-Pass vs. Multi-Pass Generation](#4-single-pass-vs-multi-pass-generation)
5. [Output Structuring for Quality](#5-output-structuring-for-quality)
6. [Advanced Techniques](#6-advanced-techniques)
7. [Composite Architecture](#7-composite-architecture)
8. [Sources](#8-sources)

---

## 1. Document Structure

### 1.1 The Core Principle: Primacy and Recency

LLMs exhibit a U-shaped attention curve. Information at the **beginning** and **end** of the context window is recalled most accurately; information in the **middle** is neglected. This is the "Lost in the Middle" effect (Liu et al., 2023), caused by Rotary Position Embedding (RoPE) decay patterns fundamental to transformer attention.

The measured impact is significant:

- **>30% accuracy degradation** when relevant information shifts from start/end to middle positions.
- **24% accuracy drop** at 30K tokens vs. short-context on Llama-3.1-8B, even with perfect retrieval.
- **17-20% degradation** attributable to input length alone, regardless of position.

The practical consequence: **document structure is not cosmetic — it directly affects output quality.**

### 1.2 Recommended Section Ordering

Place the most important information at the beginning and end. Place reference material in the middle. This ordering also aligns with prompt caching (stable prefix, volatile suffix) for cost efficiency.

```
┌─────────────────────────────────────────────────────┐
│ BEGINNING (primacy zone — high recall)              │
│                                                     │
│  1. Role and identity                               │
│  2. Output format requirements                      │
│  3. Hard constraints and rules                      │
│                                                     │
├─────────────────────────────────────────────────────┤
│ MIDDLE (lower recall — reference material)          │
│                                                     │
│  4. Background knowledge / knowledge base           │
│  5. Examples (few-shot, if applicable)              │
│  6. Repository structure / dependency context       │
│  7. Supporting documents / playbooks                │
│                                                     │
├─────────────────────────────────────────────────────┤
│ END (recency zone — high recall)                    │
│                                                     │
│  8. Target content (files being modified, etc.)     │
│  9. The specific task instruction                   │
│  10. Restatement of critical constraints            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Why this ordering works:**

- Items 1-3 establish the framework the LLM will use to interpret everything that follows. Placed first, they benefit from primacy bias and set the generation trajectory.
- Items 4-7 are reference material the LLM consults as needed. They tolerate the middle-position recall penalty because they are supporting context, not the core task.
- Items 8-10 are what the LLM must act on. Placed last, they benefit from recency bias and are closest to the generation point.

### 1.3 Cache-Efficient Ordering (Anthropic)

For Anthropic's batch API, prompt caching is prefix-based. The cache key is built from **tools → system → messages** in order, and any change to earlier content invalidates everything after it. This creates a natural alignment with the attention-optimal ordering:

```
BREAKPOINT 1 (most stable — cached across all tasks):
  ├─ Tool definitions (via API tools field)
  ├─ Role prompt
  ├─ Output format requirements
  └─ Project-level instructions (e.g., CLAUDE.md)

BREAKPOINT 2 (stable within a task — cached across retries):
  ├─ Repository structure / repo map
  ├─ Playbooks / knowledge base
  ├─ Task description
  ├─ Target file list
  └─ Target file contents

BREAKPOINT 3 (semi-stable — added during execution):
  ├─ Exploration results (context discovered mid-execution)
  └─ Additional context items

VOLATILE (changes on retry — invalidates cache from here):
  └─ Previous attempt errors with code context
```

Cache reads cost 10% of base input price on Anthropic. With multi-step execution and retries, this ordering yields up to 80% cost reduction on input tokens.

### 1.4 Instruction Repetition for Long Contexts

For documents exceeding ~8K tokens, repeat critical instructions at both the beginning and end of the context. OpenAI's GPT-4.1 guide explicitly recommends this. The technique works across models because it exploits both primacy and recency simultaneously, and it costs minimal additional tokens relative to the quality improvement.

```
SYSTEM: "You must respond with valid JSON. Do not include explanatory text."
...
[8K+ tokens of context]
...
USER: "Remember: respond with valid JSON only. No explanatory text."
```

### 1.5 Section Delimiters

Use explicit delimiters to help the model parse document structure. The choice of delimiter format is model-dependent (see Section 2), but the principle is universal: **named, consistent delimiters improve parsing accuracy for all models.**

Good:
```xml
<task_description>
Refactor the authentication module to use JWT tokens.
</task_description>

<target_files>
src/auth/handler.py
src/auth/tokens.py
</target_files>
```

Also good:
```markdown
## Task Description
Refactor the authentication module to use JWT tokens.

## Target Files
- src/auth/handler.py
- src/auth/tokens.py
```

Bad:
```
Here is the task: Refactor the authentication module to use JWT tokens.
The files are src/auth/handler.py and src/auth/tokens.py.
```

---

## 2. Model-Specific Considerations

Different LLMs have different structural preferences. The differences are significant enough that a multi-model orchestrator should maintain model-specific prompt templates. This section covers the major model families.

### 2.1 Claude (Anthropic)

**Structural preference:** XML tags. Claude was trained on XML-tagged data and is the model most natively responsive to XML-structured prompts.

**Key characteristics:**
- **Literal instruction following.** Claude 4.x does exactly what you say. If you want the model to go beyond the stated task (e.g., suggest additional improvements), you must explicitly ask. Provide motivation and context behind instructions, not just the instructions themselves.
- **System prompt for scene-setting.** Use the system parameter for role, tone, and persistent behavioral constraints. Put detailed task-specific instructions in the user message.
- **Assistant prefill for document completion.** Supply partial content in the `assistant` role and Claude continues from that point. Prefilling `{` forces JSON output; prefilling `<result>` forces XML output. *Note: prefilling is deprecated on Claude Opus 4.6 and Sonnet 4.5 — use structured outputs or system prompt instructions instead on these models.*
- **Cache control.** Explicit `cache_control: {"type": "ephemeral"}` markers, up to 4 breakpoints. Minimum 1,024 tokens per checkpoint. Default 5-minute TTL; 1-hour TTL available at 2x write cost (recommended for batch workloads).
- **Few-shot examples.** 3-5 diverse, relevant examples recommended for complex tasks. Wrap in XML tags.

**Recommended prompt structure:**
```xml
<system>
You are a code generation assistant for a Python project.

<output_requirements>
Respond with a valid JSON object containing "edits" and "explanation" fields.
</output_requirements>

<project_context>
{repo_map, playbooks, project instructions}
</project_context>
</system>

<user>
<task>
Refactor the authentication module to use JWT tokens.
</task>

<target_files>
{file contents}
</target_files>

<instructions>
Use the edits field for search/replace changes to existing files.
Respond with valid JSON only.
</instructions>
</user>
```

### 2.2 OpenAI (GPT-4.1)

**Structural preference:** Markdown headers or XML tags. Both work well; XML tags performed best in Cursor's testing for document retrieval tasks.

**Key characteristics:**
- **Equally literal instruction following (as of GPT-4.1).** Like Claude 4.x, GPT-4.1 follows instructions very literally. Prompts must be precise with no contradictions or vague directions.
- **Zero-shot often outperforms few-shot.** GPT-4.1 performs worse with few-shot examples in some benchmarks. Clear structural organization with zero-shot prompting is often more effective.
- **Agentic boost.** Three instructions that increased SWE-bench scores by ~20%: (1) "Keep going until the query is completely resolved," (2) "Use tools to gather information rather than guessing," (3) "Plan extensively before each action."
- **Automatic prefix caching.** No explicit markers needed — the system caches the longest matching prefix for prompts >1,024 tokens. Cached tokens cost 50% of base input price.
- **Instructions at both ends.** For long contexts (>8K tokens), GPT-4.1 measurably benefits from instructions at both the beginning and end.
- **Use the tools field.** Pass tool definitions via the API `tools` field, not inline in the prompt. This yielded a 2% improvement on SWE-bench Verified.

**Recommended prompt structure:**
```markdown
# Role and Objective
You are a code generation assistant for a Python project.

# Instructions
## Output Format
Respond with a valid JSON object containing "edits" and "explanation" fields.

## Constraints
- Use search/replace edits for existing files
- Do not modify files not listed in Target Files

# Context
{repo_map, playbooks, project instructions}

# Target Files
{file contents}

# Task
Refactor the authentication module to use JWT tokens.

# Reminder
Respond with valid JSON only. Use the edits field for all changes.
```

### 2.3 DeepSeek

**Critical distinction:** DeepSeek R1 (reasoning) and DeepSeek V3 (chat) require fundamentally different prompt structures. Using the wrong approach degrades output significantly.

**DeepSeek R1 (reasoning model):**
- **No system prompt.** Place all instructions in the user message. R1's RL training was tuned to respond to user queries directly.
- **No chain-of-thought prompting.** R1 reasons internally; explicit "think step by step" instructions interfere with its native reasoning and degrade performance.
- **No few-shot examples.** Research consistently shows they degrade R1's output.
- **Minimal, direct prompts.** State the problem and let the model determine the reasoning path. Tell R1 the purpose, not the steps.
- **Temperature 0.5-0.7** (0.6 recommended), top_p 0.95.

**DeepSeek V3 (chat model):**
- Standard prompting techniques work: system prompts, few-shot, chain-of-thought.
- Supports function calling, JSON mode, prefix completions.
- Context caching with stable prefix (similar design to Anthropic).

**Key implication for multi-model orchestrators:** The orchestrator must detect whether it is targeting a reasoning-class or chat-class model and select the appropriate prompt construction strategy.

### 2.4 Llama (Meta)

**Structural preference:** Special tokens specific to each Llama version. Using the wrong token format significantly degrades output.

**Key characteristics:**
- **Version-specific chat templates.** Llama 2 uses `[INST]`/`<<SYS>>`, Llama 3.x uses `<|start_header_id|>`, Llama 4 uses `<|header_start|>`. Always use `tokenizer.apply_chat_template()` or pass message dicts to ensure correct formatting.
- **Limited effective context.** Claims 128K token context but effective performance degrades above ~10-32K tokens depending on the task. Heavily subject to lost-in-the-middle effects.
- **Standard prompt engineering works.** Few-shot, chain-of-thought, and role prompting all effective. XML tags within message content can help organize information, though Llama doesn't natively favor them.
- **Local inference.** No cloud-side caching — prompt efficiency matters more for throughput since tokens are processed locally.

### 2.5 Granite (IBM)

**Structural preference:** Special token vocabulary (`<|start_of_role|>`, `<|end_of_role|>`, `<|end_of_text|>`).

**Key characteristics:**
- Supports multiple system turns within a single conversation (unusual).
- Tools supplied via the API are automatically formatted as a system prompt.
- Standard prompt engineering techniques apply.
- Fill-in-the-Middle (FIM) for code tasks via `<|fim_suffix|>` and `<|fim_middle|>` tokens.

### 2.6 Kimi (Moonshot AI)

**Structural preference:** OpenAI-compatible API format.

**Key characteristics:**
- Mixture-of-Experts architecture (1T total, 32B activated).
- Minimal system prompt sufficient for most tasks.
- 256K token context via API.
- Temperature 0.6, top_p 0.95 recommended.
- Strong tool-calling support; model autonomously decides when to invoke tools.

### 2.7 Cross-Model Comparison

| Dimension | Claude | GPT-4.1 | DeepSeek R1 | DeepSeek V3 | Llama 3/4 | Granite | Kimi |
|---|---|---|---|---|---|---|---|
| **Delimiter format** | XML tags | Markdown/XML | N/A (minimal) | Standard | Special tokens | Special tokens | OpenAI-compatible |
| **System prompt** | Scene-setting | Full instructions | **Avoid** | Standard | Via special tokens | Standard | Minimal |
| **Few-shot** | 3-5 recommended | Zero-shot often better | **Degrades quality** | Standard | Standard | Standard | Standard |
| **CoT prompting** | XML `<thinking>` tags | "Think step by step" | **Degrades quality** | Standard | Standard | Standard | Mode-dependent |
| **Prefill/completion** | Supported natively | Not supported | Not applicable | Prefix completion | End with assistant token | End with assistant token | Standard |
| **Cache mechanism** | Explicit markers (4 breakpoints) | Automatic prefix | Stable prefix | Stable prefix | N/A (local) | N/A | N/A |
| **Effective context** | ~200K | ~1M (100% needle-in-haystack) | 64K | 128K | 10-32K effective | Varies | 256K |

### 2.8 Section Ordering by Model

The core ordering (Section 1.2) works across all models, but the emphasis shifts:

- **Claude:** Maximize XML tag usage. Place examples in `<examples>` blocks in the middle. Use `cache_control` breakpoints to separate stable and volatile sections.
- **GPT-4.1:** Use markdown headers for structure. Repeat instructions at both beginning and end for long contexts. Omit few-shot examples unless they demonstrably help.
- **DeepSeek R1:** Flatten everything into a single user message. Remove system prompt, examples, and step-by-step instructions. State the problem directly and concisely.
- **Llama:** Keep contexts short (<10K tokens if possible). Use correct special tokens. Standard ordering works.

---

## 3. Token-Budget Triage

When token-constrained, cut in this order (first cut = lowest impact on quality):

### 3.1 Cut Priority (First to Last)

| Priority | What to Cut | Impact | Rationale |
|---|---|---|---|
| 1 (cut first) | Verbose explanations in instructions | Minimal | LLMs parse structured instructions efficiently; prose wastes tokens |
| 2 | Transitive dependency signatures | Low | Rarely consulted unless the task directly involves those APIs |
| 3 | Few-shot examples (except for DeepSeek R1, which never uses them) | Low-Medium | Zero-shot with clear format instructions often suffices, especially for GPT-4.1+ |
| 4 | Full repository map | Medium | Replace with a focused map of only the relevant subtree |
| 5 | Direct dependency file contents | Medium | Replace with interface signatures (function signatures + docstrings) |
| 6 | Background knowledge / playbooks | Medium-High | Summarize to key points rather than removing entirely |
| 7 | Target file contents | High | **Degrade gracefully**: switch from full contents to relevant excerpts around the modification points |
| 8 (cut last) | Task instruction + constraints + output format | **Do not cut** | These are the core of the document — without them, the generation has no direction |

### 3.2 Graceful Degradation Strategy

Rather than binary include/exclude, use progressive representation levels:

```
FULL           → Complete file contents (highest token cost)
SIGNATURES     → Function/class signatures + docstrings
NAMES_ONLY     → List of exported symbols
OMITTED        → Not included; LLM can request via exploration
```

Forge implements this in its token budget packing: when a context item at `FULL` representation doesn't fit, it's automatically downgraded to `SIGNATURES` before being dropped entirely.

### 3.3 The Exploration Fallback

If the system supports LLM-guided context exploration (Forge's Phase 7), aggressive token-budget triage becomes less risky: anything cut from the initial context can be requested on-demand by the LLM in a follow-up round. The initial document should include enough context for the LLM to know what it *could* ask for (e.g., the repo map showing what files exist) even if the file contents are omitted.

---

## 4. Single-Pass vs. Multi-Pass Generation

### 4.1 When Single-Pass Is Sufficient

A single document completion call is sufficient when:

- The output is **<2,000 words** (below the training-data length ceiling for most models).
- The task is **well-defined and narrow** (one file, one function, one focused question).
- The context is **complete** — the LLM has everything it needs to produce the final answer.
- The output format is **structured** (JSON, code edits, structured fields) rather than free-form prose.

In these cases, adding a planning preamble to the prompt (e.g., "First, list the key points you want to make, then write the response") is usually sufficient to guide generation quality without needing a separate planning call.

### 4.2 When Multi-Pass Is Better

Multi-pass generation is worth the additional API calls when:

- The output exceeds **~2,000 words** (LLMs struggle to maintain coherence in single-pass long generation).
- The task is **complex or ambiguous** — planning clarifies the approach before committing to generation.
- The output requires **cross-section coherence** (e.g., a report where section 3 must be consistent with section 1).
- **Quality justifies the cost** — multi-pass adds 2-8x the API calls but measurably improves output.

### 4.3 In-Document Planning ("Think First")

The simplest multi-pass technique is not multi-pass at all: instruct the LLM to generate a plan or outline at the beginning of its response, then use that plan to guide the rest of the generation. This works within a single document completion call.

**How it works:**
```
System: "Before generating your response, first produce a brief plan
listing the key points and structure. Then follow your plan to generate
the full response."
```

**Strengths:**
- No additional API calls.
- The plan tokens act as a chain-of-thought, steering later generation.
- Works with all models (including reasoning models, where the plan happens in the thinking trace).

**Limitations:**
- The plan and the response share the same output token budget.
- No opportunity for external validation of the plan before generation proceeds.
- For very long outputs, the plan itself may fall into the "middle" of the context and lose influence.

**Recommendation:** Use in-document planning as the default for outputs under ~2,000 words. For longer outputs, use explicit multi-pass.

### 4.4 Explicit Multi-Pass: Outline → Expand → Refine

For documents exceeding ~2,000 words or requiring high quality, decompose generation into discrete batch calls:

**Pass 1: Plan** (1 batch call)
```
Input:  Task description + source materials + constraints
Output: Structured outline with section headers, descriptions,
        and target word counts per section
Model:  Use the highest-quality model here — plan quality
        bounds everything downstream
```

**Pass 2: Expand** (N batch calls, parallelizable)
```
For each section in the outline:
  Input:  Full outline (global coherence)
        + Running summary of prior sections (Chain of Density)
        + Section-specific instructions + word count target
  Output: Section content (200-500 words per section)

Sections without dependencies fan out in parallel.
Dependent sections execute sequentially, each receiving
a running summary of previously generated content.
```

**Pass 3: Assemble** (deterministic, no LLM call)
```
Concatenate expanded sections in outline order.
Validate structure matches the outline.
```

**Pass 4: Refine** (1-2 batch calls)
```
Round A: Generate structured feedback on the assembled document.
Round B: Revise the document given original instructions + draft + feedback.
Cap at 2-4 iterations (diminishing returns per Self-Refine research).
```

**Key insight from the AgentWrite research (ICLR 2025):** The plan should be internal scaffolding, not part of the final output. Ablation studies showed that including the plan in the final document slightly improved length but *decreased quality*.

### 4.5 Maintaining Coherence Across Sections

When expanding sections independently (especially in parallel), cross-section coherence is the primary risk. Three techniques mitigate this:

1. **Full outline in every expansion call.** Each section-expansion prompt includes the complete outline so the LLM knows the global structure and can position its section appropriately.

2. **Chain of Density running summary.** For sequential expansion, maintain a progressively condensed summary of all previously generated sections. Inject this summary (not the full prior text) into each subsequent section's prompt. This avoids the lost-in-the-middle problem that would arise from injecting all prior text.

3. **Cross-reference instructions.** Explicitly tell the LLM how the current section relates to its neighbors: "This section follows [Section 2: Authentication] and precedes [Section 4: Authorization]. Ensure terminology is consistent."

---

## 5. Output Structuring for Quality

### 5.1 Structured Output Schemas

Constrained decoding (forcing the LLM to produce output matching a JSON schema) achieves 100% format compliance. OpenAI reports going from 35.9% reliability with prompt-only approaches to 100% with strict schema mode.

**Even when using schema enforcement, include natural-language descriptions of what each field should contain.** Constrained decoding ensures format compliance but doesn't ensure field *quality*. Telling the model "the `explanation` field should describe the rationale for each change, not just list the files modified" consistently improves the content within the schema.

### 5.2 The "List Points First" Technique

Asking the LLM to list its key points before generating the full response is a form of in-document chain-of-thought that improves coherence and coverage:

```
Instruction: "Before writing your analysis, list the 5-8 key points
you will cover, ordered by importance. Then write the full analysis
following this structure."
```

**Why this works:** The list of points functions as a self-generated outline. Because autoregressive generation conditions on all prior tokens, the listed points constrain and guide the generation that follows. Points listed early have the strongest influence (primacy within the generation itself), so ordering them by importance front-loads the most critical material.

**When to use:** Any free-form generation task where completeness and coherence matter. Not needed for structured output (JSON, code edits) where the schema already constrains the output.

### 5.3 Tool-Based Output Enforcement

For batch processing systems, using the API's tool/function-calling mechanism to structure output is more reliable than prompt-only instructions:

```python
# Anthropic: tool_use with Pydantic schema
params = {
    "tools": [{"name": "response", "input_schema": schema}],
    "tool_choice": {"type": "tool", "name": "response"},
}
```

This forces the LLM to produce a structured tool call matching the schema. Combined with Pydantic validation on the receiving end, it eliminates parsing failures and ensures every batch response is machine-readable.

### 5.4 Diff-Based Output for Code

For code generation tasks, requesting search/replace edits rather than full file replacement has multiple advantages:

- **Token efficiency.** The LLM outputs only the changed portions, not unchanged lines.
- **Precision.** The search string must match exactly once, catching hallucinated edits.
- **Safety.** Prevents silent code destruction when files are too large for the context window.
- **Verifiability.** Each edit is independently inspectable and reversible.

The tradeoff: the LLM needs current file contents in the prompt to produce accurate search strings. This means target file contents should be high-priority context items that are cut last during token-budget triage (Section 3).

---

## 6. Advanced Techniques

### 6.1 Skeleton-of-Thought (SoT)

A two-phase technique from Microsoft Research (ICLR 2024) that maps directly to batch fan-out/gather:

**Phase 1: Skeleton.** The LLM generates a concise outline of 3-10 points, each described in 3-5 words.

**Phase 2: Parallel expansion.** Each skeleton point is expanded independently via parallel batch calls. Each expansion receives the original question, the full skeleton, and the specific point to expand.

**Results:** 2x speedup with quality improvements in 60%+ of cases. Quality improves because explicit planning forces better structure.

**Limitations:** Not suitable for sequential reasoning tasks (math, step-by-step logic). Points are expanded independently, so cross-point dependencies are not handled. Use Chain of Density summaries (Section 4.5) to mitigate.

**Batch implementation:**
```
Step 1: [1 batch call]   → Generate skeleton (3-10 points)
Step 2: [N batch calls]  → Expand each point in parallel
Step 3: [Deterministic]  → Concatenate in skeleton order
```

### 6.2 Self-Refine (NeurIPS 2023)

A single LLM serves as generator, critic, and refiner in a closed loop:

1. **Generate** initial output.
2. **Critique:** Same LLM provides structured feedback — localize problems, explain what's wrong, suggest fixes.
3. **Refine:** Same LLM revises the output given original prompt + previous output + feedback.
4. **Repeat** up to 4 iterations (diminishing returns beyond this).

**Results:** Up to 49.2% absolute improvement over single-pass. Even frontier models (GPT-4, Claude Opus) benefit from self-refinement.

**Stopping criteria:** Fixed iteration count (4 recommended) OR terminate when feedback contains no actionable items.

**Batch implementation:**
```
Round 1: [Batch call] Generate initial draft
Round 2: [Batch call] Generate structured feedback on draft
Round 3: [Batch call] Revise draft given feedback
Round 4: [Batch call] (Optional) Second feedback + revision cycle
```

Each round is a stateless document completion — fully batch-compatible.

### 6.3 Cross-Refine (Two-Model Variant)

Use a different (possibly cheaper or more critical) model for the feedback step. This enables effective refinement even when the generator model has weaker self-evaluation capabilities. Relevant for multi-model orchestrators that use cheaper models for generation and frontier models for quality control.

### 6.4 Self-Consistency via Fan-Out

Generate N candidate outputs (N=3-5) at higher temperature, then select the best one:

- **For structured answers:** Majority voting.
- **For free-form text:** LLM judge selects the best candidate or synthesizes the strongest elements from each.

**Batch implementation:**
```
Step 1: [N parallel batch calls] Generate N candidates
Step 2: [1 batch call] LLM judge selects/synthesizes best
```

The Self-Certainty variant (2025) achieves the same accuracy as full self-consistency with 46% fewer samples by using the LLM's output probability distribution to estimate quality.

### 6.5 RecurrentGPT: Simulated Memory for Unbounded Generation

For arbitrarily long generation, maintain a structured state object (plan, running summary, entity state) that is updated after each generation step and injected into the next step's context. Each step is a standalone document completion call:

```
Step N prompt:
  - Original task description
  - Current plan state
  - Running summary of content generated so far (condensed)
  - Entity/state tracker (key facts to maintain consistency)
  - Instruction: generate the next section and update the state

Step N output:
  - Generated section content
  - Updated plan state
  - Updated running summary
  - Updated entity tracker
```

This pattern enables coherent generation across documents of any length, with each step remaining within the model's effective context window.

### 6.6 Query-Aware Contextualization

Place the task instruction both before and after the reference material. Research shows this dramatically improves information retrieval — all tested models achieved near-perfect performance with this pattern:

```
USER: "I need to refactor the authentication module to use JWT tokens."

<reference_material>
{8K+ tokens of code, documentation, etc.}
</reference_material>

"Given the reference material above, refactor the authentication module
to use JWT tokens. Specifically, modify handler.py and tokens.py."
```

### 6.7 Explicit Citation Requests

Ask the model to cite which source material it used. This forces attention to the reference material (mitigating lost-in-the-middle) and provides verifiability:

```
"For each change you make, cite the specific file and line number from
the reference material that motivated the change."
```

---

## 7. Composite Architecture

Based on all research above, here is a composite pattern optimized for a batch-first orchestration system with unlimited request budget:

### 7.1 Standard Quality (Most Tasks)

```
PHASE 1: GENERATE  [1 batch call]
  - In-document planning preamble ("List key points first, then respond")
  - Full context with cache-efficient ordering
  - Structured output via tool_use / JSON schema
  Total: 1 batch call

PHASE 2: VALIDATE  [Deterministic]
  - Schema validation, linting, tests
  - If pass → done
  - If fail → retry with error context (Phase 8 pattern)
  Total: 0-2 additional batch calls for retries
```

### 7.2 High Quality (Complex or Long-Form Tasks)

```
PHASE 1: PLAN  [1 batch call]
  - Use highest-quality model
  - Output: structured outline with section descriptions and word counts
  - Validate plan structure deterministically

PHASE 2: EXPAND  [N parallel batch calls]
  - One call per section
  - Each receives: full outline + Chain of Density summary + section instructions
  - Independent sections fan out in parallel
  - Dependent sections execute sequentially

PHASE 3: ASSEMBLE  [Deterministic]
  - Concatenate sections in outline order
  - Validate structure matches outline

PHASE 4: REFINE  [1-2 batch calls]
  - Generate structured feedback → revise → validate
  - Cap at 2 feedback-revision cycles

Total: 4-8 batch calls
```

### 7.3 Maximum Quality (High-Stakes Tasks)

```
PHASES 1-4: Same as High Quality (above)

PHASE 5: MULTI-CANDIDATE  [3x Phases 1-4]
  - Generate 3 complete candidates via Phases 1-4
  - Each candidate uses different temperature or prompt variation

PHASE 6: JUDGE  [1 batch call]
  - LLM judge evaluates all candidates on: accuracy, completeness,
    coherence, style
  - Selects best candidate or synthesizes strongest elements
  - Use bidirectional evaluation to control for position bias

Total: 13-25 batch calls
Only justified when error cost is high relative to compute cost.
```

### 7.4 Key Design Principles

1. **Invest in planning.** Plan quality bounds everything downstream. Use the best model and highest token budget for planning.
2. **The plan is scaffolding, not output.** Do not include the plan in the final deliverable.
3. **Maintain coherence via running summaries.** Use Chain of Density summaries between sequential sections, not full prior text.
4. **Parallelize independent work.** Skeleton-of-Thought and fan-out/gather patterns yield speedups with quality improvements when sections are independent.
5. **Refinement has diminishing returns.** Cap at 2-4 iterations. The biggest improvement comes from the first feedback-revision cycle.
6. **Fresh context beats accumulated history.** Each batch call should receive a cleanly constructed context rather than an ever-growing message history. Research on multi-turn degradation shows LLMs lose coherence in extended conversations.
7. **Conservative context budgets.** Despite advertised context windows of 128K-1M tokens, effective performance degrades above 4K-32K tokens for most models. Curate context aggressively.
8. **Model-specific templates are necessary.** A single template cannot optimize for all models (Section 2).

---

## 8. Sources

### Academic Papers

| Paper | Year | Key Finding |
|---|---|---|
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) (Liu et al.) | 2023 | U-shaped attention; >30% degradation for middle-positioned information |
| [Serial Position Effects of LLMs](https://arxiv.org/abs/2406.15981) | 2025 (ACL) | Primacy/recency biases are widespread but model/task-dependent |
| [Skeleton-of-Thought](https://arxiv.org/abs/2307.15337) (Ning et al.) | 2024 (ICLR) | Two-phase outline-then-expand achieves 2x speedup with quality gains |
| [Self-Refine](https://arxiv.org/abs/2303.17651) (Madaan et al.) | 2023 (NeurIPS) | Iterative self-feedback improves output up to 49.2% |
| [LongWriter / AgentWrite](https://arxiv.org/abs/2408.07055) | 2025 (ICLR) | Two-stage plan-then-write produces coherent 20K-word outputs |
| [Re3: Recursive Reprompting](https://arxiv.org/abs/2210.06774) | 2022 (EMNLP) | Recursive context reconstruction for long-form generation |
| [Navigating the Path of Writing](https://arxiv.org/html/2404.13919v1) | 2024 | Outline-guided generation enhances text quality |
| [Self-Certainty](https://arxiv.org/pdf/2502.18581) | 2025 | 46% fewer samples for same self-consistency accuracy |
| [Demystifying Long Chain-of-Thought](https://arxiv.org/pdf/2502.03373) | 2025 | Extended reasoning chains exhibit emergent planning behaviors |
| [Found in the Middle](https://arxiv.org/pdf/2403.04797) | 2024 | Multi-scale PoE improves middle-position accuracy 20-40% |
| [Context Length Alone Hurts](https://arxiv.org/html/2510.05381v1) | 2025 | 17-20% degradation from input length alone |
| [Integrating Planning into Single-Turn Generation](https://arxiv.org/html/2410.06203v1) | 2024 | Auxiliary planning task improves single-pass long-form output |

### Official Documentation

- [Anthropic Claude 4 Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices)
- [Anthropic Prompt Caching](https://docs.claude.com/en/docs/build-with-claude/prompt-caching)
- [Anthropic Prompt Engineering Overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [Anthropic Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [OpenAI GPT-4.1 Prompting Guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Llama 3.1 Prompt Formats](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/)
- [Llama 4 Prompt Formats](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/)
- [DeepSeek R1 Prompting](https://docs.together.ai/docs/prompting-deepseek-r1)
- [Granite 4.0 Prompt Engineering Guide](https://github.com/ibm-granite/granite-4.0-language-models/blob/main/Granite%204.0%20Prompt%20engineering%20guide%20v2.md)
- [Kimi K2](https://github.com/MoonshotAI/Kimi-K2)

### Practitioner Guides

- [Scaling LLMs with Batch Processing](https://latitude.so/blog/scaling-llms-with-batch-processing-ultimate-guide) (Latitude)
- [Structured Outputs: Everything You Should Know](https://humanloop.com/blog/structured-outputs) (Humanloop)
- [How to Use LLMs for Coherent Long-Form Content](https://www.opencredo.com/blogs/how-to-use-llms-to-generate-coherent-long-form-content-using-hierarchical-expansion) (OpenCredo)
- [Solving Lost in the Middle](https://www.getmaxim.ai/articles/solving-the-lost-in-the-middle-problem-advanced-rag-techniques-for-long-context-llms/) (Maxim AI)
- [Microsoft Research: Skeleton-of-Thought](https://www.microsoft.com/en-us/research/blog/skeleton-of-thought-parallel-decoding-speeds-up-and-improves-llm-output/)
