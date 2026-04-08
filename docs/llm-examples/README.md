# Forge - Worked LLM Examples

This directory contains worked examples that trace Forge's workflow
from start to finish. Each example demonstrates how a real task moves
through the universal workflow step — context assembly, LLM call,
result serialization, and state transition — so readers can see the
system in action rather than just reading about it in design docs.

## Structure

Each example is self-contained and organized as a sequence of numbered
step directories:

```text
example-name/
  README.md              # Overview: what the example demonstrates
  templates/             # Prompt templates used by this example
  step_0001/
    README.md            # What happened, decisions made, parent step
    llm_prompt.md        # The full prompt sent to the LLM (if applicable)
    llm_result.md        # The raw LLM response (if applicable)
  step_0002/
    README.md
    llm_prompt.md
    llm_result.md
  ...
```

### Step directories

Each `step_NNNN/` directory captures a single workflow step. Its
`README.md` describes:

- **Parent step** — the step ID that invoked this step (e.g.,
  `step_0001`). The first step has no parent. In a fan-out, multiple
  steps share the same parent — for example, `step_0003`,
  `step_0004`, and `step_0005` might all list `step_0002` as their
  parent when the planner fans work out to parallel sub-tasks.
- **What was performed** — the action taken (context assembly, LLM
  call, validation, file write, state transition, etc.).
- **Decisions made** — any choices the orchestrator or planner made
  and the reasoning behind them.
- **Outcome** — what the step produced and how it influenced the next
  step.

Steps that involve an LLM call also include:

- **`llm_prompt.md`** — the fully assembled prompt (system message,
  context, and user message) exactly as sent to the model.
- **`llm_result.md`** — the model's raw response before any
  post-processing.

Steps that do not involve an LLM call (e.g., deterministic context
assembly, validation, file writes) contain only the `README.md`.

### Templates directory

Each example includes a `templates/` directory with the prompt
templates used during that example. These are the source templates
before variable substitution — compare them with the fully rendered
prompts in `llm_prompt.md` to see how context is injected.

## How to read an example

1. Start with the example's top-level `README.md` for an overview of
   the task and what the example demonstrates.
2. Walk through the step directories in order. Each step builds on
   the previous one.
3. For LLM steps, read the prompt to understand what context was
   assembled and why, then read the result to see how the model
   responded.
4. Pay attention to the decision rationale in each step's README —
   this is where the "why" of the orchestrator's behavior is
   documented.
5. For fan-out examples, note that sibling steps share a parent and
   execute in parallel — read them as independent branches that
   converge at a gather step.
