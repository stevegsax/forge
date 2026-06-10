# Step 1: Classify

- **Parent step:** None (pipeline entry point)
- **Model tier:** CLASSIFICATION -> `claude-haiku-4-5-20251001`
- **LLM calls:** 1
- **PlanDAG version:** v1 (skeleton created)

## Action

Classify the user's request against the workflow catalog. The LLM
receives the raw request text plus descriptions of all available
workflow types (software, research, analysis) and returns the best
match with a confidence score.

## Decision

The request explicitly says "Write a python module," which maps
directly to the `software` workflow type. Confidence is high (0.95)
because the request is unambiguous — it asks to create code.

## Outcome

- `workflow_type: "software"`, `confidence: 0.95`
- A skeleton `PlanDAG` v1 is created with:
  - `plan_id: "plan-7a3f"`
  - A single root node (`node-root`, `is_leaf: false`)
  - No children, no edges
  - `workflow_type: "software"`
- The plan record is persisted to `plans.db` with `status: "draft"`

## Files

- [llm_prompt.md](llm_prompt.md) — classification prompt sent to the LLM
- [llm_result.md](llm_result.md) — LLM response with workflow type and confidence
