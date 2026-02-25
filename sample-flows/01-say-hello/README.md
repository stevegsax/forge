# 01 — Say Hello

The simplest possible Forge flow: deliver a user request to the LLM and receive a structured response. No exploration rounds, no validation, no commit. This validates the basic message delivery mechanism before layering on additional pipeline stages.

## Submit

Assemble context, then submit the request to the Anthropic Message Batches API. The workflow suspends after submission.

![Submit sequence](sequence-submit.svg)

## Retrieve

The batch poller detects completion, delivers the raw response via Temporal signal, and the workflow resumes to parse it.

![Retrieve sequence](sequence-retrieve.svg)

## Steps

1. [Assemble context](step-01-assemble-context.md) — build the system prompt and user prompt from the task definition and domain config
2. [Submit batch request](step-02-call-llm.md) — submit to the Anthropic Message Batches API; workflow suspends until the poller delivers a result
3. [Parse response](step-03-parse-response.md) — deserialize the raw response JSON, extract the `LLMResponse` from the `tool_use` block, and collect usage statistics

## What this flow skips

- **Exploration** — no `--explore` rounds; the LLM generates immediately
- **Auto-discovery** — no target files, so no import graph analysis or repo map
- **Validation** — no ruff lint/format checks, no tests
- **Write / commit** — the result is returned but not written to a worktree
