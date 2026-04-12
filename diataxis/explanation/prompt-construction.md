# Prompt Construction

Prerequisites: [The Universal Workflow Step](workflow-step.md).

The system prompt is the most consequential artifact Forge produces. Every LLM call in Forge is a single, self-contained document completion — there is no chat history, no multi-turn conversation, no persistent memory between calls. The entire world the LLM knows about must be packed into one prompt. The structure of that prompt directly bounds the quality of the output: if a relevant file is missing, the LLM hallucinates; if the token budget is wasted on irrelevant content, the signal-to-noise ratio drops; if the ordering is wrong, caching fails and every call costs twice as much.

This document covers the structure of the system prompt — its eleven sections, their ordering, and why the ordering is what it is. It does not cover how Forge decides which files and content populate those sections; that process is [Context Assembly](context-assembly.md), which picks up where this topic leaves off. For the authoritative section table, see the [Prompt Construction Reference](../reference/prompt-construction.md).

## The prompt as a structured document

The prompt sent to the LLM is not a chat transcript. It is a structured document assembled from multiple sources, each occupying a defined section within the system prompt. The system prompt carries all substantive context; the user prompt is intentionally minimal — a short domain-specific instruction like "Generate the requested code changes."

This design follows from Forge's batch-first principle. Batch APIs do not support multi-turn conversations. Every request must be self-contained. By packing all context into the system prompt, Forge ensures that the same prompt works whether it is sent synchronously or queued for batch processing. The LLM receives one complete document, produces one structured response, and the orchestrator evaluates the result. That is the entire interaction.

## The eleven sections

The system prompt has eleven sections, assembled in a specific order. That order is not arbitrary — it is optimized for Anthropic's prompt caching, which works on prefixes: if two requests share the same prefix, the cached portion is not re-processed, saving up to 80% on input token costs for the cached content.

The ordering principle is simple: content that never changes goes first; content that changes on every call goes last.

**Sections 1–2: Role statement and output requirements.** These come from the domain configuration (`DomainConfig.role_prompt` and `DomainConfig.output_requirements`). They are identical for every call in a given domain — a code-generation task and a test-writing task that share the same domain produce the same role and output sections. They anchor the very beginning of every prompt and are never displaced. The first cache breakpoint is placed after section 2, meaning every call to the same domain shares this cached prefix.

**Sections 3–5: Project instructions, repository structure, and playbooks.** Project instructions come from `CLAUDE.md` in the repository root. The repository structure is a PageRank-ranked file tree produced by `code_intel/repo_map.py`. Playbooks are lessons from forge's own playbook store, retrieved by tag overlap with the current task. All three are stable for the lifetime of a repository (or at least for the duration of a work session). They form the second tier of cached content.

**Sections 6–7: Task description and target file contents.** The task description and target file list define what "done" means. They are fixed for the life of a task but differ between tasks. The actual contents of the target files come next — they are stable across exploration rounds but may change between retries (a failed attempt may have partially modified a file). The second cache breakpoint is placed after section 7, so retries within the same step share the cached prefix through the target file contents.

**Sections 8–9: Dependencies and interface context.** Direct dependency file contents (when `--include-deps` is enabled) and extracted interface signatures from transitive imports. These are stable within a task and build outward from the task center. For how these are discovered and ranked, see [Context Assembly](context-assembly.md).

**Section 10: Exploration results.** The accumulated responses from the exploration loop — files the LLM requested, symbol lists, search results. These grow round by round during exploration and then stabilize once exploration completes. The third cache breakpoint is placed after section 10, so retries benefit from cached exploration context.

**Section 11: Previous errors.** Only present on retry. Contains the structured error output from the failed attempt (lint errors, test failures) plus AST-derived code context showing the enclosing function and the exact line that caused each error. This section is the most volatile content in the prompt — it changes on every retry attempt — and is placed last so that everything preceding it remains cache-eligible.

## Why this ordering works for caching

The three cache breakpoints create a hierarchy. On a first attempt at a task's first step, nothing is cached and the full prompt is processed. On the second step of the same task, sections 1–5 are served from cache. On a retry within the same step, sections 1–7 (and often 1–10) are served from cache — the retry pays only for the new error section.

The practical consequence is that a task that takes three steps with one retry each pays full cost once (the first step's first attempt) and gets cache-rate pricing for five subsequent calls. For a planned task that executes multiple steps, the cache savings compound: the stable prefix (role, output, project instructions, repo structure, playbooks) is processed once and reused across every step and every retry.

Observed cache hit rates in production usage are typically 70–85% of input tokens across a multi-step planned task. The retry in the golden-path tutorial shows this directly: attempt 2 takes 4.9 seconds where attempt 1 took 7.8 seconds, because 8,689 of 9,536 input tokens were served from cache.

## Error injection on retry

When validation fails and a step is retried, the orchestrator does not retry blind. It builds an error section that is appended as section 11 of the system prompt. The error section includes:

**Structured error output** — the raw `ruff` lint errors, format violations, or test failure output, exactly as the validation activity produced them.

**AST-derived code context** — for each error with a file path and line number, Forge parses the source file with Python's `ast` module to find the enclosing function or class. It then includes a code snippet showing the scope header and the error line (marked with `# <-- ERROR`), so the LLM has immediate visual context around each failure without needing to see the entire file again.

This approach follows research from the SWE-bench community showing that agents with error feedback significantly outperform agents that retry without context. The AST-contextualized format is similar to Aider's approach (which uses tree-sitter for the same purpose).

The error section is placed last because it is the most volatile content in the prompt — it changes on every retry attempt. If error context were injected mid-prompt, every retry would break the cache for everything that follows. By placing errors last, the entire preceding prefix remains cache-eligible, so a retry pays only for the new error section rather than re-processing the whole prompt.

## What this means for downstream topics

Prompt construction describes the destination — the structure of what the LLM sees. [Context Assembly](context-assembly.md) describes the journey — how the system decides what content populates sections 3–10. The two topics are tightly coupled but distinct: you can understand the prompt structure without knowing how import graph analysis or PageRank ranking work, and you can understand the discovery algorithms without knowing the cache breakpoint placement. Both contribute to prompt quality, but from opposite ends.

For a concrete end-to-end example of prompt construction in action, see [The Golden Path](../tutorials/golden-path.md). For the section table and cache breakpoint reference, see [Prompt Construction Reference](../reference/prompt-construction.md).
