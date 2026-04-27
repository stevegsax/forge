+++
title = "Context Assembly"
weight = 51
description = "How Forge decides what goes into the prompt: file discovery via import graphs, PageRank ranking, token budget packing, progressive disclosure, and the exploration loop."
topic = "context-assembly"
covers = [
    "How import graph analysis discovers relevant files without manual specification",
    "How PageRank ranks files by structural importance",
    "How the token budget packer selects what fits (knapsack-style priority packing)",
    "Progressive disclosure: why only target files and repo map are included by default",
    "How the exploration loop lets the LLM pull additional context on demand",
    "Why exploration rounds are separate document completions (not tool calls within a conversation)",
    "Tradeoffs: structural analysis over embeddings, signatures over full content, fixed ordering over adaptive ranking",
]
detail = "Context assembly is Forge's core competency for deciding what populates the prompt sections described in prompt-construction. Walk through the full assembly pipeline: discovery, ranking, packing, progressive disclosure, and the exploration loop. Explain the tradeoffs. Do NOT re-describe the 11-section structure or cache ordering — that is prompt-construction's topic. The dividing line: prompt construction describes the destination; context assembly describes the journey."
+++
Prerequisites: [Prompt Construction](prompt-construction/).

Context assembly is Forge's answer to the question: given a task, what should the LLM see? The system prompt's eleven sections are described in [Prompt Construction](prompt-construction/) — the ordering, cache breakpoints, and error injection format. This topic picks up from there: how does the system decide what content populates those sections? That process — discovering relevant files, ranking them by structural importance, packing them into a token budget, and letting the LLM pull additional context on demand — is context assembly.

For the data models, provider specifications, and priority ordering table, see [Context Assembly Reference](../reference/context-assembly/). For a concrete example of context assembly in action, see [The Golden Path](../tutorials/golden-path/).

## How Context Discovery Works

When a task specifies `target_files`, Forge discovers relevant context automatically. This replaces the manual burden of listing every file the LLM needs to see.

### Import Graph Analysis

Forge uses `grimp` to build the import dependency graph for the Python package. Given the target files, it traces imports to find:

- **Direct imports** -- files that the target files import. These are the immediate dependencies.
- **Transitive imports** -- files imported by the direct imports, up to a configurable depth (default: 2 levels).
- **Downstream dependents** -- files that import the target files. These are included in the repo map for structural awareness but not in the prompt by default.

The import graph handles the hard parts of Python import resolution: relative imports, namespace packages, `src/` layouts, `__init__.py` re-exports, and editable installs. Circular imports are handled gracefully -- each module appears at most once in traversal results.

### PageRank Ranking

Not all files in the import graph are equally useful for context. A utility module imported by 30 files is more structurally important than a leaf module imported by one. Forge runs PageRank (via `networkx`) on the file-level dependency graph to rank files by structural centrality.

The PageRank computation uses personalization: seed weights are placed on the target files so that files closer to the current task rank higher than files that are central to the codebase but unrelated to the task. The output is a ranked list of files with importance scores.

This approach is validated by production usage in Aider and by benchmark results showing that deterministic AST-derived dependency graphs achieve higher correctness scores than embedding-based retrieval, at a fraction of the cost.

### Symbol Extraction

For files that are too large to include in full, or that fall below the priority cutoff for full inclusion, Forge extracts the public interface using Python's `ast` module. This produces function signatures (with type annotations and docstrings), class definitions (with method signatures), type aliases, and constants -- without implementation bodies.

A 500-line module's extracted signatures might be 20 lines. This preserves the information the LLM needs to produce correct imports and type-compatible code, at a fraction of the token cost. Research on signature extraction shows 78-82% token reduction while preserving the information LLMs require.

## Token Budget Packing

Context assembly is a bin-packing problem. There is a fixed budget (default: 100,000 tokens, targeting approximately 50% of the model's 200k context window), and the goal is to pack the most useful information within that budget.

The packing algorithm follows a rationale-driven priority ordering rather than a mechanical one. The task description is always included first because it defines "done" -- without it, no other context has meaning, and nothing should crowd it out. The target files come next because they are the source material the LLM must read to produce correct output; the LLM cannot reason about changes to a file it has never seen. Direct import contents (when requested via `--include-deps`) follow, because the LLM needs to understand the contracts its changes must satisfy. Interface context -- the extracted signatures of transitively imported modules, ranked by PageRank -- comes next; signatures preserve type compatibility information at a fraction of the cost of full file content. The repo map and playbooks are lower priority because they provide structural orientation and accumulated lessons, which are helpful but not load-bearing. Manually specified `context_files` come last, reflecting the fact that the assembler cannot judge their relevance as well as it can judge import graph membership.

Items are packed in priority order. Within each priority tier, items are sorted by PageRank importance score (descending). When an item would exceed the budget, the algorithm attempts to reduce it (from full content to signatures only) before skipping it. Binary search determines how many tier-4 items fit in the remaining budget.

The target of 50-60% context utilization is deliberate. Research shows that model accuracy degrades as token volume increases, creating diminishing returns on additional context. Reserving headroom for the LLM's output and internal reasoning produces better results than filling the context window to capacity.

## Progressive Disclosure

By default, Forge assembles a lean prompt: only target file contents and the repo map are included upfront. Dependency contents -- even direct imports -- are omitted. This is the progressive disclosure strategy.

The rationale is that the LLM often does not need to see every dependency to produce correct output. For a task like "add error handling to the API client," the LLM needs the client file's current content and a structural overview of the codebase. If it needs the exception hierarchy or a utility function's signature, it can request that context through the exploration loop.

This keeps prompts small by default, reducing cost and improving accuracy (less irrelevant context means less noise for the model to filter). The `--include-deps` flag overrides progressive disclosure for tasks where the LLM is likely to need dependency context upfront.

## The Exploration Loop

The exploration loop is the mechanism that lets the LLM request additional context on demand. It runs before the generation call, not during it.

In a traditional agentic loop, the LLM calls tools mid-conversation: it generates a tool-call message, the system executes the tool, and the result is appended to the conversation history. Forge does not use this pattern. Each exploration round is a separate, complete document-completion request. The orchestrator manages the iteration.

The loop is driven by a lightweight LLM (classification tier) rather than the full generation model. This matters because exploration rounds are cheap -- the exploration LLM does not produce code, only context requests -- and because using a smaller model keeps the overall cost of the exploration phase well below the cost of the generation call itself.

Each round is a complete, standalone document completion. The exploration LLM receives the task description, target files, whatever context has accumulated so far, and a menu of available context providers. It responds with a list of requests -- read this file, search for this symbol, extract signatures from this module -- which the orchestrator dispatches to provider handlers. Providers are intentionally thin: they do I/O (reading files, running searches, querying the import graph) and return results, nothing more. The results are appended to the accumulated context, and the next round begins.

The loop ends when the exploration LLM signals it has enough context (an empty request list) or when the round limit is reached. At that point, all accumulated results are assembled into the Exploration Results section of the system prompt, and the actual generation call begins.

The reason this design uses separate document completions rather than tool calls within a single conversation is batch compatibility. Tool calls require persistent conversation state: the model generates a partial response, the system executes the tool, and the result is injected back into the same conversation turn. Batch APIs do not support this pattern -- each request must be fully self-contained. By making each exploration round an independent request, Forge's exploration mechanism works identically in both synchronous and batch execution paths. The orchestrator, not the model, manages iteration state. This is the architectural payoff of the "orchestrator manages the loop" pattern: the LLM call remains a pure, stateless document completion regardless of how many exploration rounds precede it.

Exploration is also used before the planner call. The planner benefits from exploring the codebase before decomposing a task into steps.

## Tradeoffs and Alternatives

Forge's context assembly makes several deliberate tradeoffs:

- **Structural analysis over embeddings.** Import graph analysis and PageRank are deterministic: the same target files always produce the same ranked file list. Embedding-based retrieval (as used by some competing tools) can surface semantically related files that are not reachable through imports, but it requires infrastructure (vector stores, embedding models) and is non-deterministic. Sourcegraph Cody deprecated their embeddings-based retrieval in favor of native code search, finding it performed better without the overhead.

- **Signatures over full content.** Including extracted signatures instead of full file content loses implementation details but gains token efficiency. This is the right tradeoff when the LLM needs to produce type-compatible code but does not need to understand the implementation. When it does need implementation details, the exploration loop provides them on demand.

- **Fixed priority ordering over adaptive ranking.** The priority tiers (target files > dependencies > interfaces > repo map > manual files) are fixed. An adaptive system could re-rank based on the specific task, but fixed ordering is simpler, predictable, and cache-friendly. The exploration loop provides the adaptive component.

For how the assembled content is structured within the system prompt — section ordering, cache breakpoints, and error injection — see [Prompt Construction](prompt-construction/). For the complete context pipeline as it relates to the broader workflow step pattern, see [The Universal Workflow Step](workflow-step/). For practical guidance on controlling context assembly, see [How to Control Context Assembly](../howto/control-context-assembly/).
