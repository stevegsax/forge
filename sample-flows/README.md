# Sample Message Flows

Concrete documentation of the message exchanges between Forge and the LLM. Each flow shows exactly what gets sent to and received from the Anthropic Messages API, step by step.

## Why these exist

Forge is a generic orchestrator — the same pipeline handles code generation, research, documentation, and other domains. These sample flows make the abstract pipeline concrete by tracing a single request from input through context assembly, API call, and response parsing.

## Conventions

- **Batch mode by default.** All flows document the batch API path (`messages.batches.create` + poller + signal) unless explicitly noted otherwise. This matches Forge's default (`sync_mode=False`).
- **Separate diagrams at signal boundaries.** When a stage is triggered by an asynchronous signal (e.g., the batch poller delivering a result), it gets its own sequence diagram. Each diagram should represent a contiguous, synchronous span of work.

## Directory layout

Each flow lives in a numbered subdirectory:

```
sample-flows/
├── README.md
├── Makefile
└── 01-say-hello/
    ├── README.md
    ├── step-01-assemble-context.md
    ├── step-02-call-llm.md
    ├── step-03-parse-response.md
    ├── sequence-submit.mmd
    └── sequence-retrieve.mmd
```

- `README.md` — flow overview with rendered sequence diagrams and links to each step
- `step-NN-<name>.md` — one file per pipeline stage, showing inputs and outputs
- `sequence-*.mmd` — Mermaid sequence diagram sources (rendered to SVG/PNG via `make`)

## Building diagrams

Requires [mermaid-cli](https://github.com/mermaid-js/mermaid-cli):

```bash
npm install -g @mermaid-js/mermaid-cli
```

Then from this directory:

```bash
make          # build all diagrams
make clean    # remove generated images
```

## Adding a new flow

1. Create a subdirectory: `NN-<short-name>/`
2. Add a `README.md` with an overview, rendered sequence diagrams, and links to each step
3. Add `step-NN-<stage>.md` files for each pipeline stage
4. Add `sequence-*.mmd` diagrams for each phase of the flow
5. Add the flow to the index below

## Flows

| # | Name | Description |
|---|------|-------------|
| 01 | [Say Hello](01-say-hello/) | Simplest possible flow: deliver a user request to the LLM and receive a structured response. No exploration, no validation, no commit. |
