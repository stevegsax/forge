# Planning Prompts in AI Coding Tools: Research Report

## Motivation

Forge's planner currently uses a mechanical prompt ("You are a task decomposition assistant") with Sonnet, a tiny repo map (2048 tokens), and no reasoning guidance. This report surveys how leading open-source AI coding tools approach planning and task decomposition, with the goal of identifying concrete improvements for Forge's planner prompt.

---

## 1. Claude Code

**Source**: [Piebald-AI/claude-code-system-prompts](https://github.com/Piebald-AI/claude-code-system-prompts) (v2.1.38, Feb 2026)

### Architecture

Claude Code has **three distinct plan mode variants**, selected based on configuration:

1. **5-Phase Plan Mode** -- the most structured variant
2. **Iterative Plan Mode** -- collaborative pair-programming style
3. **Subagent Plan Mode** -- simplified variant for delegated planning

The model self-selects when to plan via the `EnterPlanMode` tool, with explicit criteria:

- New feature implementation
- Multiple valid approaches exist
- Changes affect existing behavior
- Architectural decisions required
- Multi-file changes (2-3+ files)
- Unclear requirements needing exploration
- User preferences matter

Simple tasks (typos, single-function additions, tasks with detailed instructions) skip planning.

### The 5-Phase Plan Mode

```
Phase 1: Initial Understanding
  - Explore codebase using up to N parallel Explore subagents
  - Search for existing functions/utilities to reuse
  - Ask clarifying questions

Phase 2: Design
  - Launch Plan subagents with "architect" persona
  - Multiple perspectives for complex tasks (simplicity vs performance vs maintainability)
  - Default: at least 1 Plan agent; skip for trivial tasks

Phase 3: Review
  - Read critical files identified by agents
  - Ensure alignment with user's original request
  - Clarify remaining ambiguities

Phase 4: Final Plan
  - Context section explaining why the change is made
  - Only the recommended approach (not all alternatives)
  - Critical file paths and existing functions to reuse
  - Verification section for end-to-end testing

Phase 5: Exit Plan Mode
  - Call ExitPlanMode for user approval
```

### The Plan Agent Subagent

The Plan agent receives an explicit persona:

> "You are a software architect and planning specialist for Claude Code."

It operates in **strictly read-only mode** and must end with a "Critical Files for Implementation" section listing 3-5 files with reasons.

### The Iterative Plan Mode

A lighter-weight loop:

```
Repeat until done:
  1. Explore (read-only tools)
  2. Update the plan file with discoveries
  3. Ask the user when hitting ambiguities
```

Key instruction: "Never ask what you could find out by reading the code."

### Key Lessons

- **Separation of exploration and design** into distinct subagents with different prompts
- **Multiple perspectives** for complex tasks (e.g., bug fix: root cause vs workaround vs prevention)
- **Plan is a file**, not in-memory -- persists across sessions
- **Read-only constraint** during planning prevents premature execution
- **Explicit reuse mandate**: "Actively search for existing functions, utilities, and patterns that can be reused"

---

## 2. OpenAI Codex CLI

**Source**: [openai/openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/gpt-5/codex_prompting_guide.ipynb), [openai/codex](https://github.com/openai/codex)

### Architecture

Codex uses a single agent loop with an `update_plan` tool rather than a separate planning phase. The system prompt opens with:

> "You are Codex, based on GPT-5. You are running as a coding agent in the Codex CLI on a user's computer."

### Planning via the `update_plan` Tool

Planning is integrated into the execution flow rather than being a separate mode:

```json
{
  "name": "update_plan",
  "parameters": {
    "explanation": "string - reasoning for the update",
    "plan": [
      { "step": "string - description", "status": "pending|in_progress|completed" }
    ]
  }
}
```

Constraints:

- At most one step can be `in_progress` at a time
- Skip planning for straightforward tasks
- Update the plan after completing subtasks
- Reconcile all TODOs before finishing
- Avoid committing to future work not done immediately

### Autonomy Guidance

The key planning-relevant instruction:

> "Once the user gives a direction, proactively gather context, plan, implement, test, and refine without waiting for additional prompts."

### File Exploration Workflow

```
1. Think first about ALL the files you will need
2. Batch all file reads together
3. Use multi_tool_use.parallel for parallelization
4. Only use sequential calls when logically unavoidable
```

### Custom Instructions (AGENTS.md)

Codex reads `AGENTS.md` files in a hierarchical chain:

1. Global: `~/.codex/AGENTS.override.md` then `~/.codex/AGENTS.md`
2. Project: Git root down to CWD, each level checked
3. Later files override earlier ones
4. Combined cap: 32 KiB

### Key Lessons

- **Plan as a tool, not a phase** -- the LLM decides when to create/update the plan inline
- **Lightweight planning** -- just a list of steps with statuses, no separate planner model
- **Bias for action** -- plan only when needed, execute immediately otherwise
- **Parallel file reads** -- batch context gathering upfront

---

## 3. Google Gemini CLI

**Source**: [google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli), [Gemini CLI System Prompt Gists](https://gist.github.com/ksprashu/61194be375dba10d8950df43e33742fb)

### Architecture

Gemini CLI has the most layered approach: (1) a primary workflow embedded in the system prompt, (2) an explicit Plan Mode, (3) a `write_todos` tool, and (4) sub-agents for delegated research.

### Primary Workflow: Research -> Strategy -> Execution

All development work follows this three-phase lifecycle:

**Phase 1: Research**
> "Systematically map the codebase and validate assumptions. Use `grep` and `glob` search tools extensively (in parallel if independent)."

If the Codebase Investigator sub-agent is enabled:
> "Utilize specialized sub-agents (e.g., `codebase_investigator`) as the primary mechanism for initial discovery when the task involves complex refactoring, codebase exploration or system-wide analysis."

**Phase 2: Strategy**
> "Formulate a grounded plan based on your research. Share a concise summary of your strategy."

**Phase 3: Execution** (iterative per sub-task)
```
For each sub-task:
  - Plan: Define specific implementation approach and testing strategy
  - Act: Apply targeted, surgical changes
  - Validate: Run tests and workspace standards
```

The validation mandate is strongly emphasized:
> "Validation is the only path to finality. Never assume success or settle for unverified changes."

### Plan Mode (4 Phases)

When activated via `enter_plan_mode`, the system prompt is **entirely replaced** with planning-specific instructions:

```
Phase 1: Requirements Understanding
  - Analyze request, identify core requirements
  - Ask clarifying questions (prefer multiple-choice)
  - Do NOT explore the project yet

Phase 2: Project Exploration
  - Only after requirements are clear
  - Use read-only tools to explore the project
  - Identify existing patterns and conventions

Phase 3: Design & Planning
  - Only after exploration is complete
  - Plan MUST include iterative development steps,
    verification steps, file paths, function signatures

Phase 4: Review & Approval
  - Present the plan via exit_plan_mode
  - Address feedback and iterate if rejected
```

Plans are saved as markdown files in a dedicated plans directory. An approved plan is injected into the primary workflow's Strategy step:
> "An approved plan is available for this task. You MUST read this file before proceeding."

### The Codebase Investigator Sub-Agent

A specialized read-only sub-agent with:

- Temperature 0.1 (low creativity)
- Max 3 minutes / 10 turns
- **Scratchpad methodology**: maintains a dynamic scratchpad with checklist, unresolved questions, key findings, and dead ends
- Termination condition: "Your mission is complete ONLY when your Questions to Resolve list is empty"
- Structured JSON output: `SummaryOfFindings`, `ExplorationTrace`, `RelevantLocations`

### Context Compression

When history grows too large, Gemini compresses into a structured XML `<state_snapshot>`:

```xml
<state_snapshot>
    <overall_goal>...</overall_goal>
    <active_constraints>...</active_constraints>
    <key_knowledge>...</key_knowledge>
    <artifact_trail>...</artifact_trail>
    <file_system_state>...</file_system_state>
    <recent_actions>...</recent_actions>
    <task_state>
        1. [DONE] Map existing API endpoints
        2. [IN PROGRESS] Implement OAuth2 flow
        3. [TODO] Add unit tests
    </task_state>
</state_snapshot>
```

### Key Lessons

- **Requirements before exploration** -- Phase 1 of Plan Mode explicitly prohibits exploring the project until requirements are clear
- **Plan must include verification steps** -- not just "what to build" but "how to verify"
- **Scratchpad methodology** for the investigator -- tracking unresolved questions prevents premature convergence
- **Directive vs Inquiry distinction** -- assume requests are inquiries (analysis only) unless they contain explicit implementation instructions
- **Plan mode replaces the entire system prompt** rather than appending constraints

---

## 4. Aider

**Source**: [Aider-AI/aider](https://github.com/Aider-AI/aider), [aider.chat docs](https://aider.chat/docs/)

### Architecture

Aider does **not** have multi-step planning or task decomposition. Instead, it uses an **Architect/Editor split** -- a two-hop single-turn pattern where reasoning is separated from formatting.

### The Architect Prompt

```python
main_system = """Act as an expert architect engineer and provide direction
to your editor engineer.
Study the change request and the current code.
Describe how to modify the code to complete the request.
The editor engineer will rely solely on your instructions, so make them
unambiguous and complete.
Explain all needed code changes clearly and completely, but concisely.
Just show the changes needed.

DO NOT show the entire updated function/file/etc!
"""
```

The architect gets:

- No formatting rules (empty `system_reminder`)
- No few-shot examples
- Full repo map (tree-sitter signatures ranked by PageRank)
- Full contents of files in the chat

### The Editor Prompt

```python
main_system = """Act as an expert software developer who edits source code.
{final_reminders}
Describe each change with a *SEARCH/REPLACE block* per the examples below.
All changes to files must use this *SEARCH/REPLACE block* format.
ONLY EVER RETURN CODE IN A *SEARCH/REPLACE BLOCK*!
"""
```

The editor gets:

- No repo map (`map_tokens=0`)
- No conversation history
- The architect's entire response as the user message
- Detailed SEARCH/REPLACE formatting examples

### The Context Coder

A separate mode for automatic file discovery:

> "Act as an expert code analyst. Understand the user's question or request, solely to determine ALL the existing source files which will need to be modified."

Returns: (1) files to edit with relevant symbols, (2) symbols from other files needed for understanding.

### Behavioral Modifiers

Conditionally injected via `{final_reminders}`:

- **lazy_prompt**: "You are diligent and tireless! You NEVER leave comments describing code without implementing it!"
- **overeager_prompt**: "Pay careful attention to the scope of the user's request. Do what they ask, but no more."

These apply only to the editor, not the architect.

### Key Lessons

- **Separate reasoning from formatting** -- the architect focuses purely on "what to change" while the editor focuses on "how to express it as diffs"
- **Repo map with PageRank** -- tree-sitter signatures ranked by graph centrality (Forge already does similar)
- **No planning for multi-step work** -- each user message produces exactly one change; multi-step is human-driven
- **Model-specific behavioral modifiers** -- `lazy` and `overeager` flags per model configuration

---

## 5. Goose (Block)

**Source**: [block/goose](https://github.com/block/goose)

### Architecture

Goose is a Rust-based general-purpose AI agent with an extension/MCP-based tool system, sub-agent delegation, and a YAML recipe system for reusable workflows.

### System Prompt

> "You are a general-purpose AI agent called Goose, created by Block."

The prompt is dynamically assembled with available extensions and their tools, using prefix-based namespacing.

### Recipe System

Recipes are YAML-based workflow definitions:

```yaml
version: 1.0.0
title: "Recipe Name"
description: "What it does"
instructions: |
  Multi-line prompt with detailed instructions
  and Jinja2 conditional templating
activities:
  - "Phase 1 description"
  - "Phase 2 description"
extensions:
  - type: builtin
    name: developer
parameters:
  test_depth: "standard"
```

Recipes support:

- **Parameter substitution** for dynamic workflows
- **Conditional sections** via Jinja2 (`{% if ... %}`)
- **Cron scheduling** for automated execution
- **Max turns** constraints on execution

### Sub-Agent System

Sub-agents operate with restricted capabilities (cannot spawn further sub-agents). Each has:

- Its own system prompt and conversation history
- Separate session persistence
- Constrained tool access

### Key Lessons

- **Recipes as reusable task templates** -- pre-defined workflows with parameters
- **Sub-agent recursion prevention** -- one level of delegation only
- **Context compaction** monitors token usage and compresses history

---

## 6. BMAD Method

**Source**: [bmad-method](https://github.com/bmad-method/bmad-method)

### Architecture

BMAD (Business/Market Analyst and Developer) is a comprehensive workflow orchestration system using persona-based agents with sequential document-driven planning. It is not a coding tool per se but a methodology for LLM-driven software development.

### Planning Workflow

BMAD defines a strict sequence of document generation:

1. **Business Analyst** creates a PRD (Product Requirements Document)
2. **Architect** creates a technical architecture document
3. **Product Manager** creates epics and stories from the PRD + architecture
4. **Developer Agent** implements from the stories

Each agent has a rich persona and operates within defined checklists.

### Story Quality Competition Checklist

BMAD includes a "quality competition" framework with five validation gates:

1. **Load and Understand** -- extract metadata, resolve variables
2. **Exhaustive Source Analysis** -- epics, architecture, previous stories, git history
3. **Disaster Prevention Gap Analysis** -- reinvention, wrong tech, structure violations, regressions, vague specs
4. **LLM-Dev-Agent Optimization** -- clarity, actionability, scannability, token efficiency, unambiguity
5. **Improvement Recommendations** -- categorized as Critical/Enhancement/Optimization

### Key Lessons

- **Document-driven planning** -- each phase produces a persistent artifact that feeds the next
- **Disaster prevention checklist** -- explicitly checking for common failure modes before implementation
- **LLM optimization of prompts** -- a meta-step to ensure instructions are clear and token-efficient for the implementing agent

---

## 7. OpenSpec

**Source**: [OpenSpec project](https://github.com/openspec-dev/openspec)

### Architecture

OpenSpec structures planning through a "proposal" phase that generates three artifacts:

1. **proposal.md** -- motivation, scope, expected impact (simplified PRD)
2. **tasks.md** -- atomic implementation steps ("AI's action guide for itself")
3. **design.md** (optional) -- technical decisions (library choices, indexing strategies)

### Task Decomposition Strategy

Tasks follow an atomic principle:

> "This breakdown is crucial -- it allows AI to 'check off' each completed step during subsequent implementation, maintaining progress control."

### Prompt Injection System

OpenSpec uses `AGENTS.md` with `<openspec-instructions>` XML blocks to force AI agents into its planning workflow when they detect planning-related keywords.

### Key Lessons

- **Three-artifact output** -- separating "why" (proposal), "what" (tasks), and "how" (design) into distinct documents
- **Atomic task decomposition** -- indivisible units that can be tracked and verified independently
- **Convention-based agent steering** -- using AGENTS.md to redirect any AI tool into the planning workflow

---

## 8. Beads (Steve Yegge)

**Source**: [steveyegge/beads](https://github.com/steveyegge/beads) (15.8k stars)

### Architecture

Beads is not a planner -- it is a **distributed, git-backed graph issue tracker** that gives AI coding agents persistent memory. It replaces markdown TODO lists with a structured dependency graph. Planning/decomposition is delegated to the AI agent; Beads provides the data structure and query interface.

### How Decomposition Works

Beads enables a three-step planning pattern:

1. Create a design document or brainstorm with the agent
2. Create an epic: `bd create "User Authentication System" -t epic -p 1`
3. Have the agent decompose it: ask the agent to read the epic and create issues that complete the task

The recommended agent integration prompt:

```
At the start of each session:
1. Run `bd ready --json` to see available work
2. Choose an issue to work on
3. Update its status: `bd update <id> --status in_progress`

While working:
- Create new issues for any bugs you discover
- Link related issues with `bd dep add`

When done:
- Close the issue: `bd close <id> --reason "Description of what was done"`
```

### Dependency Graph

Beads supports four relationship types:

- **blocks** -- prevents dependent work from starting
- **parent-child** -- connects tasks to epics
- **related** -- acknowledges connections without blocking
- **discovered-from** -- audit trail for work found during other tasks

The `bd ready` command queries the graph for all open issues with no uncompleted blocking dependencies, returning prioritized work in JSON format. This is the critical planning primitive.

### Key Lessons

- **Persistent plan tracking** -- plans survive across sessions via git-backed storage
- **Dependency-aware work selection** -- `bd ready` returns only unblocked tasks (similar to what Forge's step executor does when picking the next step)
- **Discovered work** -- issues found during implementation get linked back to their origin, creating an audit trail
- **Passive infrastructure** -- requires explicit prompting; agents do not proactively use it

---

## Comparative Analysis

| Dimension | Claude Code | Codex CLI | Gemini CLI | Aider | Forge (current) |
|-----------|------------|-----------|------------|-------|-----------------|
| **When to plan** | Model self-selects via tool | Inline, always available | Model or user selects | Never (human-driven) | Explicit `plan=True` flag |
| **Planner persona** | "Software architect and planning specialist" | Single agent with plan tool | "Gemini CLI, specializing in software engineering" | "Expert architect engineer" | "Task decomposition assistant" |
| **Exploration before planning** | Dedicated Explore subagents (parallel) | Batch file reads | Codebase Investigator (T=0.1) | Repo map + ask mode | Phase 7 exploration (10 rounds) |
| **Plan output** | Markdown file with context, approach, verification | JSON todo list with statuses | Markdown file with steps + verification | N/A | Structured TaskStep list |
| **Verification in plan** | Yes (required) | Implicit (reconcile TODOs) | Yes (mandatory) | N/A | No |
| **Reuse mandate** | Explicit ("search for existing functions") | Implicit ("follow existing patterns") | Explicit ("identify existing patterns") | Via repo map | Not mentioned |
| **Multiple perspectives** | Yes (simplicity vs performance vs maintainability) | No | No | No | No |
| **Plan mode constraint** | Strictly read-only | None (inline planning) | Read-only + plans directory only | N/A | N/A (separate Temporal activity) |
| **Requirements clarification** | AskUserQuestion tool | N/A (batch mode) | ask_user with multiple choice | Human in the loop | N/A |

---

## Recommendations for Forge

Based on this research, the following improvements would have the highest impact on Forge's planner quality:

### 1. Upgrade the Planner Persona

**Current**: "You are a task decomposition assistant."

**Proposed**: A richer persona that frames planning as an architectural activity:

> "You are a senior software architect planning implementation work. Your job is to study the codebase, understand existing patterns and conventions, and decompose the task into precise, ordered steps that a junior developer could execute without ambiguity."

### 2. Add a Research-First Mandate

Every tool surveyed emphasizes understanding before planning. Add explicit instructions:

- Map the relevant parts of the codebase before decomposing
- Identify existing functions, utilities, and patterns to reuse
- Understand the current architecture and conventions
- Verify assumptions against actual code (not just the repo map)

Forge's Phase 7 exploration already supports this. The planner prompt should explicitly instruct the LLM to use exploration rounds for research before producing the plan.

### 3. Require Verification Steps

Both Claude Code and Gemini CLI require plans to include verification. Each step should specify:

- What constitutes success for that step
- How to verify (test commands, assertions, lint checks)
- What regressions to watch for

### 4. Add a Reuse Mandate

Explicitly instruct the planner to prefer reusing existing code:

> "Before creating new abstractions, search for existing functions, classes, and patterns that already solve part of the problem. Reference them by file path and function name in the plan."

### 5. Separate "Why" from "What"

Following OpenSpec's three-artifact pattern, the plan should include:

- **Context**: Why this change is needed and what problem it solves
- **Approach**: The chosen strategy and key design decisions
- **Steps**: The ordered implementation units

### 6. Add Scope Control Instructions

Both Codex and Aider include explicit scope control. Add:

> "Each step should make the minimum change necessary. Do not refactor surrounding code, add documentation, or improve error handling beyond what the task requires."

### 7. Consider Model Routing

Every tool uses its best model for planning:

- Claude Code uses the main model (Opus when available)
- Gemini CLI uses the configured model
- Aider's architect uses the main model (not the weaker editor model)

Forge currently uses Sonnet for planning. The planner is where model quality matters most -- investing Opus-tier tokens in planning pays dividends in execution quality downstream.

### 8. Include Design Documents in Context

The planner currently receives: task description, target file hints, repo map, and context files. It should also receive:

- Project conventions (from CLAUDE.md / design docs)
- Relevant phase specifications
- Architecture documents

These anchor the plan in project-specific patterns rather than generic software engineering.

### 9. Scratchpad / Chain-of-Thought

Gemini CLI's Codebase Investigator uses a scratchpad methodology. Consider adding explicit chain-of-thought instructions:

> "Before producing the final plan, reason through: (1) What files are involved? (2) What are the dependencies between changes? (3) What is the correct ordering? (4) Where can I reuse existing code? (5) What could go wrong?"

This can be implemented via extended thinking / reasoning tokens if the model supports it, or via explicit `<thinking>` sections in the prompt.
