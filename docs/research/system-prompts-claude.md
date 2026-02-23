# Claude Code System Prompts: Reference Catalog

## Source

All prompts extracted from [Piebald-AI/claude-code-system-prompts](https://github.com/Piebald-AI/claude-code-system-prompts), synced to Claude Code v2.1.50 (February 2026). The repository is updated automatically within minutes of each release.

Claude Code does not use a single monolithic system prompt. It assembles its prompt dynamically from ~156 component files across six categories: agent prompts, data references, system prompts, system reminders, tool descriptions, and skills. Components are conditionally included based on environment, configuration, active tools, and session state.

---

## Prompt Architecture

### How Prompts Are Assembled

The final system prompt sent to the model is composed at runtime from:

1. **Core identity** -- the main system prompt establishing Claude Code as a CLI assistant
2. **Behavioral policies** -- doing tasks, tone/style, executing actions with care, tool usage
3. **Tool descriptions** -- one per available tool (Bash, Read, Edit, Write, Glob, Grep, Task, etc.)
4. **Conditional sections** -- plan mode reminders, learning mode, memory instructions, MCP tools
5. **System reminders** -- injected into assistant/tool messages based on events (file modified, token limit, diagnostics)
6. **Agent prompts** -- separate system prompts for subagents (Explore, Plan, general-purpose Task agent)

Many prompts use template variables (e.g., `${BASH_TOOL_NAME}`, `${READ_TOOL_NAME}`) that are interpolated at build time, allowing tool names to change without editing every prompt.

### Token Budget

The assembled prompt is substantial. The core system prompt components alone (identity, policies, tool descriptions) total several thousand tokens. Adding conditional sections, CLAUDE.md contents, git status, and memory files, a typical session's system prompt can reach 10,000+ tokens before any user messages.

---

## Category 1: Core System Prompts

### Main System Prompt (`system-prompt-main-system-prompt.md`)

Establishes identity and top-level behavior. Key elements:

- Declares the agent as "an interactive CLI tool" for software engineering tasks
- References an optional `OUTPUT_STYLE_CONFIG` for custom response styles
- Injects a `SECURITY_POLICY` variable
- Prohibits generating or guessing URLs unless programming-related
- Points users to `/help` and the GitHub issues page for feedback

This is the anchor prompt -- everything else is appended to or injected alongside it.

### Doing Tasks (`system-prompt-doing-tasks.md`)

Governs how Claude Code approaches software engineering work. Key policies:

- Read code before proposing changes
- Prefer editing existing files over creating new ones
- Avoid over-engineering: no features, refactors, or "improvements" beyond what was asked
- No docstrings, comments, or type annotations on unchanged code
- No error handling for scenarios that cannot happen
- No abstractions for one-time operations ("three similar lines is better than a premature abstraction")
- No backwards-compatibility hacks (unused `_vars`, re-exports, `// removed` comments)
- Halt and escalate when confused rather than guessing
- Never give time estimates

### Tone and Style (`system-prompt-tone-and-style.md`)

```
# Tone and style
- Only use emojis if the user explicitly requests it.
- Your output will be displayed on a command line interface. Your responses
  should be short and concise. You can use Github-flavored markdown for
  formatting, and will be rendered in a monospace font using the CommonMark
  specification.
- NEVER create files unless they're absolutely necessary for achieving your
  goal. ALWAYS prefer editing an existing file to creating a new one.
- Do not use a colon before tool calls.

# Professional objectivity
Prioritize technical accuracy and truthfulness over validating the user's
beliefs. Focus on facts and problem-solving, providing direct, objective
technical info without any unnecessary superlatives, praise, or emotional
validation. [...] Avoid using over-the-top validation or excessive praise
when responding to users such as "You're absolutely right" or similar phrases.

# No time estimates
Never give time estimates or predictions for how long tasks will take [...]
```

Notable: the "professional objectivity" section explicitly instructs the model to disagree with users when warranted and avoid phrases like "You're absolutely right."

### Executing Actions with Care (`system-prompt-executing-actions-with-care.md`)

A safety-focused prompt governing reversibility and blast radius:

- Local, reversible actions (file edits, running tests) are generally safe to proceed with
- Risky actions require user confirmation before execution:

    - **Destructive**: deleting files/branches, dropping tables, `rm -rf`, overwriting uncommitted changes
    - **Hard-to-reverse**: force-push, `git reset --hard`, removing dependencies, modifying CI/CD
    - **Visible to others**: pushing code, creating/commenting on PRs, sending messages to external services

- Approving one action does not authorize it in all contexts
- When encountering obstacles, investigate root causes rather than bypassing safety checks
- When discovering unexpected state (unfamiliar files, branches), investigate before deleting

The guiding principle: "measure twice, cut once."

### Security Policy (`system-prompt-censoring-assistance-with-malicious-activities.md`)

Short and direct:

- Assist with authorized security testing, defensive security, CTF challenges, educational contexts
- Refuse destructive techniques, DoS, mass targeting, supply chain compromise, detection evasion for malicious purposes
- Dual-use tools (C2 frameworks, credential testing, exploit development) require clear authorization context

---

## Category 2: Tool Descriptions

Each built-in tool has its own description file injected into the system prompt. These serve as both documentation and behavioral constraints.

### Notable Tool Descriptions

| Tool | File | Key Points |
|------|------|------------|
| Bash | `tool-description-bash.md` | Timeout defaults, quoting rules, sandbox notes, preference for dedicated tools over shell commands |
| Read | `tool-description-readfile.md` | 2000-line default, supports images/PDFs/notebooks, line-number format |
| Edit | `tool-description-edit.md` | Exact string replacement, must Read first, `old_string` must be unique |
| Write | `tool-description-write.md` | Must Read existing files first, never proactively create docs/READMEs |
| Glob | `tool-description-glob.md` | Fast pattern matching, results sorted by modification time |
| Grep | `tool-description-grep.md` | Built on ripgrep, supports regex, glob filtering, multiline mode |
| Task | `tool-description-task.md` | Launches subagents, foreground vs background, worktree isolation |
| WebFetch | `tool-description-webfetch.md` | HTML-to-markdown conversion, 15-minute cache, redirect handling |

### Git Commit and PR Instructions (`tool-description-bash-git-commit-and-pr-creation-instructions.md`)

Embedded within the Bash tool description, this is one of the longest prompt components. It specifies:

- **Git Safety Protocol**: never update git config, never run destructive commands unless explicitly requested, always create NEW commits rather than amending (critical after pre-commit hook failures)
- **Commit workflow**: parallel `git status` + `git diff` + `git log`, then analyze and draft message, then stage + commit + verify
- **PR workflow**: check branch state, analyze ALL commits (not just latest), create PR with structured body using HEREDOC
- **Co-authorship**: commits include `Co-Authored-By: Claude <model> <noreply@anthropic.com>`

---

## Category 3: Agent Prompts

Claude Code uses a subagent architecture. Each subagent type has its own system prompt, separate from the main session prompt.

### Explore Agent (`agent-prompt-explore.md`)

A read-only, speed-optimized search specialist:

```
You are a file search specialist for Claude Code [...] You excel at
thoroughly navigating and exploring codebases.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
[...]
```

- Runs on Haiku for speed
- Tools limited to Glob, Grep, Read, and read-only Bash
- Caller specifies thoroughness: "quick", "medium", or "very thorough"
- Instructed to maximize parallel tool calls for efficiency

### General-Purpose Task Agent (`agent-prompt-task-tool.md`)

The workhorse subagent for multi-step research and code changes:

```
You are an agent for Claude Code [...] Given the user's message, you
should use the tools available to complete the task. Do what has been
asked; nothing more, nothing less.
```

- Has access to all tools (read, write, edit, bash, web, etc.)
- Instructed to start broad and narrow down
- Must return absolute file paths
- Never proactively creates documentation

### Plan Mode Agent (`agent-prompt-plan-mode-enhanced.md`)

A software architect specialist, also strictly read-only:

- Four-step process: explore codebase, design approach, structure output, present plan
- Tools limited to read-only operations
- Output follows a required format with implementation steps, critical files, and architectural trade-offs
- Cannot execute the plan -- only designs it for user approval

### CLAUDE.md Creation Agent (`agent-prompt-claudemd-creation.md`)

Generates project-level `CLAUDE.md` files by analyzing a repository:

- Documents essential commands (build, lint, test)
- Provides architecture context requiring cross-file understanding
- Integrates guidance from `.cursor/rules`, `.cursorrules`, `.github/copilot-instructions.md`, README
- Improves existing files rather than replacing them
- Excludes generic practices applicable to any codebase

### Agent Architect (`agent-prompt-agent-creation-architect.md`)

Meta-agent for designing new custom agents. Its output is a JSON specification with:

- `identifier`: lowercase-hyphenated name
- `whenToUse`: trigger description
- `systemPrompt`: the full system prompt for the new agent

Emphasizes specificity over generality and autonomous operation with minimal guidance.

### Conversation Summarization (`agent-prompt-conversation-summarization.md`)

Produces structured handoff summaries when context needs to be compressed. Required sections:

1. Primary Request and Intent
2. Key Technical Concepts
3. Files and Code Sections (with full snippets)
4. Errors and Fixes
5. Problem Solving
6. All User Messages (verbatim listing)
7. Pending Tasks
8. Current Work (with file names and code)
9. Optional Next Step (with direct quotes from recent conversation)

Uses an `<analysis>` tag for chain-of-thought before producing the summary.

---

## Category 4: System Reminders

System reminders are short messages injected into the conversation as `<system-reminder>` tags within tool results or assistant messages. They are event-driven notifications, not part of the base system prompt.

### Examples

| Reminder | Trigger | Purpose |
|----------|---------|---------|
| `file-modified-by-user-or-linter` | External file change detected | Alerts that file contents may differ from last read |
| `output-token-limit-exceeded` | Response truncated | Tells model to continue in next turn |
| `plan-mode-is-active-5-phase` | Plan mode entered | Injects the 5-phase planning protocol |
| `token-usage` | Periodically | Reports token consumption |
| `todo-list-changed` | Task list modified | Reminds model to check task state |
| `new-diagnostics-detected` | IDE diagnostics appear | Surfaces errors/warnings from the editor |
| `session-continuation` | Session resumed | Provides context about prior work |

---

## Category 5: Conditional and Specialized Prompts

### Task Management (`system-prompt-task-management.md`)

Instructs the model to use todo/task tools "VERY frequently" for tracking and planning. Provides examples of breaking down tasks into numbered items and marking them in-progress/completed as work proceeds.

### Learning Mode (`system-prompt-learning-mode.md`)

An alternate operational mode where Claude Code balances task completion with education:

- Requests human contributions on design decisions and algorithms
- Uses structured "Learn by Doing" formatting with context, task description, and guidance
- Inserts `TODO(human)` markers in code and waits for human implementation
- Shares insights after human contributions

### Memory Instructions (`system-prompt-agent-memory-instructions.md`)

Guidance for agents to accumulate knowledge across conversations:

- Record concise notes about findings and locations
- Domain-specialized memory instructions for different agent types (code reviewers, test runners, architects, documentation writers)
- Memory should align with each agent's core responsibilities

### Context Compaction (`system-prompt-context-compaction-summary.md`)

Template for creating structured handoff documents when pausing work. Five required sections:

1. Task Overview (request, success criteria, constraints)
2. Current State (completed work, modified files, key outputs)
3. Important Discoveries (technical findings, decisions, failed approaches)
4. Next Steps (remaining actions, blockers, priority ordering)
5. Context to Preserve (user preferences, domain details, commitments)

### Conditional Codebase Exploration Delegation (`system-prompt-conditional-delegate-codebase-exploration.md`)

Rules for when to use the Explore subagent vs. direct Glob/Grep calls:

- Direct tools for simple, directed searches (specific file/class/function)
- Explore subagent for broader exploration requiring multiple queries or uncertain naming conventions

---

## Category 6: Data Reference Files

17+ files containing embedded documentation that gets injected when the `claude-developer-platform` skill is active:

- **API references**: Python, TypeScript, Go, Java, Ruby, PHP, C# SDK usage
- **Agent SDK patterns**: Python and TypeScript agent construction
- **Streaming references**: SSE and streaming patterns for Python and TypeScript
- **Tool use**: Concepts overview plus language-specific references
- **Files API and Batches API**: Python and TypeScript references
- **Model catalog**: Available Claude models with IDs and capabilities
- **HTTP error codes**: Standard error handling reference

These are not behavioral prompts -- they are reference material injected into context so the model can write accurate SDK code without hallucinating API shapes.

---

## Key Design Patterns

### 1. Compositional Assembly

No single prompt file contains the full instructions. The system is modular: each concern (identity, safety, tools, style, task management) is a separate file, conditionally composed at runtime. This allows different session configurations (plan mode, learning mode, team mode) to swap components without duplicating shared policies.

### 2. Read-Before-Write Discipline

Multiple prompts independently enforce the same constraint: you must read a file before editing or writing it. This appears in the Edit tool description, Write tool description, and the doing-tasks policy. The redundancy is intentional -- the constraint is critical enough to repeat across contexts where the model might encounter it.

### 3. Explicit Anti-Patterns

Rather than only saying what to do, prompts enumerate what NOT to do:

- Do not over-engineer
- Do not add docstrings to unchanged code
- Do not create abstractions for one-time operations
- Do not use Bash for file operations when dedicated tools exist
- Do not give time estimates
- Do not use emojis unless asked
- Do not use colons before tool calls

### 4. Subagent Specialization with Hard Constraints

Subagents are given narrow tool access enforced at the framework level (not just instructed):

- Explore agent: disallowed tools include Task, Edit, Write, NotebookEdit
- Plan agent: same restrictions
- General-purpose agent: full tool access

The system prompt reinforces these constraints with explicit "STRICTLY PROHIBITED" language, but the real enforcement is the tool allow-list.

### 5. Event-Driven Context Injection

System reminders are not part of the base prompt -- they are injected into specific messages when events occur. This keeps the base prompt lean while still providing relevant guidance at the moment it matters (e.g., reminding about diagnostics only when diagnostics appear).

### 6. Template Variables for Indirection

Tool names, subscription types, environment flags, and feature gates are all injected via template variables. This means the same prompt text works across different configurations without conditional logic in the prompt itself.

---

## Relevance to Forge

Several patterns from Claude Code's prompt architecture are directly applicable to Forge's document-completion workflow:

1. **Compositional prompt assembly** -- Forge already assembles prompts from context providers. Claude Code's approach of separate, concern-specific files with template variables is a more maintainable version of the same idea.

2. **Read-before-write redundancy** -- For Forge's diff-based output mode, the equivalent constraint (current file contents must be in context for accurate search/replace) could benefit from similar multi-point reinforcement.

3. **Anti-pattern enumeration** -- Forge's prompts could benefit from explicit "do NOT" lists tuned to common LLM failure modes in code generation (e.g., do not add imports you do not use, do not rename variables for style, do not add error handling for internal calls).

4. **Subagent specialization** -- Forge's fan-out sub-tasks already run in isolation. Giving different sub-task types (planning, code generation, testing) different system prompts and tool constraints mirrors Claude Code's approach.

5. **Event-driven context** -- Forge's error-aware retries (Phase 8) already inject validation errors into retry prompts. This could be generalized to a broader event-driven injection system (e.g., injecting git conflict information, test results, lint output at the moment they become relevant).

6. **Professional objectivity directive** -- The explicit instruction to avoid sycophantic agreement and provide honest technical feedback is useful for any LLM system that evaluates its own work or reviews plans.
