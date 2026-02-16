# Phase 15: MCP & Agent Skills Integration

## Summary

Extend Forge workflows to leverage external capabilities provided by MCP (Model Context Protocol) servers and Agent Skills (agentskills.io standard). Instead of allowing the LLM to call MCP tools directly (which would break batch-mode compatibility), capabilities are presented in the exploration loop's unified catalog. When the LLM requests an action, the orchestrator dispatches it as a Temporal child workflow.

## Motivation

Forge's exploration loop (Phase 7) gives the LLM access to context providers for gathering information before code generation. However, many useful operations require **external tool execution** — running database queries, calling APIs, interacting with cloud services, executing specialized scripts — that go beyond Forge's built-in providers.

MCP servers expose structured tools with JSON Schema-validated inputs/outputs. Agent Skills package reusable instructions, scripts, and resources that any compatible agent can use. By integrating both, Forge workflows can access a broad ecosystem of capabilities without building each one into the core.

## Architecture

### Unified Capability Catalog

The exploration LLM sees a single flat list of available capabilities:

1. **Context providers** (existing) — `read_file`, `search_code`, etc.
2. **MCP tools** — discovered from configured MCP servers at workflow start.
3. **Agent Skills** — discovered from `SKILL.md` files in configured directories.

All capabilities are presented as `ContextProviderSpec` entries (name, description, parameters). The LLM doesn't need to know the dispatch mechanism — it requests by name.

### Request Flow

```
Exploration LLM
    │
    ├─ ContextRequest (provider=read_file)   ─► fulfill_context_requests activity (existing)
    │
    └─ ActionRequest (capability=mcp_github_search) ─► ForgeActionWorkflow child workflow
        │                                                    │
        │                                                    ├─ MCP tool → direct dispatch activity
        │                                                    │
        │                                                    └─ Skill → LLM-mediated subagent activity
        │
        └─ Results flow back as ContextResult ─► accumulated_context for next round
```

### Dispatch Model (Hybrid)

- **MCP tools**: Direct dispatch. The activity starts an MCP client session, calls the tool with the provided arguments, and returns the result. No intermediate LLM call needed because MCP tools have well-defined schemas.

- **Agent Skills**: LLM-mediated subagent. A secondary LLM call receives the SKILL.md instructions, the action request, and available scripts/references. The LLM interprets the skill's instructions and produces the result. This handles Skills' free-form instruction format.

### Batch Compatibility

All LLM calls continue through the batch API (Phase 14). Actions are **not** LLM calls — they are Temporal activities and child workflows that execute between batch rounds. The exploration loop's structure is unchanged:

1. Submit exploration LLM call (batch or sync) → get `ExplorationResponse`
2. Fulfill context requests (activity, synchronous)
3. Dispatch action requests (child workflows, parallel)
4. Accumulate results → next exploration round

### MCP Server Lifecycle

MCP server connections are **per-activity** — the activity starts the MCP client session, makes the tool call, and tears down. This keeps:
- Workflows deterministic (no side-effect state)
- Activities idempotent and retryable
- No long-lived process management needed

For frequently-called MCP servers, a connection pool can be added as a future optimization.

## Configuration

### MCP Servers: `forge-mcp.json`

Located in the project root (or specified via `--mcp-config`):

```json
{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
            "env": {}
        },
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
                "GITHUB_TOKEN": "${GITHUB_TOKEN}"
            }
        }
    }
}
```

Environment variable interpolation (`${VAR}`) is supported.

### Agent Skills

Skill directories are configured via:
- `--skills-dir` CLI flag (repeatable)
- `forge-mcp.json` `"skillsDirs"` array
- Default: `.forge/skills/` in the project root

Each skill is a directory containing a `SKILL.md` file with YAML frontmatter:

```markdown
---
name: deploy-preview
description: Deploy a preview environment for the current branch
---

## Instructions
1. Build the application: `scripts/build.sh`
2. Deploy to preview: `scripts/deploy.sh`
...
```

## Data Models

### New Models

```python
class CapabilityType(StrEnum):
    PROVIDER = "provider"
    MCP_TOOL = "mcp_tool"
    SKILL = "skill"

class ActionRequest(BaseModel):
    """LLM's request to execute an MCP tool or Skill."""
    capability: str
    capability_type: CapabilityType
    params: dict[str, Any] = Field(default_factory=dict)
    reasoning: str

class ActionResult(BaseModel):
    """Result of executing an action."""
    capability: str
    capability_type: CapabilityType
    content: str
    success: bool
    estimated_tokens: int

class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

class MCPConfig(BaseModel):
    """Top-level external capabilities configuration."""
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    skills_dirs: list[str] = Field(default_factory=list)

class SkillDefinition(BaseModel):
    """Parsed Agent Skill from SKILL.md."""
    name: str
    description: str
    instructions: str
    skill_path: str
    allowed_tools: list[str] = Field(default_factory=list)

class ActionWorkflowInput(BaseModel):
    """Input to ForgeActionWorkflow."""
    action: ActionRequest
    mcp_config: MCPConfig = Field(default_factory=MCPConfig)
    skill_definitions: dict[str, SkillDefinition] = Field(default_factory=dict)
    repo_root: str = ""
    worktree_path: str = ""
    model_name: str = ""
    sync_mode: bool = False

class ActionWorkflowResult(BaseModel):
    """Output of ForgeActionWorkflow."""
    result: ActionResult
```

### Modified Models

`ExplorationResponse` gains an `actions` field:

```python
class ExplorationResponse(BaseModel):
    requests: list[ContextRequest] = Field(...)
    actions: list[ActionRequest] = Field(
        default_factory=list,
        description="Action requests for MCP tools or Skills. Empty together with requests signals readiness.",
    )
```

`ForgeTaskInput` gains capability configuration:

```python
class ForgeTaskInput(BaseModel):
    # ... existing fields ...
    mcp_config: MCPConfig = Field(default_factory=MCPConfig)
    skills_dirs: list[str] = Field(default_factory=list)
```

## New Modules

### `src/forge/mcp/config.py`
Load and validate `forge-mcp.json`. Environment variable interpolation.

### `src/forge/mcp/client.py`
Async MCP client wrapper. Tool discovery (`list_tools`) and invocation (`call_tool`).

### `src/forge/skills/loader.py`
SKILL.md frontmatter parser. Skill directory scanner. Progressive loading (metadata first, full content on demand).

### `src/forge/capabilities.py`
Unified capability registry. Merges built-in providers, MCP tools, and Skills into a single catalog. Provides `discover_capabilities()` and `get_capability_specs()`.

### `src/forge/activities/actions.py`
Two activities:
- `execute_mcp_tool` — Direct MCP tool invocation.
- `execute_skill` — LLM-mediated skill execution.

### `ForgeActionWorkflow` (in `src/forge/workflows.py`)
Child workflow that dispatches to the appropriate activity based on `CapabilityType`.

## Exploration Loop Changes

The `_run_exploration_loop` method in `ForgeTaskWorkflow` is updated:

```python
async def _run_exploration_loop(self, task, repo_root, worktree_path, max_rounds, ...):
    # Discover capabilities at loop start
    capability_specs = get_capability_specs(providers, mcp_config, skill_defs)

    for round_num in range(1, max_rounds + 1):
        exploration_input = ExplorationInput(
            task=task,
            available_providers=capability_specs,  # unified list
            accumulated_context=accumulated,
            ...
        )
        result = await self._call_exploration(exploration_input)

        if not result.requests and not result.actions:
            break  # Ready to generate

        # Fulfill context requests (existing path)
        if result.requests:
            context_results = await fulfill_context_requests(...)
            accumulated.extend(context_results)

        # Dispatch action requests as child workflows (parallel)
        if result.actions:
            action_results = await self._dispatch_actions(result.actions, ...)
            accumulated.extend(action_results)

    return accumulated
```

## CLI Changes

```
forge run --mcp-config forge-mcp.json --skills-dir ./skills ...
```

## Decisions

- **D84**: Unified capability catalog — providers, MCP tools, and Skills in one list.
- **D85**: Actions as child workflows for isolation, batch compat, and observability.
- **D86**: Hybrid dispatch — direct for MCP, LLM-mediated for Skills.
- **D87**: Per-activity MCP connections (no long-lived processes).
- **D88**: JSON config file for MCP servers (Claude Desktop convention).
- **D89**: ExplorationResponse extended with `actions` field (backwards compatible).

## Testing Strategy

- Unit tests for config parsing, SKILL.md loading, capability registry.
- Mock MCP client for tool discovery/invocation tests.
- Integration tests for action workflow dispatch.
- Existing exploration tests continue to pass (backwards compat).
