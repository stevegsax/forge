# Task Decomposition Strategy

**Status:** Draft
**Date:** 2026-03-06

---

## Overview

This document describes Forge's task decomposition system: a multi-transform pipeline that takes a user's natural language request and produces a validated, dependency-ordered DAG of atomic tasks. The system uses iterative refinement with human clarification gates and adversarial LLM-as-judge review before finalizing a plan.

### Design Principles

1. **Many small transforms over few large ones.** Each LLM call does one thing. We prefer three simple calls to one complex one.
2. **Conservative atomicity.** Leaf tasks should be simpler than what we think an LLM can handle. Err toward splitting.
3. **Accuracy through adversarial review.** Plans are challenged by adversarial judge personas before acceptance.
4. **Auditability by default.** Every intermediate version of the plan is persisted.
5. **Pause, don't guess.** When the system needs information from the user, the workflow pauses until they respond.

---

## Transform Pipeline

The pipeline is a sequence of Temporal activities connected by a workflow. Each transform is a separate LLM call (or deterministic function) that produces a versioned artifact.

```
User Input
  |
  v
[1. Classify]        -- Pick workflow type (software, research, etc.)
  |
  v
[2. Clarify]         -- Ask structured questions, pause for user
  |                     (loop until no more questions)
  v
[3. Goal Statement]  -- Produce a precise goal; user confirms
  |
  v
[4. First Pass]      -- Broad decomposition (~3-7 top-level tasks)
  |
  v
[5. Recursive Split] -- Each non-leaf node -> children (loop)
  |                     (ask user if ambiguous)
  v
[6. Dependency Analysis] -- Add ordering edges to the DAG
  |
  v
[7. Acceptance Criteria] -- Add "definition of done" per leaf
  |
  v
[8. Deterministic Checks] -- Structural validation (no LLM)
  |
  v
[9. Adversarial Review]   -- 3 judges, 2-of-3 consensus
  |                          (up to 3 rounds; then escalate)
  v
[10. User Approval]  -- Present JSON + rendered DAG
```

Each transform reads the current plan version, produces a new version, and persists both to the plan database. The workflow is a linear sequence with two loop points: Clarify (steps 2-3) and Review (steps 8-9, up to 3 attempts).

---

## Data Model

### PlanNode

The unit of work in the DAG. Every node has exactly one of three execution types.

```python
class ExecutionType(StrEnum):
    """How this node gets executed."""
    LLM_CALL = "llm_call"              # Document completion via Forge
    HUMAN_ACTION = "human_action"       # Human does it, reports back
    DETERMINISTIC = "deterministic"     # A program/script/function call

class PlanNode(BaseModel):
    """A single node in the plan DAG."""
    node_id: str                        # UUID
    title: str                          # Short human-readable label
    description: str                    # What this node accomplishes
    execution_type: ExecutionType
    workflow_type: str                  # "software", "research", etc.
    acceptance_criteria: list[str]      # Definition of done
    estimated_complexity: str           # "trivial" | "simple" | "moderate"
    context: dict[str, Any]             # Workflow-type-specific metadata
    children: list[str]                 # UUIDs of child nodes (if not a leaf)
    is_leaf: bool                       # True = executable, False = container
```

### PlanEdge

Explicit dependency between nodes.

```python
class EdgeType(StrEnum):
    """Relationship between nodes."""
    DEPENDS_ON = "depends_on"           # Target must complete before source starts
    PARENT_CHILD = "parent_child"       # Decomposition relationship

class PlanEdge(BaseModel):
    """A directed edge in the plan DAG."""
    edge_id: str                        # UUID
    source_id: str                      # Node that depends
    target_id: str                      # Node that is depended on
    edge_type: EdgeType
    rationale: str                      # Why this dependency exists
```

### PlanDAG

The top-level plan artifact.

```python
class PlanDAG(BaseModel):
    """A complete plan as a directed acyclic graph."""
    plan_id: str                        # UUID
    version: int                        # Monotonically increasing
    goal_statement: str                 # The agreed-upon goal
    workflow_type: str                  # Primary workflow type
    nodes: dict[str, PlanNode]          # node_id -> PlanNode
    edges: list[PlanEdge]
    metadata: PlanMetadata              # Timestamps, user info, etc.
```

### PlanVersion

For auditability, every mutation creates a new version.

```python
class PlanVersion(BaseModel):
    """A snapshot of the plan at a point in time."""
    plan_id: str
    version: int
    transform_name: str                 # Which pipeline step produced this
    plan_dag: PlanDAG
    parent_version: int | None          # Previous version (None for v1)
    created_at: datetime
    llm_interaction_id: str | None      # FK to interactions table
```

---

## Transform Details

### 1. Classify

**Input:** User's raw request text.
**Output:** A `workflow_type` string and confidence score.
**Method:** The LLM receives the user's request plus a catalog of available workflow types (loaded from the workflow templates directory). Each entry has a `description.md` explaining when to use it. If confidence is below a threshold, the system presents the options to the user as a structured form.

```
Workflow templates catalog:
- software: "Building, modifying, or debugging software programs..."
- research: "Investigating a topic, gathering evidence, synthesizing findings..."
- ...

User request: "write a program that uses WebGPU to draw a rotating cube"

Classify this request. Return the workflow_type and your confidence (0-1).
```

**Model tier:** CLASSIFICATION (lightweight, fast).

### 2. Clarify

**Input:** User request + workflow type.
**Output:** A list of `ClarificationQuestion` objects, or an empty list (no questions needed).
**Method:** The LLM receives the user request, the workflow type, and the workflow's `clarify.prompt.j2` template. It produces structured questions with defaults.

```python
class ClarificationQuestion(BaseModel):
    """A question for the user with a suggested default."""
    question_id: str                    # UUID
    question_text: str
    question_type: str                  # "choice" | "text" | "confirm"
    options: list[str] | None           # For "choice" type
    default: str | None                 # Suggested answer
    importance: str                     # "required" | "recommended" | "optional"
    rationale: str                      # Why this matters for the plan
```

**Temporal pattern:** When questions are produced, the workflow emits them as a signal payload and pauses via `workflow.wait_condition()` until the user responds via signal. The user sees a structured form with defaults pre-filled.

**Loop:** After the user responds, the system may ask follow-up questions. The loop terminates when the LLM produces an empty question list.

**Model tier:** GENERATION.

### 3. Goal Statement

**Input:** User request + clarification answers + workflow type.
**Output:** A precise, unambiguous goal statement.
**Method:** The LLM synthesizes everything into a single goal statement. The user must confirm it. If the user rejects, the system returns to step 2 (Clarify) with the user's feedback.

**Temporal pattern:** Same signal/wait pattern. The user receives the goal statement and responds with "approve" or "revise: <feedback>".

**Model tier:** GENERATION.

### 4. First Pass Decomposition

**Input:** Goal statement + workflow type + workflow-specific decompose template.
**Output:** PlanDAG v1 with ~3-7 top-level nodes (may not be leaves yet).
**Method:** The LLM produces a broad decomposition. Nodes at this stage may be containers (non-leaf) that need further splitting.

The workflow's `decompose.prompt.j2` template provides domain-specific guidance. For software: "think about data models, then business logic, then integration, then tests." For research: "think about research questions, then sources, then synthesis."

**Model tier:** REASONING (this is the hardest step).

### 5. Recursive Split

**Input:** PlanDAG with non-leaf nodes.
**Output:** PlanDAG where all nodes satisfy the atomicity test.
**Method:** For each non-leaf node, the LLM is asked to decompose it into children. After the LLM proposes children, a second LLM call (the atomicity judge) confirms each child is truly atomic — completable in a single LLM call, human response, or method invocation. If the judge says "not atomic," the child becomes a non-leaf and gets split in the next iteration.

**Cross-workflow spawning:** If a software node requires research (e.g., "investigate which WebGPU library to use"), the LLM can mark it with a different `workflow_type`. This creates a sub-plan that goes through its own full pipeline (steps 1-10) and must be approved by the user independently.

**Clarification:** If the LLM encounters ambiguity during splitting, it raises a `ClarificationQuestion` and the workflow pauses.

**Loop termination:** The loop ends when every node is either a leaf (confirmed atomic) or a container whose children are all leaves.

**Model tier:** REASONING for splitting, CLASSIFICATION for atomicity check.

### 6. Dependency Analysis

**Input:** PlanDAG with all leaf nodes identified.
**Output:** PlanDAG with `DEPENDS_ON` edges added.
**Method:** The LLM reviews all leaf nodes and identifies ordering constraints. It produces edges with rationale ("node B reads the database schema created by node A").

This is a single LLM call that receives the full node list and produces a list of edges. A deterministic validation step follows to confirm the graph is acyclic.

**Model tier:** REASONING.

### 7. Acceptance Criteria

**Input:** PlanDAG with dependencies.
**Output:** PlanDAG where every leaf node has `acceptance_criteria` populated.
**Method:** For each leaf node, the LLM produces specific, testable acceptance criteria. The workflow's `criteria.prompt.j2` template provides domain-specific guidance.

For software leaves: "what tests should pass? what lint checks apply? what does correct output look like?"
For research leaves: "what question is answered? what sources are cited? what claims are supported?"
For human action leaves: "what artifact does the human produce? how is completion verified?"

This can be parallelized — each leaf's criteria generation is independent.

**Model tier:** GENERATION.

### 8. Deterministic Checks

**Input:** Completed PlanDAG.
**Output:** `DeterministicResult` (pass/fail with details).
**Method:** No LLM. Pure functions that validate structural properties:

- DAG is acyclic
- All node IDs are unique UUIDs
- All edge references point to existing nodes
- Every leaf has at least one acceptance criterion
- Every leaf has an execution_type
- No orphan nodes (every non-root node has a parent edge)
- Cross-workflow references are valid
- Container nodes have at least 2 children
- No circular parent-child relationships

If any check fails, the system loops back to step 5 (Recursive Split) with the failure details injected into the prompt. This counts as one of the 3 allowed revision attempts.

### 9. Adversarial Review

**Input:** PlanDAG that passed deterministic checks.
**Output:** Consensus verdict from 3 judges.
**Method:** Three independent LLM calls, each with a different adversarial persona adapted from the DeepResearch cognitive model. Each judge must argue AGAINST the plan before arguing FOR it, then produce a verdict.

#### Persona Selection

The system selects 3 of 7 personas based on workflow type:

| Persona | Focus | Best for |
|---------|-------|----------|
| **Expert Skeptic** | Edge cases, failure modes, missing error handling | software, research |
| **Detail Analyst** | Precise specifications, missing parameters, vague criteria | software, research |
| **Completeness Auditor** | Coverage gaps, missing steps, overlooked requirements | all |
| **Dependency Critic** | Ordering errors, hidden dependencies, parallelism opportunities | software |
| **Scope Guardian** | Scope creep, unnecessary steps, gold-plating | all |
| **Feasibility Assessor** | Whether leaf tasks are truly achievable in one call | all |
| **Consistency Checker** | Contradictions between nodes, conflicting acceptance criteria | research, software |

Default selection by workflow type:

- **software:** Expert Skeptic, Completeness Auditor, Dependency Critic
- **research:** Detail Analyst, Completeness Auditor, Scope Guardian
- **generic:** Completeness Auditor, Scope Guardian, Feasibility Assessor

#### Judge Prompt Structure

Each judge receives the same plan but a different persona prompt:

```
You are {persona_name}. Your role is to {persona_description}.

## Plan Under Review
{serialized_plan_dag}

## Evaluation Criteria
Score each dimension 1-5:
1. COMPLETENESS: Does the plan cover the entire goal?
2. GRANULARITY: Is each leaf task truly atomic (one LLM call / one human action / one function)?
3. FEASIBILITY: Can each leaf task actually be completed as described?
4. DEPENDENCY_CORRECTNESS: Are ordering constraints correct and complete?
5. ACCEPTANCE_CRITERIA_QUALITY: Are the "done" conditions specific and testable?

## Required Response Structure

### Arguments AGAINST This Plan
List every weakness, gap, risk, and failure mode you can identify.
Be thorough and adversarial. Assume the plan WILL fail and explain why.

### Arguments FOR This Plan
Now argue that the plan is adequate despite the weaknesses above.
Which weaknesses are acceptable? Which are mitigated by other aspects of the plan?

### Verdict
APPROVE or REJECT.
If REJECT, list the specific changes required (not suggestions — requirements).

### Scores
{criterion}: {score} - {rationale}
```

#### Consensus Rule

- **2-of-3 APPROVE:** Plan is accepted.
- **2-of-3 REJECT:** Plan is revised. The system collects all required changes from rejecting judges, feeds them back to step 5 (Recursive Split) as revision instructions, and produces a new plan version.
- **Up to 3 rounds.** If after 3 rounds judges still reject, the system stops and escalates to the user with the judge feedback.

**Model tier:** REASONING for all judges (this is where quality matters most).

**Parallelism:** All 3 judge calls are independent and run concurrently.

### 10. User Approval

**Input:** Judge-approved PlanDAG.
**Output:** User's approval or rejection.
**Method:** The user receives:

1. The PlanDAG as formatted JSON
2. A rendered DOT diagram (generated by a deterministic `plan_to_dot()` function)
3. A summary showing: node count, leaf count, estimated total steps, workflow types involved

The workflow pauses via signal until the user responds. If the user rejects, the workflow terminates (per requirement: revision only on explicit request).

---

## Workflow Templates

Templates are organized in a directory-per-workflow layout using Jinja2:

```
src/forge/workflow_templates/
├── _shared/
│   ├── clarify_base.prompt.j2
│   ├── goal_base.prompt.j2
│   ├── criteria_base.prompt.j2
│   └── judge_base.prompt.j2
├── software/
│   ├── description.md
│   ├── classify.prompt.j2
│   ├── clarify.prompt.j2
│   ├── decompose.prompt.j2
│   ├── split.prompt.j2
│   ├── criteria.prompt.j2
│   └── judge.prompt.j2
├── research/
│   ├── description.md
│   ├── clarify.prompt.j2
│   ├── decompose.prompt.j2
│   ├── split.prompt.j2
│   ├── criteria.prompt.j2
│   └── judge.prompt.j2
└── ...
```

**`description.md`** contains a natural-language description of the workflow type, its capabilities, and when it should be selected. These are concatenated into the classification prompt.

**Template variables** available in all templates:

- `{{ goal }}` — the confirmed goal statement
- `{{ user_request }}` — the original user request
- `{{ clarification_answers }}` — dict of question_id -> answer
- `{{ workflow_type }}` — the selected workflow type
- `{{ current_plan }}` — serialized current PlanDAG (for revision steps)
- `{{ revision_feedback }}` — judge feedback (for revision steps)
- `{{ repo_context }}` — repo map if applicable

**Shared templates** (`_shared/`) provide base structure. Workflow-specific templates extend them with domain guidance. For example, `software/decompose.prompt.j2`:

```jinja2
{% extends "_shared/decompose_base.prompt.j2" %}

{% block domain_guidance %}
When decomposing software tasks, consider this order:
1. Data models and schemas first
2. Core business logic second
3. Integration and glue code third
4. Tests and validation last

Each leaf task should modify at most 2-3 files.
Prefer creating new files over modifying existing ones when possible.
{% endblock %}
```

---

## Plan Database

A separate SQLite database at `$XDG_STATE_HOME/forge/plans.db` stores all plan data. This keeps the plan lifecycle independent of the observability store.

### Tables

#### plans

| Column | Type | Description |
|--------|------|-------------|
| plan_id | TEXT PK | UUID |
| goal_statement | TEXT | The agreed-upon goal |
| workflow_type | TEXT | Primary workflow type |
| status | TEXT | draft, reviewing, approved, rejected, executing, completed |
| user_request | TEXT | Original user input |
| created_at | TIMESTAMP | |
| updated_at | TIMESTAMP | |

#### plan_versions

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| plan_id | TEXT FK | References plans |
| version | INTEGER | Monotonically increasing per plan |
| transform_name | TEXT | Pipeline step that produced this version |
| plan_json | TEXT | Full PlanDAG serialized as JSON |
| parent_version | INTEGER NULL | Previous version number |
| llm_interaction_id | TEXT NULL | FK to observability store interactions |
| created_at | TIMESTAMP | |

#### clarifications

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| plan_id | TEXT FK | References plans |
| question_id | TEXT | UUID |
| question_json | TEXT | Serialized ClarificationQuestion |
| answer | TEXT NULL | User's response |
| answered_at | TIMESTAMP NULL | |
| created_at | TIMESTAMP | |

#### judge_reviews

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| plan_id | TEXT FK | References plans |
| version | INTEGER | Which plan version was reviewed |
| round | INTEGER | Review round (1-3) |
| persona | TEXT | Judge persona name |
| arguments_against | TEXT | |
| arguments_for | TEXT | |
| verdict | TEXT | approve / reject |
| required_changes | TEXT NULL | JSON list (if reject) |
| scores_json | TEXT | Serialized scores |
| llm_interaction_id | TEXT NULL | FK to observability store |
| created_at | TIMESTAMP | |

---

## Temporal Workflow Design

### DecompositionWorkflow

The top-level workflow that orchestrates the entire pipeline.

```python
@workflow.defn
class DecompositionWorkflow:
    """Orchestrates the full decomposition pipeline."""

    def __init__(self) -> None:
        self._user_responses: list[UserResponse] = []
        self._status: str = "initializing"

    @workflow.signal
    async def user_response(self, response: UserResponse) -> None:
        self._user_responses.append(response)

    @workflow.run
    async def run(self, input: DecompositionInput) -> DecompositionResult:
        # 1. Classify
        classification = await workflow.execute_activity(
            "classify_request", ...)

        # 2-3. Clarify + Goal (loop)
        goal = await self._clarify_and_confirm_goal(input, classification)

        # 4-7. Decompose (with recursive split loop)
        plan = await self._decompose(goal, classification)

        # 8-9. Validate + Review (up to 3 rounds)
        plan = await self._validate_and_review(plan)

        # 10. User approval
        approved = await self._await_user_approval(plan)

        return DecompositionResult(plan=plan, approved=approved)
```

### Activity Boundaries

Each transform maps to one or more Temporal activities:

| Activity | Timeout | Retry | Heartbeat |
|----------|---------|-------|-----------|
| `classify_request` | 30s | 2 attempts | — |
| `generate_clarifications` | 60s | 2 attempts | — |
| `generate_goal_statement` | 60s | 2 attempts | — |
| `first_pass_decompose` | 5min | 3 attempts | 60s |
| `split_node` | 2min | 2 attempts | — |
| `check_atomicity` | 30s | 2 attempts | — |
| `analyze_dependencies` | 2min | 2 attempts | — |
| `generate_acceptance_criteria` | 60s | 2 attempts | — |
| `run_deterministic_checks` | 5s | 1 attempt | — |
| `run_adversarial_judge` | 5min | 2 attempts | 60s |
| `generate_dot` | 5s | 1 attempt | — |
| `persist_plan_version` | 5s | 2 attempts | — |

### Human Interaction Pattern

All human interaction uses the same Temporal signal/wait pattern:

```python
async def _await_user_input(self, prompt: UserPrompt) -> UserResponse:
    """Emit a prompt and wait for user response via signal."""
    # Emit the prompt (the CLI or API polls for pending prompts)
    await workflow.execute_activity(
        "emit_user_prompt", prompt, ...)

    # Wait for response signal
    await workflow.wait_condition(
        lambda: len(self._user_responses) > 0,
        timeout=timedelta(hours=72),  # 3 days max
    )
    return self._user_responses.pop(0)
```

The CLI or API client polls for pending prompts and presents them to the user as structured forms. The user's response is sent back as a signal.

### Cross-Workflow Sub-Plans

When a node in a software plan requires research (or vice versa), the system spawns a **child DecompositionWorkflow** for that sub-plan:

```python
sub_plan_result = await workflow.execute_child_workflow(
    DecompositionWorkflow.run,
    DecompositionInput(
        user_request=node.description,
        workflow_type=node.workflow_type,
        parent_plan_id=plan.plan_id,
        parent_node_id=node.node_id,
    ),
    id=f"decompose-sub-{node.node_id}",
)
```

The child workflow goes through the full pipeline including user approval. The parent workflow waits for the child to complete.

---

## DOT Visualization

A pure function converts PlanDAG to Graphviz DOT syntax for human review:

```python
def plan_to_dot(plan: PlanDAG) -> str:
    """Convert a PlanDAG to Graphviz DOT syntax."""
```

Node styling by execution type:

- `LLM_CALL` → blue box
- `HUMAN_ACTION` → orange hexagon
- `DETERMINISTIC` → green parallelogram

Edge styling by type:

- `DEPENDS_ON` → solid arrow
- `PARENT_CHILD` → dashed arrow

Container (non-leaf) nodes are rendered as subgraph clusters.

The DOT output is rendered to SVG via the `graphviz` Python package. Both JSON and SVG are presented to the user at approval time.

---

## Model Routing

Each transform specifies its capability tier. The system resolves the tier to a concrete model via the existing `ModelConfig` + `resolve_model()` mechanism. Multi-provider support is a future extension — the architecture supports it because each activity receives a model name string, not a provider instance.

| Transform | Tier | Rationale |
|-----------|------|-----------|
| Classify | CLASSIFICATION | Fast, cheap, low-stakes |
| Clarify | GENERATION | Needs good question formulation |
| Goal Statement | GENERATION | Synthesis, not reasoning |
| First Pass Decompose | REASONING | Hardest step, highest quality needed |
| Recursive Split | REASONING | Structural decisions |
| Atomicity Check | CLASSIFICATION | Binary yes/no judgment |
| Dependency Analysis | REASONING | Requires understanding relationships |
| Acceptance Criteria | GENERATION | Writing, not reasoning |
| Adversarial Judges | REASONING | Must find real weaknesses |

---

## Relationship to Existing Planner

This system **replaces** the current planner (`activities/planner.py`). The existing planner's responsibilities map to this design as follows:

| Current | New |
|---------|-----|
| `build_planner_system_prompt()` | `decompose.prompt.j2` template |
| `build_planner_user_prompt()` | `split.prompt.j2` template |
| `assemble_planner_context()` | `classify_request` + `first_pass_decompose` activities |
| `call_planner()` | `first_pass_decompose` + `split_node` activities |
| `PlanStep` model | `PlanNode` model (richer, DAG-aware) |
| `Plan` model | `PlanDAG` model (versioned, with edges) |
| `DomainConfig.planner_domain_instruction` | Per-workflow `decompose.prompt.j2` |
| `eval/judge.py` | Adversarial review with multiple personas |
| `eval/deterministic.py` | Extended deterministic checks (step 8) |

The existing `ForgeTaskWorkflow` execution pipeline (steps, retries, fan-out) remains unchanged. The new `PlanDAG` produces `PlanNode` leaves that are translated into the existing execution model when the plan is approved and execution begins.

---

## Testing Strategy

### Unit Tests (Pure Functions)

- Template rendering with various inputs
- Deterministic check functions (all 9+ checks)
- `plan_to_dot()` conversion
- DAG cycle detection
- PlanDAG serialization/deserialization
- Persona selection by workflow type
- Consensus calculation (2-of-3 logic)

### Integration Tests (with Mock LLM)

- Full pipeline with mock LLM responses at each stage
- Clarification loop with simulated user responses
- Judge rejection → revision → approval cycle
- Cross-workflow sub-plan spawning
- Plan version persistence and retrieval

### Temporal Tests (with Test Server)

- Signal/wait for user interaction
- Activity timeout and retry behavior
- Child workflow spawning for sub-plans
- Workflow cancellation mid-pipeline

### Evaluation Framework

Extend the existing `eval/` framework:

- Eval cases that include expected goal statements, expected node counts, expected dependency structures
- Compare decomposition quality across model versions
- Regression detection for decomposition changes

---

## Open Questions

1. **Repo context in decomposition.** The current planner receives a repo map. Should the new decomposition pipeline also explore the repo during splitting (step 5), or is the repo map sufficient? The current Phase 7 exploration could be integrated.

2. **Plan execution translation.** How exactly do `PlanNode` leaves map back to the existing `ForgeTaskWorkflow` execution model? A translation layer is needed. This is intentionally deferred to a separate design doc.

3. **Batch mode for judges.** The 3 adversarial judges are independent and could use the batch API for cost savings. Should we support this, or is the latency penalty (waiting for batch completion) unacceptable for an interactive planning workflow?

4. **Plan resumption.** If the user closes their terminal mid-pipeline, can they resume? Temporal provides durability, but the CLI needs a "resume plan" command.

5. **Plan templates / recipes.** Should common plan shapes (e.g., "build a CRUD app") be pre-defined templates that the decomposition pipeline can start from, similar to Goose's recipe system?
