# Task Decomposition: Behavioral Scenarios

**Status:** Draft
**Date:** 2026-03-07

Gherkin-style scenarios for every stage of the decomposition pipeline described in [DECOMPOSITION.md](DECOMPOSITION.md).

---

## Feature: Request Classification

    Determine the workflow type (software, research, etc.) from the user's
    raw request, using the catalog of workflow template descriptions.

### Scenario: Unambiguous software request

    Given a user submits the request "write a program that uses WebGPU to draw a rotating cube"
    And the workflow template catalog contains "software" and "research"
    When the classify_request activity runs
    Then the result workflow_type is "software"
    And the result confidence is greater than 0.8
    And no clarification is needed for workflow type

### Scenario: Unambiguous research request

    Given a user submits the request "research the connection between aerobic exercise and ADHD symptoms"
    And the workflow template catalog contains "software" and "research"
    When the classify_request activity runs
    Then the result workflow_type is "research"
    And the result confidence is greater than 0.8

### Scenario: Ambiguous request triggers user choice

    Given a user submits the request "analyze the performance of our sorting algorithm"
    And the workflow template catalog contains "software", "research", and "code_review"
    When the classify_request activity runs
    And the result confidence is less than 0.6
    Then the workflow pauses for user input
    And the user is presented with a structured form listing the candidate workflow types
    And each option includes the workflow's description from description.md

### Scenario: Single workflow type available

    Given a user submits the request "write a REST API for user management"
    And the workflow template catalog contains only "software"
    When the classify_request activity runs
    Then the result workflow_type is "software"
    And the result confidence is 1.0
    And no clarification is needed for workflow type

### Scenario: Classification result is persisted

    Given a user submits any request
    When the classify_request activity completes
    Then a plan_version record is created with transform_name "classify"
    And the plan record status is "draft"

---

## Feature: Clarification Loop

    Ask structured questions to resolve ambiguity in the user's request.
    Loop until the LLM produces no more questions. Pause the workflow
    for each round of questions.

### Scenario: No clarification needed

    Given a classified request with workflow_type "software"
    And the user's request is "write a Python function that returns the Fibonacci sequence up to n"
    When the generate_clarifications activity runs
    Then the result is an empty question list
    And the pipeline advances to goal statement

### Scenario: Single round of clarification

    Given a classified request with workflow_type "software"
    And the user's request is "write a program that uses WebGPU to draw a rotating cube"
    When the generate_clarifications activity runs
    Then the result contains at least one ClarificationQuestion
    And each question has a question_type of "choice", "text", or "confirm"
    And each question has a non-null default
    And each question has an importance of "required", "recommended", or "optional"
    And the workflow pauses via workflow.wait_condition
    When the user responds via signal with answers for each question
    Then the answers are stored in the clarifications table
    And the generate_clarifications activity runs again with the answers included
    And if the result is an empty question list the pipeline advances

### Scenario: Multiple rounds of clarification

    Given a classified request with workflow_type "research"
    And the user's request is "research quantum computing"
    When the generate_clarifications activity runs
    Then the result contains questions about scope, depth, and audience
    When the user responds with answers
    And the generate_clarifications activity runs again
    Then the result contains follow-up questions based on the first answers
    When the user responds with answers to the follow-up questions
    And the generate_clarifications activity runs again
    Then the result is an empty question list
    And the pipeline advances to goal statement

### Scenario: User accepts defaults

    Given the generate_clarifications activity produces questions with defaults
    When the user responds accepting all defaults without modification
    Then each answer is recorded as the default value
    And the pipeline advances normally

### Scenario: Clarification questions include rationale

    Given the generate_clarifications activity runs
    Then every ClarificationQuestion in the result has a non-empty rationale field
    And the rationale explains why the answer affects the plan

---

## Feature: Goal Statement

    Synthesize the user's request and clarification answers into a precise,
    unambiguous goal statement. The user must confirm or revise.

### Scenario: User approves goal statement

    Given the clarification loop is complete
    And the user's request is "write a program that uses WebGPU to draw a rotating cube"
    And the user answered "TypeScript" for language and "browser" for platform
    When the generate_goal_statement activity runs
    Then the result is a single goal statement string
    And the goal statement incorporates all clarification answers
    And the workflow pauses for user confirmation
    When the user responds with "approve"
    Then the goal statement is stored in the plans table
    And a plan_version record is created with transform_name "goal_statement"
    And the pipeline advances to first pass decomposition

### Scenario: User revises goal statement

    Given the generate_goal_statement activity has produced a goal statement
    And the workflow has paused for user confirmation
    When the user responds with "revise: should also support keyboard controls"
    Then the pipeline returns to step 2 (clarify) with the user's feedback
    And the user's feedback is included in the next generate_clarifications call

### Scenario: Goal statement is specific and testable

    Given the generate_goal_statement activity runs
    Then the goal statement does not contain vague words like "good", "nice", "proper"
    And the goal statement specifies measurable outcomes where possible
    And the goal statement references specific technologies mentioned in clarification answers

---

## Feature: First Pass Decomposition

    Produce a broad decomposition of the goal into ~3-7 top-level nodes.
    Nodes at this stage may be containers that need further splitting.

### Scenario: Software task first pass

    Given an approved goal statement for a WebGPU rotating cube program
    And the workflow_type is "software"
    When the first_pass_decompose activity runs using the software/decompose.prompt.j2 template
    Then the result is a PlanDAG with 3 to 7 top-level nodes
    And at least one node addresses data models or setup
    And at least one node addresses core logic
    And each node has a title and description
    And a plan_version record is created with transform_name "first_pass"

### Scenario: Research task first pass

    Given an approved goal statement for researching aerobic exercise and ADHD
    And the workflow_type is "research"
    When the first_pass_decompose activity runs using the research/decompose.prompt.j2 template
    Then the result is a PlanDAG with 3 to 7 top-level nodes
    And at least one node addresses source gathering
    And at least one node addresses synthesis or analysis

### Scenario: First pass nodes may be non-leaf

    Given the first_pass_decompose activity has run
    Then some nodes may have is_leaf set to false
    And non-leaf nodes have an empty acceptance_criteria list
    And the pipeline advances to recursive split

### Scenario: Domain-specific template is used

    Given the workflow_type is "software"
    When the first_pass_decompose activity runs
    Then the LLM prompt includes content from software/decompose.prompt.j2
    And the prompt includes the domain_guidance block from the template

---

## Feature: Recursive Split

    Decompose non-leaf nodes into children until all nodes pass the
    atomicity test. Each child is confirmed atomic by a separate LLM call.

### Scenario: Non-leaf node is split into atomic children

    Given a PlanDAG with a non-leaf node "implement rendering pipeline"
    When the split_node activity runs for that node
    Then the node receives 2 or more children
    And PARENT_CHILD edges are created from each child to the parent
    When the check_atomicity activity runs for each child
    And all children are confirmed atomic
    Then each child has is_leaf set to true
    And the parent node retains is_leaf as false

### Scenario: Child fails atomicity check and is split further

    Given a PlanDAG with a non-leaf node "build the backend"
    When the split_node activity runs and produces child "implement API and database"
    And the check_atomicity activity runs for "implement API and database"
    And the atomicity judge says "not atomic: this involves two independent concerns"
    Then "implement API and database" is marked as non-leaf
    And the split_node activity runs again for "implement API and database"
    And the process repeats until all descendants are atomic

### Scenario: Cross-workflow node detected

    Given a PlanDAG for a software task
    And the split_node activity produces a child "research which WebGPU library has the best documentation"
    And the LLM marks that child with workflow_type "research"
    Then the child is flagged as a cross-workflow node
    And a child DecompositionWorkflow is spawned for the research sub-plan
    And the parent workflow waits for the child workflow to complete
    And the child workflow goes through the full pipeline including user approval

### Scenario: Ambiguity during splitting triggers clarification

    Given a non-leaf node with an ambiguous description
    When the split_node activity runs
    And the LLM produces a ClarificationQuestion instead of children
    Then the workflow pauses for user input
    And the user's answer is fed back into the split_node activity
    And splitting resumes

### Scenario: Recursive split terminates

    Given a PlanDAG with multiple levels of non-leaf nodes
    When the recursive split loop runs
    Then every pass reduces the number of non-leaf nodes
    And the loop terminates when all nodes are either leaves or containers whose children are all leaves

### Scenario: Container nodes have at least 2 children

    Given any non-leaf node after splitting
    Then the node has at least 2 children
    And if only one child would result, the parent is converted to a leaf instead

---

## Feature: Dependency Analysis

    Review all leaf nodes and add DEPENDS_ON edges with rationale.
    Validate the result is acyclic.

### Scenario: Dependencies identified between leaf nodes

    Given a PlanDAG with leaf nodes "create database schema" and "implement data access layer"
    When the analyze_dependencies activity runs
    Then a DEPENDS_ON edge is created from "implement data access layer" to "create database schema"
    And the edge has a non-empty rationale
    And a plan_version record is created with transform_name "dependency_analysis"

### Scenario: Independent nodes have no dependency edges

    Given a PlanDAG with leaf nodes "write unit tests for module A" and "write unit tests for module B"
    And the two modules are independent
    When the analyze_dependencies activity runs
    Then no DEPENDS_ON edge exists between the two nodes

### Scenario: Cyclic dependency detected and rejected

    Given the analyze_dependencies activity produces edges that form a cycle
    When the deterministic acyclic validation runs
    Then the validation fails
    And the system loops back to the analyze_dependencies activity with the cycle details
    And the LLM is instructed to break the cycle

### Scenario: Dependency edges reference existing nodes only

    Given the analyze_dependencies activity runs
    Then every DEPENDS_ON edge references source_id and target_id values that exist in the PlanDAG nodes dict

---

## Feature: Acceptance Criteria

    Add specific, testable "definition of done" conditions to every leaf node.
    Criteria generation is parallelized across leaves.

### Scenario: Software leaf receives acceptance criteria

    Given a leaf node with execution_type "llm_call" and workflow_type "software"
    And the node description is "implement the vertex shader for the rotating cube"
    When the generate_acceptance_criteria activity runs using software/criteria.prompt.j2
    Then the node's acceptance_criteria list is non-empty
    And at least one criterion references a testable condition
    And criteria are specific to the node's description

### Scenario: Human action leaf receives acceptance criteria

    Given a leaf node with execution_type "human_action"
    And the node description is "obtain an API key for the WebGPU validation service"
    When the generate_acceptance_criteria activity runs
    Then the acceptance_criteria include what artifact the human must produce
    And the acceptance_criteria include how completion is verified

### Scenario: Deterministic leaf receives acceptance criteria

    Given a leaf node with execution_type "deterministic"
    And the node description is "run ruff lint on all Python files"
    When the generate_acceptance_criteria activity runs
    Then the acceptance_criteria include the expected exit code or output condition

### Scenario: Criteria generation is parallelized

    Given a PlanDAG with 10 leaf nodes
    When the acceptance criteria step runs
    Then up to 10 generate_acceptance_criteria activities are started concurrently
    And the step completes when all activities have returned

### Scenario: Criteria include measurable conditions

    Given any leaf node after criteria generation
    Then each acceptance criterion is phrased as a verifiable condition
    And no criterion uses vague language like "works correctly" or "is complete"

---

## Feature: Deterministic Checks

    Validate structural properties of the PlanDAG without an LLM.
    These are pure functions.

### Scenario: Valid DAG passes all checks

    Given a PlanDAG where:
      - The graph is acyclic
      - All node IDs are unique UUIDs
      - All edge references point to existing nodes
      - Every leaf has at least one acceptance criterion
      - Every leaf has an execution_type
      - Every non-root node has a PARENT_CHILD edge
      - Container nodes have at least 2 children
    When the run_deterministic_checks activity runs
    Then the result has all_passed set to true
    And every check has status "pass"

### Scenario: Cyclic graph fails validation

    Given a PlanDAG with a DEPENDS_ON cycle between nodes A -> B -> C -> A
    When the run_deterministic_checks activity runs
    Then the "acyclic" check has status "fail"
    And the check details list the cycle path
    And all_passed is false

### Scenario: Leaf without acceptance criteria fails

    Given a PlanDAG with a leaf node that has an empty acceptance_criteria list
    When the run_deterministic_checks activity runs
    Then the "leaf_has_criteria" check has status "fail"
    And the check details list the offending node_id

### Scenario: Orphan node fails validation

    Given a PlanDAG with a node that has no PARENT_CHILD edge and is not the root
    When the run_deterministic_checks activity runs
    Then the "no_orphans" check has status "fail"

### Scenario: Edge referencing nonexistent node fails

    Given a PlanDAG with an edge whose target_id does not exist in the nodes dict
    When the run_deterministic_checks activity runs
    Then the "valid_edge_references" check has status "fail"

### Scenario: Container with single child fails

    Given a PlanDAG with a non-leaf node that has exactly 1 child
    When the run_deterministic_checks activity runs
    Then the "container_min_children" check has status "fail"

### Scenario: Deterministic failure triggers revision

    Given the run_deterministic_checks activity returns all_passed as false
    Then the system loops back to step 5 (recursive split) with the failure details
    And this counts as one of the 3 allowed revision attempts
    And a plan_version record is created with transform_name "deterministic_revision"

---

## Feature: Adversarial Review

    Three LLM judges independently evaluate the plan. Each argues against
    before arguing for, then votes APPROVE or REJECT. 2-of-3 consensus
    required. Up to 3 rounds before escalation.

### Scenario: Persona selection for software workflow

    Given a PlanDAG with workflow_type "software"
    When the adversarial review step selects personas
    Then the selected personas are "Expert Skeptic", "Completeness Auditor", and "Dependency Critic"

### Scenario: Persona selection for research workflow

    Given a PlanDAG with workflow_type "research"
    When the adversarial review step selects personas
    Then the selected personas are "Detail Analyst", "Completeness Auditor", and "Scope Guardian"

### Scenario: Persona selection for generic workflow

    Given a PlanDAG with workflow_type "generic"
    When the adversarial review step selects personas
    Then the selected personas are "Completeness Auditor", "Scope Guardian", and "Feasibility Assessor"

### Scenario: Judge argues against before arguing for

    Given a judge with the "Expert Skeptic" persona
    When the run_adversarial_judge activity runs
    Then the result contains a non-empty arguments_against section
    And the result contains a non-empty arguments_for section
    And arguments_against appears before arguments_for in the response structure

### Scenario: Judge produces scores for all criteria

    Given any judge persona
    When the run_adversarial_judge activity runs
    Then the result contains scores for all 5 criteria:
      - COMPLETENESS
      - GRANULARITY
      - FEASIBILITY
      - DEPENDENCY_CORRECTNESS
      - ACCEPTANCE_CRITERIA_QUALITY
    And each score is between 1 and 5 inclusive
    And each score has a non-empty rationale

### Scenario: 2-of-3 judges approve

    Given 3 judge activities run concurrently
    And 2 judges return verdict "approve"
    And 1 judge returns verdict "reject"
    Then the consensus is APPROVED
    And the plan advances to user approval
    And all 3 reviews are stored in the judge_reviews table

### Scenario: 3-of-3 judges approve

    Given 3 judge activities run concurrently
    And all 3 judges return verdict "approve"
    Then the consensus is APPROVED
    And the plan advances to user approval

### Scenario: 2-of-3 judges reject on first round

    Given 3 judge activities run concurrently
    And 2 judges return verdict "reject" with required_changes
    And this is round 1 of 3
    Then the consensus is REJECTED
    And the required_changes from both rejecting judges are collected
    And the system loops back to step 5 (recursive split) with revision instructions
    And the revision instructions include the collected required_changes
    And a new plan_version is created with transform_name "judge_revision"
    And the review round counter increments to 2

### Scenario: Judges reject through 3 rounds

    Given the judges reject the plan in round 1
    And the revised plan is rejected again in round 2
    And the revised plan is rejected again in round 3
    Then the workflow stops
    And the plan status is set to "escalated"
    And the user receives all judge feedback from all 3 rounds
    And the workflow does not attempt further automatic revision

### Scenario: Judges run concurrently

    Given the adversarial review step begins
    Then 3 run_adversarial_judge activities are started in parallel
    And the step waits for all 3 to complete before evaluating consensus

### Scenario: Judge reviews are persisted

    Given any judge activity completes
    Then a judge_reviews record is created with:
      - the correct plan_id and version
      - the round number
      - the persona name
      - the arguments_against and arguments_for text
      - the verdict
      - the required_changes (if reject)
      - the serialized scores

---

## Feature: User Approval

    Present the final plan to the user for approval. If rejected,
    the workflow terminates. Revision requires a new explicit request.

### Scenario: User approves the plan

    Given a PlanDAG that passed adversarial review
    When the plan is presented to the user
    Then the user receives the PlanDAG as formatted JSON
    And the user receives a rendered DOT diagram as SVG
    And the user receives a summary with node count, leaf count, and workflow types
    And the workflow pauses for user response via signal
    When the user responds with "approve"
    Then the plan status is set to "approved"
    And the workflow returns a DecompositionResult with approved set to true

### Scenario: User rejects the plan

    Given a PlanDAG that passed adversarial review
    And the user receives the plan
    When the user responds with "reject"
    Then the plan status is set to "rejected"
    And the workflow returns a DecompositionResult with approved set to false
    And no automatic revision is attempted

### Scenario: User approval times out

    Given the workflow is waiting for user approval
    And the user does not respond within 72 hours
    Then the workflow.wait_condition raises a timeout
    And the plan status is set to "timed_out"
    And the workflow returns a DecompositionResult with approved set to false

---

## Feature: Plan Versioning

    Every mutation to the plan creates a new version record.
    All versions are retained for auditability.

### Scenario: Version chain through the pipeline

    Given a user's request goes through the full pipeline
    Then plan_versions records exist for at least these transform_names:
      - "classify"
      - "goal_statement"
      - "first_pass"
      - "recursive_split" (one or more)
      - "dependency_analysis"
      - "acceptance_criteria"
    And each version's parent_version points to the previous version number
    And version numbers are monotonically increasing

### Scenario: Judge revision creates additional versions

    Given the adversarial review rejects the plan in round 1
    Then a plan_version with transform_name "judge_revision" is created
    And the revised plan goes through steps 5-8 again
    And each of those steps creates additional version records

### Scenario: Version JSON is complete and parseable

    Given any plan_version record
    Then the plan_json column contains a valid JSON string
    And the JSON deserializes to a valid PlanDAG model
    And the PlanDAG contains all nodes and edges for that point in time

---

## Feature: Cross-Workflow Sub-Plans

    A node in one workflow type can require a different workflow type.
    This spawns a child DecompositionWorkflow with its own full pipeline.

### Scenario: Software plan spawns research sub-plan

    Given a software PlanDAG with a node marked workflow_type "research"
    And the node description is "research which WebGPU library has the best documentation"
    When the recursive split encounters this node
    Then a child DecompositionWorkflow is spawned with:
      - user_request set to the node description
      - workflow_type set to "research"
      - parent_plan_id set to the current plan's plan_id
      - parent_node_id set to the node's node_id
    And the parent workflow waits for the child to complete

### Scenario: Child sub-plan requires independent user approval

    Given a child DecompositionWorkflow is spawned for a research sub-plan
    Then the child workflow goes through all 10 pipeline steps
    And the child workflow pauses for user approval independently
    And the user must approve the sub-plan before the parent can continue

### Scenario: Child sub-plan is stored separately

    Given a child DecompositionWorkflow completes
    Then a separate plan record exists in the plans table
    And the child plan's versions are stored independently
    And the parent plan node references the child plan_id in its context dict

---

## Feature: Workflow Templates

    Jinja2 templates organized by workflow type provide domain-specific
    prompt guidance for each pipeline step.

### Scenario: Template loading by workflow type

    Given the workflow_type is "software"
    When the first_pass_decompose activity needs a prompt
    Then the template at software/decompose.prompt.j2 is loaded
    And the template extends _shared/decompose_base.prompt.j2

### Scenario: Template variables are populated

    Given a decompose template with variables {{ goal }} and {{ user_request }}
    When the template is rendered
    Then {{ goal }} is replaced with the confirmed goal statement
    And {{ user_request }} is replaced with the original user request
    And {{ workflow_type }} is replaced with the current workflow type

### Scenario: Missing template falls back to shared

    Given a workflow type that does not have a custom clarify.prompt.j2
    When the generate_clarifications activity needs a prompt
    Then the template at _shared/clarify_base.prompt.j2 is used

### Scenario: New workflow type added by template alone

    Given a new directory "fact_check/" is created in workflow_templates/
    And it contains description.md and the required prompt templates
    When a user submits a request classified as "fact_check"
    Then the decomposition pipeline uses the fact_check templates
    And no Python code changes are required

---

## Feature: DOT Visualization

    Convert PlanDAG to Graphviz DOT syntax and render to SVG.

### Scenario: Node styling by execution type

    Given a PlanDAG with nodes of each execution type
    When plan_to_dot converts the DAG
    Then LLM_CALL nodes are rendered as blue boxes
    And HUMAN_ACTION nodes are rendered as orange hexagons
    And DETERMINISTIC nodes are rendered as green parallelograms

### Scenario: Edge styling by type

    Given a PlanDAG with DEPENDS_ON and PARENT_CHILD edges
    When plan_to_dot converts the DAG
    Then DEPENDS_ON edges are rendered as solid arrows
    And PARENT_CHILD edges are rendered as dashed arrows

### Scenario: Container nodes rendered as clusters

    Given a PlanDAG with a non-leaf node containing 3 children
    When plan_to_dot converts the DAG
    Then the non-leaf node is rendered as a subgraph cluster
    And the cluster contains its 3 child nodes

### Scenario: DOT output is valid Graphviz syntax

    Given any PlanDAG
    When plan_to_dot produces output
    Then the output parses without error by the graphviz Python package
    And the output renders to a valid SVG

---

## Feature: Model Routing

    Each transform uses a capability tier that resolves to a concrete model.

### Scenario: Classification uses lightweight model

    Given the classify_request activity runs
    Then the model tier is CLASSIFICATION
    And the resolved model is a fast, inexpensive model

### Scenario: Decomposition uses reasoning model

    Given the first_pass_decompose activity runs
    Then the model tier is REASONING
    And the resolved model supports extended thinking

### Scenario: Judges use reasoning model

    Given the run_adversarial_judge activity runs
    Then the model tier is REASONING
    And the resolved model is capable of nuanced analysis

### Scenario: Criteria generation uses generation model

    Given the generate_acceptance_criteria activity runs
    Then the model tier is GENERATION
    And the resolved model balances quality and cost

---

## Feature: Human Interaction Pattern

    All human interaction uses the Temporal signal/wait pattern.
    The CLI polls for pending prompts and presents structured forms.

### Scenario: Prompt emitted and response received

    Given the workflow needs user input
    When the emit_user_prompt activity runs with a UserPrompt payload
    Then the prompt is stored for the CLI to poll
    And the workflow pauses via workflow.wait_condition
    When the user responds via the user_response signal
    Then the workflow resumes with the UserResponse

### Scenario: Structured form with defaults

    Given a clarification prompt with 3 questions
    Then each question is presented as a form field
    And "choice" questions show a dropdown with options
    And "text" questions show an input field
    And "confirm" questions show yes/no
    And all fields with defaults are pre-populated

### Scenario: Response timeout

    Given the workflow is waiting for user input
    And the user does not respond within 72 hours
    Then the workflow.wait_condition raises a timeout
    And the workflow terminates gracefully with a timeout status

---

## Feature: Plan Database Persistence

    A separate SQLite database stores all plan data including versions,
    clarifications, and judge reviews.

### Scenario: Plan record created at pipeline start

    Given a user submits a new decomposition request
    When the workflow begins
    Then a plans table record is created with:
      - a new UUID as plan_id
      - status "draft"
      - the user_request text
      - created_at set to the current timestamp

### Scenario: Clarification answers persisted

    Given the user answers a clarification question
    Then the clarifications table record is updated with:
      - the answer text
      - answered_at set to the current timestamp

### Scenario: Plan status transitions

    Given a plan goes through the full pipeline
    Then the plan status transitions through:
      - "draft" (created)
      - "reviewing" (entered adversarial review)
      - "approved" (user approved) OR "rejected" (user rejected) OR "escalated" (judges failed 3 rounds)

### Scenario: Database location follows XDG spec

    Given the XDG_STATE_HOME environment variable is set to "/tmp/test-xdg"
    Then the plan database is created at "/tmp/test-xdg/forge/plans.db"

    Given XDG_STATE_HOME is not set
    Then the plan database is created at "~/.local/state/forge/plans.db"

---

## Feature: Error Handling and Recovery

    The pipeline handles activity failures, LLM errors, and timeouts
    gracefully without losing plan state.

### Scenario: LLM activity fails and retries

    Given the first_pass_decompose activity fails with a retryable error
    Then Temporal retries the activity up to 3 times
    And no plan state is corrupted
    And the plan_version from the successful retry is persisted

### Scenario: LLM returns invalid structure

    Given the split_node activity returns a response that does not parse as valid PlanNode children
    Then the activity raises an error
    And Temporal retries with the retry policy
    And the invalid response is logged for debugging

### Scenario: Workflow resumes after worker restart

    Given a DecompositionWorkflow is paused waiting for user input
    And the Temporal worker process restarts
    When the user sends a response signal
    Then the workflow resumes from where it paused
    And no plan data is lost

### Scenario: Deterministic check failure does not count toward judge revision limit

    Given the deterministic checks fail
    And the system loops back to recursive split
    Then this loop does not count toward the 3-round adversarial review limit
    And the adversarial review round counter remains unchanged
