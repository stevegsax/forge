# The Universal Workflow Step

Every operation in Forge -- code generation, planning, exploration, conflict resolution, knowledge extraction, sanity checking -- follows the same five-phase pattern:

```
Construct --> Send --> Receive --> Serialize --> Transition
```

This is not an implementation shortcut. It is the central design decision that shapes everything else in the system. Understanding why this pattern exists, and what it makes possible, is prerequisite to understanding Forge.


## The pattern

A Forge workflow step proceeds through five phases, each implemented as a separate Temporal activity:

1. **Construct.** Assemble the complete prompt -- system prompt and user prompt -- from task metadata, discovered context, playbooks, and (on retries) previous error output.
2. **Send.** Package the prompt into an API request and submit it to the selected model.
3. **Receive and serialize.** Extract the structured response, validate it against a Pydantic schema, and write the results (new files or edits to existing files) to the git worktree.
4. **Validate.** Run deterministic checks -- linting, formatting, optionally tests -- against the written output.
5. **Transition.** Map the validation results to one of three outcome signals: `SUCCESS`, `FAILURE_RETRYABLE`, or `FAILURE_TERMINAL`.

The workflow orchestrates these activities in sequence. On `SUCCESS`, it commits and moves on. On `FAILURE_RETRYABLE`, it loops back to Construct with the error output injected into the prompt. On `FAILURE_TERMINAL`, it stops and reports the failure.

This is the entire control loop. There is no other path through the system.


## Why one pattern for everything

Most LLM orchestrators differentiate between "types" of work at the workflow level. Planning gets one workflow. Code generation gets another. Review gets a third. Each workflow has its own control flow, its own retry logic, its own way of handling failures.

Forge takes the opposite approach. The differentiation between types of work lives entirely in the prompt and context, not in the workflow machinery. Planning is a workflow step where the system prompt says "you are a planner" and the response schema expects a `Plan`. Code generation is a workflow step where the system prompt says "you are a code generation assistant" and the response schema expects files and edits. Conflict resolution is a workflow step where the system prompt describes two conflicting versions and asks for a merge.

The workflow engine does not know or care what the LLM is being asked to do. It knows how to construct a prompt, send it, process the response, validate the output, and decide what happens next. That is all it needs to know.

This has a concrete consequence: adding a new task type to Forge requires writing new prompts, not new workflows. The orchestration machinery, the retry logic, the validation pipeline, the observability instrumentation, the batch processing support -- all of it comes for free because it operates on the same five-phase pattern. See the [reference](../reference/workflow-step.md) for the exact activity signatures and data models.


## Contrast with chat-loop orchestrators

The dominant pattern in LLM orchestration is the chat loop. The LLM receives a task, decides what tool to call, receives the tool result, decides what to call next, and so on until it decides it is done. The LLM owns the control loop.

Forge inverts this. The orchestrator -- not the LLM -- owns the control loop. The LLM receives a complete prompt, produces a complete response, and the orchestrator evaluates what to do next. The LLM never decides "I need to call a tool now" mid-generation. It never decides "I am done" or "I should retry." Those decisions belong to the orchestrator.

This inversion has several consequences.

**The LLM cannot get stuck in a loop.** In a chat-loop orchestrator, the LLM can enter degenerate cycles -- calling the same tool repeatedly, oscillating between two strategies, or generating increasingly confused output as its context window fills with failed attempts. In Forge, each LLM call is independent. The orchestrator decides whether to retry, and each retry gets a fresh prompt with targeted error feedback rather than an ever-growing conversation history.

**The orchestrator can reason about progress.** Because the orchestrator controls the loop, it can count attempts, measure token spend, evaluate validation results, and make informed decisions about whether to continue. A chat-loop orchestrator can only observe what the LLM chooses to do.

**The LLM call is a pure function of its inputs.** Given the same prompt, the same model produces the same distribution of outputs. There is no hidden state from prior turns. This makes individual calls reproducible and debuggable.


## The state machine analogy

A useful way to think about the universal workflow step is as a state machine where the states are fixed but the inputs vary.

The states are always the same: Construct, Send, Receive, Validate, Transition. The transitions between states are always the same: sequential through the five phases, then branch on the outcome signal. The retry loop is always the same: on `FAILURE_RETRYABLE`, loop back to Construct with error context.

What changes between different types of work is the input to the Construct phase: the role prompt, the output requirements, the context, the task description. And what changes between attempts of the same work is the error section appended to the prompt on retry.

This is why the Forge architecture document states that "the LLM call is the universal primitive." The state machine is fixed infrastructure. The prompts and context are the only variables.


## Batch compatibility

The five-phase pattern was designed around a specific constraint: compatibility with batch APIs.

Batch APIs (such as the Anthropic Batch API) accept a set of requests, process them asynchronously, and return results later. They do not support multi-turn conversations. Each request must be self-contained -- a complete prompt that can be processed independently of any other request.

Because every Forge LLM call is a standalone document completion, not a turn in a conversation, every call is naturally batch-compatible. The orchestrator submits the request to the batch API, the workflow pauses (via a Temporal signal wait), and when the batch result arrives, the workflow resumes at the Receive phase. The prompt construction is identical whether the call runs synchronously or in batch mode.

This is not a theoretical benefit. Batch processing reduces per-token costs by 50% on the Anthropic API. For a system that makes hundreds of LLM calls per task across planning, exploration, generation, and validation, the cost difference is significant.

A chat-loop orchestrator cannot use batch APIs without fundamental redesign, because the next request depends on the LLM's choice of tool call in the current turn. The universal workflow step avoids this problem by construction.


## Testability

Each phase of the workflow step is a separate Temporal activity with a defined input type and output type. This means each phase can be tested in isolation by constructing the input and asserting on the output.

Testing the Construct phase means providing task metadata and asserting that the assembled prompt contains the expected sections. Testing the Send phase means providing a prompt and asserting that the API call is well-formed. Testing the Validate phase means providing written files and asserting that the correct checks run. Testing the Transition phase means providing validation results and asserting the correct signal.

The end-to-end flow can be tested by mocking the LLM response and verifying that the orchestrator drives the correct sequence of activities. Because the LLM call is stateless, the mock only needs to return a valid response for the given prompt -- there is no conversation state to simulate.

For a concrete walkthrough of the workflow step in action, see the [Golden Path tutorial](../tutorials/golden-path.md).


## Observability

Because every operation follows the same pattern, the observability instrumentation is uniform. Every workflow step records the same metrics: the assembled prompt (stored in the SQLite observability store), the model name and token usage (on the Temporal result payload), the validation results, and the transition signal.

A developer debugging a failed planning step uses the same tools and the same mental model as a developer debugging a failed code generation step. The `forge status --verbose` command shows the same structure regardless of what the step was doing.

This uniformity extends to OpenTelemetry tracing. The span hierarchy is the same for every step: pipeline run, workflow instance, activity (context assembly, LLM call, validation, transition), individual API request. The spans carry the same attributes. The only difference is the content of the prompt and response.


## The transition vocabulary

The five-phase pattern terminates with one of three signals:

- **`SUCCESS`** -- All validation checks passed. The step is done.
- **`FAILURE_RETRYABLE`** -- Validation checks failed, but retry attempts remain. The step will be retried with error feedback.
- **`FAILURE_TERMINAL`** -- Either no retry attempts remain, or the failure is unrecoverable. The workflow stops.

This vocabulary is deliberately small. The orchestrator does not need to understand what went wrong in detail -- that information is captured in the validation results and fed back to the LLM on retry. The orchestrator only needs to know whether to proceed, retry, or stop.

Three additional signals (`new_tasks_discovered`, `blocked_on_human`, `blocked_on_sibling`) are defined in the design document for future dynamic task evolution but are not implemented in the current plan-then-execute architecture. The current system handles those situations through existing mechanisms: upfront planning eliminates task discovery, `FAILURE_TERMINAL` covers human escalation, and dependency ordering prevents sibling blocking. See the [reference](../reference/workflow-step.md) for the full `TransitionSignal` definition.


## The relationship between workflows and activities

Temporal enforces a strict boundary: workflows must be deterministic (no I/O, no randomness, no system calls), and all side effects happen in activities. This boundary maps naturally onto the universal workflow step.

The workflow is the state machine -- it sequences the five phases and implements the retry loop. The activities are the phases themselves -- context assembly reads files, the LLM call hits the API, write output touches the filesystem, validation runs subprocesses, and transition evaluation is a pure function of validation results.

This separation means that if an activity fails mid-execution (the worker crashes, the network drops), Temporal can retry the activity on another worker without re-running the entire workflow. The workflow's state -- which phase it is in, how many retries have been attempted, what the previous error was -- is preserved by Temporal's event sourcing.

For the complete list of activities, their input/output types, timeouts, and retry policies, see the [Universal Workflow Step reference](../reference/workflow-step.md).
