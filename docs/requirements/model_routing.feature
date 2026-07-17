@model-routing @phase-11
Feature: Model Routing
  The orchestrator routes LLM calls to concrete models based on capability tiers.
  Four tiers map to default models and can be overridden per-step or via CLI flags.

  Background:
    Given the orchestrator is configured with default model routing

  # --- Capability Tiers ---

  @critical
  Scenario Outline: Default model assignment per capability tier
    When the orchestrator resolves a model for the "<tier>" capability tier
    Then the resolved model is "<model>"

    Examples:
      | tier             | model                                   |
      | reasoning        | anthropic:claude-opus-4-8               |
      | generation       | anthropic:claude-sonnet-5               |
      | summarization    | anthropic:claude-sonnet-5               |
      | classification   | anthropic:claude-haiku-4-5              |

  @critical
  Scenario: Resolve model returns provider-prefixed identifier
    When the orchestrator resolves a model for the "generation" capability tier
    Then the result includes a provider prefix in "provider:model" format

  # --- Tier Usage ---

  @standard
  Scenario: Planning uses the reasoning tier
    Given a task with planning enabled
    When the orchestrator calls the planner
    Then it uses the model assigned to the "reasoning" tier

  @standard
  Scenario: Single-step generation uses the generation tier
    Given a task in single-step mode
    When the orchestrator calls the LLM for code generation
    Then it uses the model assigned to the "generation" tier

  @standard
  Scenario: Exploration uses the classification tier
    Given a task with LLM-guided exploration enabled
    When the orchestrator calls the exploration LLM
    Then it uses the model assigned to the "classification" tier

  @standard
  Scenario: Knowledge extraction uses the summarization tier
    When the orchestrator runs the knowledge extraction workflow
    Then it uses the model assigned to the "summarization" tier

  @standard
  Scenario: Conflict resolution uses the reasoning tier
    Given a fan-out step with file conflicts
    When the orchestrator calls the conflict resolution LLM
    Then it uses the model assigned to the "reasoning" tier

  @standard
  Scenario: Sanity checks use the reasoning tier
    Given a planned task with sanity checks enabled
    When a sanity check is triggered
    Then it uses the model assigned to the "reasoning" tier

  # --- Per-Step Override ---

  @critical
  Scenario: Plan step overrides capability tier
    Given a plan step with capability_tier set to "reasoning"
    When the orchestrator executes that step
    Then it uses the model assigned to the "reasoning" tier instead of the default "generation" tier

  @standard
  Scenario: Plan step without capability tier override defaults to generation
    Given a plan step with no capability_tier specified
    When the orchestrator executes that step
    Then it uses the model assigned to the "generation" tier

  # --- CLI Override ---

  @critical
  Scenario Outline: CLI flag overrides default model for a tier
    Given the CLI flag "<flag>" is set to "custom:model-v1"
    When the orchestrator resolves a model for the "<tier>" capability tier
    Then the resolved model is "custom:model-v1"

    Examples:
      | flag                  | tier             |
      | --reasoning-model     | reasoning        |
      | --generation-model    | generation       |
      | --summarization-model | summarization    |
      | --classification-model| classification   |

  @standard
  Scenario: Non-overridden tiers keep defaults when one tier is overridden
    Given the CLI flag "--reasoning-model" is set to "custom:model-v1"
    When the orchestrator resolves a model for the "generation" capability tier
    Then the resolved model is "anthropic:claude-sonnet-5"
