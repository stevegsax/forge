"""Pure prompt-builder functions for pbook's structured-output LLM calls.

These functions have no Temporal, no I/O, and no state — workflows can
import them via ``workflow.unsafe.imports_passed_through()`` and call
them in workflow body to assemble system/user prompts before invoking
:func:`pbook.workflow_steps.llm.llm_chat`.

Splitting the prompts out of the activity modules makes them
independently testable and avoids dragging the activity-side imports
(Temporal, store, embeddings) into the workflow sandbox.
"""
