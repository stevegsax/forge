"""Composable workflow blocks for Forge.

One module per repeated workflow shape, so a shape exists once and the
workflow drivers in :mod:`forge.workflows` only decide which one runs next:

- ``step`` — the universal step pipeline (assemble → LLM → write → validate →
  act), which the drivers used to carry three hand-synchronized copies of;
- ``gather`` — the fan-out gather (start children, await with per-child failure
  isolation, merge, validate, maybe commit), formerly two copies;
- ``dispatch`` — the typed LLM call, five arms over one table;
- ``exploration`` — the LLM-guided context loop;
- ``transport`` — the batch submit/poll/fetch/parse choke point (D88);
- ``worktree`` — worktree removal and post-failure cleanup;
- ``host`` — the per-run dispatch state both drivers inherit.

Modules here run inside a Temporal workflow: they call
``workflow.execute_activity`` and must stay deterministic. Import them through
``workflow.unsafe.imports_passed_through()``.
"""
