"""Composable workflow blocks for Forge.

One module per repeated workflow shape. ``step`` owns the universal step
pipeline (assemble → LLM → write → validate → act) that ``forge.workflows``
used to carry three hand-synchronized copies of; the gather pipeline follows in
T5.3.

Modules here run inside a Temporal workflow: they call
``workflow.execute_activity`` and must stay deterministic. Import them through
``workflow.unsafe.imports_passed_through()``.
"""
