"""OCR application.

Document OCR via the Mistral batch API, built as a consumer of the Forge
platform. Runs its own Temporal worker on ``ocr-task-queue`` (same namespace as
Forge), owns its own ``ocr_``-prefixed tables in the shared database, and depends
only on ``forge-contracts`` — never on ``forge`` itself.
"""
