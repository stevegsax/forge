"""forge-contracts — the shared SPI surface between Forge (the platform) and its
consumer apps (e.g. OCR).

This package holds only contract-level code that both sides import: wire models,
queue/namespace/signal-name constants, the S3 blob-access library, the Temporal
connection helper, and shared survivable-write primitives. It must not import
``forge`` or any consumer package — the dependency arrows point inward, here.
"""

__version__ = "0.1.1"
