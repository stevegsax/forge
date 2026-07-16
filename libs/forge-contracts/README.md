# forge-contracts

The shared SPI surface between **Forge** (the Temporal platform) and its consumer
apps (e.g. **OCR**). Both sides depend on this package; neither imports the other.

Holds contract-level code only:

- `s3_blobs` — S3 blob access (blob I/O can't cross Temporal queues; both sides do
  direct S3 I/O and address blobs by key carried in contract messages).
- `types` — shared SQLAlchemy column types (`UTCDateTime`).
- *(incoming)* wire models (`BatchResult`, the batch submit-SPI request,
  `BatchJobStatus`, the `batch_jobs` read model), queue/namespace/signal-name
  constants, the Temporal connection helper, and the `persist_block` primitive.

See `forge/development-plans/separate-ocr-into-its-own-repo.md` for the design and
build sequence.
