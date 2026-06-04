# ocr

Document OCR application built on the **Forge** platform. Submits documents to the
Mistral batch OCR API, extracts text + images, and stores results.

- Own Temporal worker on `ocr-task-queue` (same namespace as Forge).
- Own `ocr_`-prefixed tables + Alembic chain in the **shared** database (`FORGE_DB_URL`).
- Depends only on `forge-contracts` (wire models, S3 blobs, Temporal connect helper,
  DB-engine helpers, constants) and `sax-llm` — never on `forge`.

See `forge/development-plans/separate-ocr-into-its-own-repo.md` for the migration
plan and design.
