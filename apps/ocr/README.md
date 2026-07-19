# ocr

Document OCR application built on the **Forge** platform: submits documents
to the Mistral batch OCR API via the platform's batch SPI, extracts
text + images, and stores results. A workspace member of the forge
monorepo at `apps/ocr` (D98).

- Own Temporal worker on `ocr-task-queue` (same namespace as Forge);
  cross-queue batch SPI calls to `forge-task-queue`.
- Own `ocr_`-prefixed tables + Alembic chain (`alembic_version_ocr`) in
  the **shared** database (`FORGE_DB_URL`); reads `batch_jobs` read-only.
- Depends only on `forge-contracts` (a fellow workspace member) — never
  on `forge`.

Run from the workspace root:

```bash
uv run --package ocr ocr --help
uv run --package ocr ocr worker
uv run --package ocr ocr submit <file.pdf>
```

Design history: `development-plans/archive/separate-ocr-into-its-own-repo.md`
at the workspace root.
