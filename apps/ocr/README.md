# ocr

Document OCR application built on the **Forge** platform: submits and polls
its own Mistral batch OCR jobs via `sax_platform.ocr.MistralOcr`, extracts
text + images, and stores results. A workspace member of the forge
monorepo at `apps/ocr` (D98).

- Own Temporal worker on `ocr-task-queue` (same namespace as Forge). The
  worker **requires `MISTRAL_API_KEY`** and fails fast without it.
- Records `batch_jobs` ledger rows on `forge-task-queue` via cross-queue
  activity calls (`persist_block`), and reads that ledger read-only — no
  signals.
- Own `ocr_`-prefixed tables + Alembic chain (`alembic_version_ocr`) in
  the **shared** database (`FORGE_DB_URL`).
- Depends on `sax_platform` (a fellow workspace member) — never on `forge`.

Run from the workspace root:

```bash
uv run --package ocr ocr --help
uv run --package ocr ocr worker
uv run --package ocr ocr submit <file.pdf>   # start-only; echoes the workflow id
```

Design history: `development-plans/archive/separate-ocr-into-its-own-repo.md`
at the workspace root.
