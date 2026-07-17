# Changelog

One row per completed task, appended on completion (newest first) per [PROCESS.md](PROCESS.md).
Work completed before this file was introduced is recorded in `git log` — including the Phase 1–14 build, store externalization (Postgres + S3), the OCR pipeline, transcript ingestion, the planner eval framework, and mTLS remote access.

| Date | Task | PR |
| ------ | ------ | ---- |
| 2026-07-16 | [T3.1 — Platform LLM client, both lanes: sax_platform born (98 tests, 99% cov; rec-7 + live-API pre-work recorded)](tasks/T3.1-platform-llm-client.md) | [#37](https://github.com/stevegsax/forge/pull/37) |
| 2026-07-16 | [T2.3d — mypy strict: pbook (177 errors; 16 recorded stub-boundary ignores) — Phase 2 complete](tasks/T2.3d-mypy-strict-pbook.md) | [#36](https://github.com/stevegsax/forge/pull/36) |
| 2026-07-16 | [T2.3c — mypy strict: ocr (zero ignores)](tasks/T2.3c-mypy-strict-ocr.md) | [#35](https://github.com/stevegsax/forge/pull/35) |
| 2026-07-16 | [T2.3b — mypy strict: sax-llm (zero ignores; one boundary cast)](tasks/T2.3b-mypy-strict-platform-llm.md) | [#34](https://github.com/stevegsax/forge/pull/34) |
| 2026-07-16 | [T2.3a — mypy strict: forge-contracts (zero ignores); member mypy wired into make + CI](tasks/T2.3a-mypy-strict-platform-contracts.md) | [#33](https://github.com/stevegsax/forge/pull/33) |
| 2026-07-16 | [T2.2 — Root gates: GitHub Actions CI, import-linter DAG contracts, 85% coverage gates everywhere](tasks/T2.2-root-gates.md) | [#32](https://github.com/stevegsax/forge/pull/32) |
| 2026-07-16 | [T0.7 — Local deployment: retire EC2 (D99); status markers closed out](tasks/T0.7-deploy-hardening.md) | — |
| 2026-07-16 | [T2.1 — COMPLETE: monorepo workspace in forge (all increments + Python 3.14 bump)](tasks/T2.1-workspace-creation.md) | — |
| 2026-07-16 | [T2.1 increment 3 — ocr absorbed into apps/ocr; D98 repo consolidation complete; task continues (3.14 bump)](tasks/T2.1-workspace-creation.md) | — |
| 2026-07-16 | [T2.1 increment 2 — sax-llm + forge-contracts absorbed into libs/; Finding A closed; ocr repointed; task continues](tasks/T2.1-workspace-creation.md) | — |
| 2026-07-16 | [T2.1 increment 1 — pbook absorbed as a workspace member at apps/pbook (D98); task continues](tasks/T2.1-workspace-creation.md) | — |
| 2026-07-16 | [T1.0 amendment — sax-llm editable everywhere; pin workaround unwound (pbook + forge)](tasks/T1.0-uniform-editable-sibling-sources.md) | — |
| 2026-07-15 | [T1.8 — Small dedup batch + kill runs-extraction](tasks/T1.8-small-dedup-batch.md) | — |
| 2026-07-15 | [T1.6b — Batch-wait failure symmetry](tasks/T1.6b-batch-wait-failure-symmetry.md) | — |
| 2026-07-15 | [T1.6a — Idempotency rekey](tasks/T1.6a-idempotency-rekey.md) | — |
| 2026-07-15 | [T1.5 — Nested fan-out propagation fix](tasks/T1.5-nested-fan-out-propagation.md) | — |
| 2026-07-15 | [T1.7 — Env scrub at model-influenced subprocess seams](tasks/T1.7-env-scrub-subprocess-seams.md) | — |
| 2026-07-15 | [T1.4 — Unblock the worker event loop](tasks/T1.4-unblock-worker-event-loop.md) | — |
| 2026-07-15 | [T1.3 — INTERIM minimal poller patch](tasks/T1.3-interim-poller-patch.md) | — |
| 2026-07-15 | [T1.2 — INTERIM batch-result correlation stopgap](tasks/T1.2-interim-batch-result-correlation.md) | — |
| 2026-07-15 | [T1.1 — Delete the dead provider stack; repatriate sax-llm's tests](tasks/T1.1-delete-dead-provider-stack.md) | — |
| 2026-07-15 | [T1.0 — Uniform editable sibling sources](tasks/T1.0-uniform-editable-sibling-sources.md) | — |
