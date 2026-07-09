-- Runs once, against $POSTGRES_DB (forge_dev), on first cluster init.
--
-- The image is pgvector/pgvector:pg16, so the `vector` extension binary is
-- present; enable it in the default dev database. Forge's own schema uses no
-- extensions — this is convenience/parity, not load-bearing — but it lets you
-- confirm the image with `\dx` and forward-fits the migration's Phase 6, where
-- pbook's store needs pgvector. pbook, when added, enables it in its own
-- database via its own migrations.
CREATE EXTENSION IF NOT EXISTS vector;
