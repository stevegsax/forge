-- Runs once, against $POSTGRES_DB (forge_dev), on first cluster init, before
-- 02-app-databases.sql (which provisions the production-parity `forge` +
-- `pbook` databases, each with `vector`).
--
-- The image is pgvector/pgvector:pg16, so the `vector` extension binary is
-- present; enable it in the default dev database. Forge's own schema uses no
-- extensions — this is convenience/parity, not load-bearing — but it lets you
-- confirm the image with `\dx`, and pbook (which shares forge_dev in dev via
-- its own schema) needs pgvector for embeddings.
CREATE EXTENSION IF NOT EXISTS vector;
