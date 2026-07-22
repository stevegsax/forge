-- Runs once, on first cluster init, after 01-extensions.sql.
--
-- Provisions the production-parity application databases (T0.9): forge/ocr
-- share `forge`; pbook gets its own `pbook` database. Both live on this
-- forge-postgres instance beside Temporal's `temporal` / `temporal_visibility`
-- and the disposable `forge_dev`, so a fresh `make stack-up` from a clean
-- volume matches the production layout that Supabase used to hold.
--
-- The `vector` extension is enabled in each (pbook needs pgvector for
-- embeddings; forge's own schema uses none — parity/convenience there). The
-- image is pgvector/pgvector:pg16, so the extension binary is already present.
--
-- CREATE DATABASE cannot run inside a transaction block; the initdb psql
-- invocation autocommits each statement, and `\connect` switches databases so
-- the extension lands in the right one.

CREATE DATABASE forge OWNER forge;
CREATE DATABASE pbook OWNER forge;

\connect forge
CREATE EXTENSION IF NOT EXISTS vector;

\connect pbook
CREATE EXTENSION IF NOT EXISTS vector;
