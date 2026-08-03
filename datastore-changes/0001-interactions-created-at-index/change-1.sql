
-- Running upgrade 003 -> 004

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_interactions_created_at ON interactions (created_at);

UPDATE alembic_version_forge SET version_num='004' WHERE alembic_version_forge.version_num = '003';

