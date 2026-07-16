"""Shared SQLAlchemy column types.

Used by Forge's store and by consumer apps' own stores (which define their own
``DeclarativeBase`` but share these column types), so the storage layer behaves
identically across repos.
"""

from __future__ import annotations

from datetime import UTC

import sqlalchemy as sa


class UTCDateTime(sa.types.TypeDecorator):
    """Always-UTC DateTime column type.

    SQLite cannot preserve tzinfo on DateTime columns, so we normalize
    to UTC on the way in (stripping tz for storage) and re-attach UTC
    on the way out. Naive inputs are assumed to already be UTC, per
    project convention. Callers always receive tz-aware UTC datetimes.
    """

    impl = sa.DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is not None:
            return value.astimezone(UTC).replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
