"""Tests for the shared ``UTCDateTime`` SQLAlchemy column type.

SQLite cannot preserve tzinfo on DateTime columns, so ``UTCDateTime`` strips
tz on the way in and re-attaches UTC on the way out. Exercised through a real
in-memory SQLite table (not by calling ``process_bind_param`` directly) so the
test proves the type behaves correctly under actual SQLAlchemy execution.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import sqlalchemy as sa

from sax_platform.contracts.types import UTCDateTime

_metadata = sa.MetaData()
_events = sa.Table(
    "events",
    _metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("happened_at", UTCDateTime),
)


def _engine() -> sa.Engine:
    engine = sa.create_engine("sqlite://")
    _metadata.create_all(engine)
    return engine


def _roundtrip(value: datetime | None) -> datetime | None:
    engine = _engine()
    with engine.begin() as conn:
        conn.execute(_events.insert().values(id=1, happened_at=value))
    with engine.connect() as conn:
        row = conn.execute(sa.select(_events.c.happened_at).where(_events.c.id == 1)).one()
    return row[0]


class TestUTCDateTime:
    def test_naive_datetime_assumed_utc_on_read(self) -> None:
        naive = datetime(2026, 1, 1, 12, 0, 0)
        result = _roundtrip(naive)
        assert result == naive.replace(tzinfo=UTC)
        assert result.tzinfo is not None

    def test_tz_aware_datetime_normalized_to_utc(self) -> None:
        eastern = timezone(timedelta(hours=-5))
        aware = datetime(2026, 1, 1, 7, 0, 0, tzinfo=eastern)  # 12:00 UTC
        result = _roundtrip(aware)
        assert result == datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    def test_utc_datetime_roundtrips_unchanged(self) -> None:
        aware = datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)
        assert _roundtrip(aware) == aware

    def test_none_roundtrips_as_none(self) -> None:
        assert _roundtrip(None) is None

    def test_process_result_value_converts_non_utc_tzaware_driver_value(self) -> None:
        """Real drivers always hand back naive values here (we strip tz on bind,
        per ``process_bind_param``), but the decoder itself must still normalize
        a tz-aware value to UTC rather than assume naive — this is that contract,
        checked directly against the public TypeDecorator hook."""
        eastern = timezone(timedelta(hours=-5))
        aware = datetime(2026, 1, 1, 7, 0, 0, tzinfo=eastern)
        result = UTCDateTime().process_result_value(aware, dialect=None)
        assert result == datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
