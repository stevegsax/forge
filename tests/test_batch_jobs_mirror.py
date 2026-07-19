"""Forge-side sync test: the read-only ``batch_jobs`` mirror tracks the authoritative table.

The platform ships a standalone read-only ``batch_jobs`` mirror
(``sax_platform.contracts.batch_jobs``) so consumer apps can SELECT the table
without importing ``forge``. The authoritative schema is ``forge.store.BatchJob``.
This module closes the schema-drift risk called out for T4.1: every mirror column
must exist on the real table with a matching SQLAlchemy type class, and the mirror
must never *under*-declare nullability. A read-only mirror may *over*-declare a
column nullable (harmless for a SELECT that never writes or CREATEs the table), but
claiming a source-nullable column is NOT NULL would let a reader assume non-null and
be surprised by a NULL — that direction is a real drift and fails.

The authoritative table may carry extra columns the mirror omits (a mirror may be a
subset), so the comparison runs over the mirror's columns, not the real table's.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sax_platform.contracts.batch_jobs import batch_jobs as mirror

from forge.store import BatchJob

_REAL = BatchJob.__table__
_MIRROR_COLUMN_NAMES = list(mirror.columns.keys())


class TestMirrorTracksAuthoritative:
    def test_every_mirror_column_exists_on_real_table(self) -> None:
        missing = set(_MIRROR_COLUMN_NAMES) - set(_REAL.columns.keys())
        assert not missing, f"mirror columns absent from forge.store.BatchJob: {sorted(missing)}"

    @pytest.mark.parametrize("name", _MIRROR_COLUMN_NAMES)
    def test_type_class_matches(self, name: str) -> None:
        assert type(mirror.columns[name].type) is type(_REAL.columns[name].type)

    @pytest.mark.parametrize("name", _MIRROR_COLUMN_NAMES)
    def test_nullability_matches_or_safely_over_declares(self, name: str) -> None:
        real_nullable = _REAL.columns[name].nullable
        mirror_nullable = mirror.columns[name].nullable
        if real_nullable == mirror_nullable:
            return  # exact match — the norm
        # The only tolerated divergence is the mirror over-declaring nullability
        # (mirror nullable, source NOT NULL): safe for a read-only SELECT mirror.
        # The reverse (mirror NOT NULL while the source allows NULL) is real drift.
        assert mirror_nullable and not real_nullable, (
            f"{name}: mirror under-declares nullability "
            f"(mirror nullable={mirror_nullable}, real nullable={real_nullable})"
        )

    def test_known_over_declared_columns(self) -> None:
        """Pin the current mirror/source nullability divergence so it stays visible.

        ``created_at``/``updated_at`` are NOT NULL in ``forge.store.BatchJob``
        (``Mapped[datetime]``) but nullable in the mirror (a plain ``sa.Column``
        defaults to nullable). This is harmless for a read-only SELECT mirror; it is
        pinned here so the divergence is explicit rather than silent, and so any new
        divergence in a different column trips this test.
        """
        over_declared = {
            name
            for name in _MIRROR_COLUMN_NAMES
            if mirror.columns[name].nullable and not _REAL.columns[name].nullable
        }
        assert over_declared == {"created_at", "updated_at"}


def test_mirror_is_a_standalone_table() -> None:
    """Sanity: the mirror is a real ``sa.Table`` distinct from the ORM-mapped one."""
    assert isinstance(mirror, sa.Table)
    assert mirror is not _REAL
