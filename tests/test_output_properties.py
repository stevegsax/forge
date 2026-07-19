"""Property-based tests for forge.activities.output — search/replace matching
and application (T0.5, AC 3).

These exercise the *pure* core (`apply_edits`, `apply_edits_detailed`, and the
matching strategies) with `hypothesis`. They carry no marker, so they run in the
default suite. Generators are kept tight (small, disjoint alphabets and bounded
sizes) so the suite stays fast and discards stay rare:

- ``_CONTEXT`` — lowercase + space + newline; used for prefix/suffix/context.
- ``_SEARCH`` — uppercase only; a search block built from it can never occur in
  the context, so ``content = prefix + search + suffix`` contains it exactly once.
- ``_REPL`` — digits only; disjoint from both, so a plain replacement can never
  reintroduce the (uppercase) search text.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from forge.activities.output import (
    EditApplicationError,
    _exact_match,
    _whitespace_normalized_match,
    apply_edits,
)
from forge.models import SearchReplaceEdit

_CONTEXT = st.text(alphabet="abcdefghijklmnopqrstuvwxyz \n", max_size=32)
_SEARCH = st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=12)
_REPL = st.text(alphabet="0123456789", max_size=12)


# ---------------------------------------------------------------------------
# Idempotency (the retry-safety property)
# ---------------------------------------------------------------------------


@given(prefix=_CONTEXT, suffix=_CONTEXT, search=_SEARCH, ins_pre=_REPL, ins_post=_REPL)
def test_insert_style_apply_twice_equals_once(
    prefix: str, suffix: str, search: str, ins_pre: str, ins_post: str
) -> None:
    """Insert-style edit (replace embeds search): applying twice under the
    idempotent skip equals applying once — no duplicated insertion on retry."""
    replace = ins_pre + search + ins_post  # search in replace -> insert-style
    content = prefix + search + suffix
    assume(content.count(search) == 1)
    edit = SearchReplaceEdit(search=search, replace=replace)

    once = apply_edits(content, [edit], idempotent=True)
    twice = apply_edits(once, [edit], idempotent=True)
    assert twice == once


@given(prefix=_CONTEXT, suffix=_CONTEXT, search=_SEARCH, replace=_REPL)
def test_plain_replace_apply_twice_equals_once(
    prefix: str, suffix: str, search: str, replace: str
) -> None:
    """Plain edit (search disjoint from replace): re-applying to already-edited
    content skips rather than raising 'not found'."""
    assume(search not in replace)  # plain replace
    content = prefix + search + suffix
    assume(content.count(search) == 1)
    edit = SearchReplaceEdit(search=search, replace=replace)

    once = apply_edits(content, [edit], idempotent=True)
    assume(search not in once)  # focus on the already-applied state, not a straddle
    twice = apply_edits(once, [edit], idempotent=True)
    assert twice == once


# ---------------------------------------------------------------------------
# Surgical application — everything outside the match is preserved
# ---------------------------------------------------------------------------


@given(prefix=_CONTEXT, suffix=_CONTEXT, search=_SEARCH, replace=_REPL)
def test_exact_match_replacement_is_surgical(
    prefix: str, suffix: str, search: str, replace: str
) -> None:
    """When search occurs exactly once verbatim, applying the edit replaces only
    the match and preserves everything around it."""
    content = prefix + search + suffix
    assume(content.count(search) == 1)
    edit = SearchReplaceEdit(search=search, replace=replace)

    assert apply_edits(content, [edit]) == prefix + replace + suffix


# ---------------------------------------------------------------------------
# Strategies agree with exact match on unperturbed content
# ---------------------------------------------------------------------------

_LINE_IDS = st.lists(
    st.integers(min_value=0, max_value=99_999), unique=True, min_size=2, max_size=8
)
_SEED = st.integers(min_value=0, max_value=1000)


def _slice_bounds(n: int, a_seed: int, span_seed: int) -> tuple[int, int]:
    """Derive a valid contiguous slice ``[a, b)`` of ``n`` lines with no discards.

    ``0 <= a < b <= n`` for ``n >= 1``.
    """
    a = a_seed % n
    b = a + 1 + span_seed % (n - a)
    return a, b


@given(ids=_LINE_IDS, a_seed=_SEED, span_seed=_SEED)
def test_whitespace_match_agrees_with_exact_on_verbatim_lines(
    ids: list[int], a_seed: int, span_seed: int
) -> None:
    """For a verbatim, line-aligned, unique search, the whitespace-normalized
    matcher returns the same span exact match would."""
    lines = [f"row_{i} = {i}\n" for i in ids]  # distinct lines
    content = "".join(lines)
    a, b = _slice_bounds(len(lines), a_seed, span_seed)
    search = "".join(lines[a:b])
    assume(content.count(search) == 1)

    exact_start = _exact_match(content, search)
    assert exact_start is not None
    ws = _whitespace_normalized_match(content, search)
    assert ws is not None
    assert ws == (exact_start, exact_start + len(search))


@given(ids=_LINE_IDS, a_seed=_SEED, span_seed=_SEED)
def test_apply_edits_uses_exact_span_on_verbatim_lines(
    ids: list[int], a_seed: int, span_seed: int
) -> None:
    """End-to-end: applying an edit whose search is present verbatim among
    distinct lines is surgical (goes through the exact-match path)."""
    lines = [f"row_{i} = {i}\n" for i in ids]
    content = "".join(lines)
    a, b = _slice_bounds(len(lines), a_seed, span_seed)
    search = "".join(lines[a:b])
    assume(content.count(search) == 1)
    replace = "REPLACED\n"

    prefix = "".join(lines[:a])
    suffix = "".join(lines[b:])
    assert apply_edits(content, [SearchReplaceEdit(search=search, replace=replace)]) == (
        prefix + replace + suffix
    )


@given(ids=_LINE_IDS)
def test_empty_search_string_always_rejected(ids: list[int]) -> None:
    """An empty search string is rejected regardless of content (invariant)."""
    content = "".join(f"row_{i} = {i}\n" for i in ids)
    with pytest.raises(EditApplicationError, match="empty search string"):
        apply_edits(content, [SearchReplaceEdit(search="", replace="x")])
