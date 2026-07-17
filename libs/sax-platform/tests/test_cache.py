"""Tests for sax_platform.llm.cache — pure prompt-cache placement policy."""

from typing import Any

import pytest
from pydantic import ValidationError

from sax_platform.llm.cache import (
    CacheSpec,
    apply_cache_control,
    estimate_tokens,
    min_cacheable_tokens,
)


class TestEstimateTokens:
    def test_chars_over_four_heuristic(self) -> None:
        assert estimate_tokens("a" * 400) == 100

    def test_empty_string_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_rounds_down(self) -> None:
        assert estimate_tokens("abc") == 0  # 3 // 4 == 0
        assert estimate_tokens("abcd") == 1  # 4 // 4 == 1


class TestMinCacheableTokens:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("claude-opus-4-8", 4096),
            ("claude-opus-4-7", 4096),
            ("claude-opus-4-6", 4096),
            ("claude-haiku-4-5", 4096),
            ("claude-haiku-4-5-20251001", 4096),
            ("claude-fable-5", 2048),
            ("claude-sonnet-4-6", 2048),
            ("claude-sonnet-4-5", 1024),
            ("claude-sonnet-4-5-20250929", 1024),
        ],
    )
    def test_known_model_prefixes(self, model: str, expected: int) -> None:
        assert min_cacheable_tokens(model) == expected

    def test_unknown_model_is_conservative(self) -> None:
        assert min_cacheable_tokens("some-future-model-nobody-has-heard-of") == 4096

    def test_empty_string_is_conservative(self) -> None:
        assert min_cacheable_tokens("") == 4096

    def test_longest_prefix_wins(self) -> None:
        # "claude-sonnet-4-5" and "claude-sonnet-4-6" share the prefix
        # "claude-sonnet-4-" but neither is a prefix of the other, so each
        # model string should resolve to its own distinct threshold rather
        # than an unrelated shorter match.
        assert min_cacheable_tokens("claude-sonnet-4-5") == 1024
        assert min_cacheable_tokens("claude-sonnet-4-6") == 2048


class TestApplyCacheControl:
    BIG_MODEL = "claude-opus-4-8"  # threshold 4096

    def _big_block(self, n_tokens: int) -> dict[str, Any]:
        # estimate_tokens is chars // 4, so 4 chars ~= 1 token.
        return {"type": "text", "text": "x" * (n_tokens * 4)}

    def test_none_spec_returns_unchanged(self) -> None:
        blocks = [self._big_block(5000)]

        result = apply_cache_control(blocks, model=self.BIG_MODEL, spec=None)

        assert result == blocks
        assert "cache_control" not in result[-1]

    def test_below_minimum_omits_breakpoint(self) -> None:
        blocks = [self._big_block(10)]  # far below 4096

        result = apply_cache_control(blocks, model=self.BIG_MODEL, spec=CacheSpec())

        assert result == blocks
        assert "cache_control" not in result[-1]

    def test_below_minimum_is_model_specific(self) -> None:
        # Same block size, but claude-sonnet-4-5's threshold (1024) is
        # cleared while claude-opus-4-8's (4096) is not.
        blocks = [self._big_block(2000)]

        opus_result = apply_cache_control(blocks, model=self.BIG_MODEL, spec=CacheSpec())
        sonnet_result = apply_cache_control(blocks, model="claude-sonnet-4-5", spec=CacheSpec())

        assert "cache_control" not in opus_result[-1]
        assert "cache_control" in sonnet_result[-1]

    def test_at_or_above_minimum_places_breakpoint_on_last_block(self) -> None:
        blocks = [self._big_block(2000), self._big_block(3000)]

        result = apply_cache_control(blocks, model=self.BIG_MODEL, spec=CacheSpec())

        assert "cache_control" not in result[0]
        assert result[-1]["cache_control"] == {"type": "ephemeral"}

    def test_default_ttl_omits_ttl_key(self) -> None:
        blocks = [self._big_block(5000)]

        result = apply_cache_control(blocks, model=self.BIG_MODEL, spec=CacheSpec(ttl="5m"))

        assert result[-1]["cache_control"] == {"type": "ephemeral"}
        assert "ttl" not in result[-1]["cache_control"]

    def test_one_hour_ttl_included(self) -> None:
        blocks = [self._big_block(5000)]

        result = apply_cache_control(blocks, model=self.BIG_MODEL, spec=CacheSpec(ttl="1h"))

        assert result[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_empty_blocks_returns_empty(self) -> None:
        result = apply_cache_control([], model=self.BIG_MODEL, spec=CacheSpec())

        assert result == []

    def test_does_not_mutate_input_list_or_blocks(self) -> None:
        original_block = self._big_block(5000)
        blocks = [original_block]

        result = apply_cache_control(blocks, model=self.BIG_MODEL, spec=CacheSpec(ttl="1h"))

        # The input list itself is untouched...
        assert blocks == [original_block]
        assert "cache_control" not in blocks[0]
        # ...and the returned last block is a new dict, not the same object.
        assert result[-1] is not original_block
        assert "cache_control" not in original_block

    def test_returns_new_list_object_even_when_unchanged(self) -> None:
        blocks = [self._big_block(10)]

        result = apply_cache_control(blocks, model=self.BIG_MODEL, spec=None)

        assert result is not blocks

    def test_non_last_blocks_are_same_object_when_breakpoint_placed(self) -> None:
        first = self._big_block(2000)
        second = self._big_block(3000)
        blocks = [first, second]

        result = apply_cache_control(blocks, model=self.BIG_MODEL, spec=CacheSpec())

        assert result[0] is first

    def test_non_text_field_treated_as_zero_tokens(self) -> None:
        # A block missing a "text" key (or with a non-string "text") should
        # not raise — it just contributes 0 estimated tokens.
        blocks: list[dict[str, Any]] = [{"type": "text"}]

        result = apply_cache_control(blocks, model=self.BIG_MODEL, spec=CacheSpec())

        assert "cache_control" not in result[-1]


class TestCacheSpec:
    def test_default_ttl_is_five_minutes(self) -> None:
        assert CacheSpec().ttl == "5m"

    def test_is_frozen(self) -> None:
        spec = CacheSpec()

        with pytest.raises(ValidationError):
            spec.ttl = "1h"  # type: ignore[misc]

    def test_rejects_invalid_ttl(self) -> None:
        with pytest.raises(ValidationError):
            CacheSpec(ttl="2h")  # type: ignore[arg-type]
