"""Tests for the forge-contracts wire models and result-payload envelope."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sax_platform.contracts.models import (
    BatchJobStatus,
    BatchResult,
    BatchSubmitResult,
    BatchSubmitSpiInput,
    dump_batch_result_payload,
    parse_batch_result_payload,
    resolve_batch_result,
)

if TYPE_CHECKING:
    import pytest


class TestBatchJobStatus:
    def test_processing_member_exists(self) -> None:
        assert BatchJobStatus.PROCESSING.value == "processing"


class TestSubmitSpi:
    def test_spi_input_defaults(self) -> None:
        spi = BatchSubmitSpiInput(s3_key="k", model="m", custom_id="c")
        assert spi.endpoint == ""
        assert spi.provider == "anthropic"

    def test_submit_result_round_trips(self) -> None:
        r = BatchSubmitResult(request_id="c", batch_id="b", provider="mistral")
        assert BatchSubmitResult.model_validate_json(r.model_dump_json()) == r


class TestResultPayloadEnvelope:
    def test_dump_parse_round_trip(self) -> None:
        images = [{"original_image_id": "img-0.jpeg", "page_index": 0}]
        env = dump_batch_result_payload('{"pages": []}', images)
        body, imgs = parse_batch_result_payload(env)
        assert body == '{"pages": []}'
        assert imgs == images

    def test_resolve_inline_has_no_images(self) -> None:
        result = BatchResult(
            request_id="r", batch_id="b", raw_response_json='{"x": 1}', result_type="LLMResponse"
        )
        body, imgs = resolve_batch_result(result)
        assert body == '{"x": 1}'
        assert imgs == []

    def test_resolve_pointer_fetches_envelope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        envelope = dump_batch_result_payload('{"pages": [1]}', [{"id": "a"}])

        from sax_platform.contracts import s3_blobs

        monkeypatch.setattr(s3_blobs, "get", lambda key: envelope.encode("utf-8"))

        result = BatchResult(
            request_id="r", batch_id="b", s3_key="some/key", result_type="LLMResponse"
        )
        body, imgs = resolve_batch_result(result)
        assert body is not None
        assert json.loads(body) == {"pages": [1]}
        assert imgs == [{"id": "a"}]
