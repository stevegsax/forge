"""Tests for the forge-contracts wire models and result-payload envelope."""

from __future__ import annotations

import json

from sax_platform.contracts.models import (
    BatchJobStatus,
    BatchResult,
    BatchSubmitResult,
    BatchSubmitSpiInput,
    dump_batch_result_payload,
    parse_batch_result_payload,
    resolve_batch_result,
)


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

    def test_resolve_pointer_uses_injected_blobs(self) -> None:
        """The envelope is fetched through the injected S3Blobs."""
        envelope = dump_batch_result_payload('{"pages": [2]}', [{"id": "b"}])

        class FakeBlobs:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def get(self, key: str) -> bytes:
                self.calls.append(key)
                return envelope.encode("utf-8")

        blobs = FakeBlobs()
        result = BatchResult(
            request_id="r", batch_id="b", s3_key="some/key", result_type="LLMResponse"
        )
        body, imgs = resolve_batch_result(result, blobs)  # type: ignore[arg-type]

        assert blobs.calls == ["some/key"]
        assert body is not None
        assert json.loads(body) == {"pages": [2]}
        assert imgs == [{"id": "b"}]

    def test_resolve_inline_ignores_injected_blobs(self) -> None:
        """An inline result never touches blob storage even when one is passed."""

        class ExplodingBlobs:
            def get(self, key: str) -> bytes:
                raise AssertionError("blob storage must not be touched for inline results")

        result = BatchResult(
            request_id="r", batch_id="b", raw_response_json='{"x": 1}', result_type="LLMResponse"
        )
        body, imgs = resolve_batch_result(result, ExplodingBlobs())  # type: ignore[arg-type]
        assert body == '{"x": 1}'
        assert imgs == []
