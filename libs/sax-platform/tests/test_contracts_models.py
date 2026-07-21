"""Tests for the forge-contracts wire models and result-payload envelope."""

from __future__ import annotations

from sax_platform.contracts.models import (
    BatchJobStatus,
    dump_batch_result_payload,
    parse_batch_result_payload,
)


class TestBatchJobStatus:
    def test_processing_member_exists(self) -> None:
        assert BatchJobStatus.PROCESSING.value == "processing"


class TestResultPayloadEnvelope:
    def test_dump_parse_round_trip(self) -> None:
        images = [{"original_image_id": "img-0.jpeg", "page_index": 0}]
        env = dump_batch_result_payload('{"pages": []}', images)
        body, imgs = parse_batch_result_payload(env)
        assert body == '{"pages": []}'
        assert imgs == images

    def test_parse_defaults_missing_images_to_empty(self) -> None:
        """An inline body was stashed without images: parse yields an empty list."""
        env = dump_batch_result_payload('{"x": 1}', [])
        body, imgs = parse_batch_result_payload(env)
        assert body == '{"x": 1}'
        assert imgs == []
