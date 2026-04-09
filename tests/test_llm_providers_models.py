"""Tests for LLM provider message models."""

from __future__ import annotations

from sax_llm.models import (
    DocumentContent,
    ImageContent,
    Message,
    ProviderResponse,
    TextContent,
    text_messages,
)


class TestContentModels:
    """Tests for content block models."""

    def test_text_content(self) -> None:
        block = TextContent(text="Hello")
        assert block.type == "text"
        assert block.text == "Hello"

    def test_image_content(self) -> None:
        block = ImageContent(media_type="image/png", data="base64data")
        assert block.type == "image"
        assert block.media_type == "image/png"
        assert block.data == "base64data"

    def test_document_content(self) -> None:
        block = DocumentContent(media_type="application/pdf", data="pdfdata")
        assert block.type == "document"
        assert block.media_type == "application/pdf"
        assert block.data == "pdfdata"


class TestMessage:
    """Tests for the Message model."""

    def test_string_content(self) -> None:
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.cache_control is False

    def test_list_content(self) -> None:
        msg = Message(
            role="user",
            content=[
                TextContent(text="What is this?"),
                ImageContent(media_type="image/jpeg", data="jpg_data"),
            ],
        )
        assert msg.role == "user"
        assert len(msg.content) == 2

    def test_cache_control_flag(self) -> None:
        msg = Message(role="system", content="System prompt", cache_control=True)
        assert msg.cache_control is True


class TestTextMessages:
    """Tests for the text_messages helper."""

    def test_returns_two_messages(self) -> None:
        msgs = text_messages("sys", "user")
        assert len(msgs) == 2

    def test_system_message(self) -> None:
        msgs = text_messages("System prompt", "User prompt")
        assert msgs[0].role == "system"
        assert msgs[0].content == "System prompt"
        assert msgs[0].cache_control is True

    def test_user_message(self) -> None:
        msgs = text_messages("System prompt", "User prompt")
        assert msgs[1].role == "user"
        assert msgs[1].content == "User prompt"
        assert msgs[1].cache_control is False

    def test_cache_system_false(self) -> None:
        msgs = text_messages("sys", "user", cache_system=False)
        assert msgs[0].cache_control is False


class TestProviderResponse:
    """Tests for ProviderResponse with text_content field."""

    def test_tool_input_default(self) -> None:
        resp = ProviderResponse(
            model_name="test",
            input_tokens=10,
            output_tokens=20,
            raw_response_json="{}",
        )
        assert resp.tool_input == {}
        assert resp.text_content is None

    def test_text_content_populated(self) -> None:
        resp = ProviderResponse(
            text_content="Hello world",
            model_name="test",
            input_tokens=10,
            output_tokens=20,
            raw_response_json="{}",
        )
        assert resp.text_content == "Hello world"
        assert resp.tool_input == {}

    def test_tool_input_populated(self) -> None:
        resp = ProviderResponse(
            tool_input={"key": "value"},
            model_name="test",
            input_tokens=10,
            output_tokens=20,
            raw_response_json="{}",
        )
        assert resp.tool_input == {"key": "value"}
        assert resp.text_content is None
