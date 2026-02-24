#!/usr/bin/env python3
"""Tests for clean-text preprocessing pipeline."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess import (
    parse_session_file,
    extract_user_text,
    extract_assistant_text,
    extract_tool_result_summary,
    session_to_text,
    should_include_session,
    redact_pii,
    process,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_SESSION = FIXTURES / "sample_session.jsonl"


class TestParseSessionFile:
    def test_extracts_messages_only(self):
        events = parse_session_file(SAMPLE_SESSION)
        assert len(events) == 8
        assert events[0]["role"] == "user"
        assert events[1]["role"] == "assistant"
        assert events[2]["role"] == "toolResult"

    def test_preserves_tool_name(self):
        events = parse_session_file(SAMPLE_SESSION)
        tool_results = [e for e in events if e["role"] == "toolResult"]
        assert all(e["toolName"] == "exec" for e in tool_results)


class TestExtractUserText:
    def test_string_content(self):
        assert extract_user_text("hello") == "hello"

    def test_list_content(self):
        content = [{"type": "text", "text": "서버 확인해줘"}]
        assert extract_user_text(content) == "서버 확인해줘"

    def test_empty_content(self):
        assert extract_user_text("") == ""
        assert extract_user_text([]) == ""


class TestExtractAssistantText:
    def test_skips_thinking(self):
        content = [
            {"type": "thinking", "thinking": "internal reasoning"},
            {"type": "text", "text": "visible response"},
        ]
        result = extract_assistant_text(content)
        assert "internal reasoning" not in result
        assert "visible response" in result

    def test_skips_tool_calls(self):
        content = [
            {"type": "text", "text": "서버 확인할게."},
            {"type": "toolCall", "toolName": "exec", "arguments": {"command": "uptime"}},
        ]
        result = extract_assistant_text(content)
        assert "서버 확인할게." in result
        assert "exec" not in result
        assert "uptime" not in result

    def test_thinking_only_returns_empty(self):
        content = [{"type": "thinking", "thinking": "just thinking"}]
        assert extract_assistant_text(content) == ""

    def test_string_content(self):
        assert extract_assistant_text("simple text") == "simple text"


class TestExtractToolResultSummary:
    def test_short_result(self):
        event = {
            "content": [{"type": "text", "text": "active (running)"}],
            "toolName": "exec",
        }
        result = extract_tool_result_summary(event)
        assert result == "[exec result: active (running)]"

    def test_long_result_truncated(self):
        event = {
            "content": "x" * 1000,
            "toolName": "exec",
        }
        result = extract_tool_result_summary(event)
        assert result is not None
        assert result.endswith("...]")
        assert len(result) < 1000

    def test_empty_result(self):
        event = {"content": "", "toolName": "exec"}
        assert extract_tool_result_summary(event) is None


class TestSessionToText:
    def test_sample_session(self):
        events = parse_session_file(SAMPLE_SESSION)
        text = session_to_text(events)
        assert "Kevin: 서버 상태 확인해줘" in text
        assert "Assistant: 서버 상태 확인할게." in text
        assert "Assistant: 서버 정상 작동 중" in text
        assert "Kevin: 고마워" in text
        assert "Assistant: ㅇㅇ" in text

    def test_no_json_artifacts(self):
        events = parse_session_file(SAMPLE_SESSION)
        text = session_to_text(events)
        assert '"type"' not in text
        assert '"toolName"' not in text
        assert '"arguments"' not in text

    def test_tool_results_included(self):
        events = parse_session_file(SAMPLE_SESSION)
        text = session_to_text(events)
        assert "[exec result:" in text


class TestShouldIncludeSession:
    def test_enough_turns(self):
        events = parse_session_file(SAMPLE_SESSION)
        assert should_include_session(events) is True

    def test_empty_events(self):
        assert should_include_session([]) is False

    def test_single_turn(self):
        events = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert should_include_session(events, min_turns=2) is False


class TestPIIRedaction:
    def test_phone_redacted(self):
        assert "[PHONE]" in redact_pii("전화번호: 010-1234-5678")

    def test_email_redacted(self):
        assert "[EMAIL]" in redact_pii("kevin@example.com")

    def test_api_key_redacted(self):
        assert "[API_KEY]" in redact_pii("sk-abc123def456ghi789jkl012mno")


class TestEndToEnd:
    def test_produces_valid_jsonl(self, tmp_path):
        output = tmp_path / "train.jsonl"
        process([SAMPLE_SESSION], output)
        assert output.exists()

        with open(output) as f:
            lines = f.readlines()

        assert len(lines) == 1  # one session = one document
        doc = json.loads(lines[0])
        assert "text" in doc
        assert "Kevin:" in doc["text"]
        assert "Assistant:" in doc["text"]

    def test_no_json_in_output(self, tmp_path):
        output = tmp_path / "train.jsonl"
        process([SAMPLE_SESSION], output)

        with open(output) as f:
            doc = json.loads(f.readline())

        # The text field should be clean natural language
        assert '"type"' not in doc["text"]
        assert '"toolCall"' not in doc["text"]
