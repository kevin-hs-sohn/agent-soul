#!/usr/bin/env python3
"""
preprocess.py — Convert OpenClaw session logs to clean text for causal LM training.

Parses OpenClaw session JSONL files, strips JSON structure and metadata,
and outputs clean conversational text suitable for standard next-token
prediction fine-tuning.

Each session becomes one text document. Documents are separated by EOS tokens
at training time (handled by train.py).

Usage:
    python preprocess.py --input sessions/ --output train.jsonl
    python preprocess.py --input session.jsonl --output train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import regex

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII Redaction Patterns
# ---------------------------------------------------------------------------

PII_PATTERNS: list[tuple[regex.Pattern, str]] = [
    (regex.compile(r"01[016789][-\s]?\d{3,4}[-\s]?\d{4}"), "[PHONE]"),
    (regex.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "[EMAIL]"),
    (regex.compile(r"\d{6}[-\s]?[1-4]\d{6}"), "[RRN]"),
    (regex.compile(r"sk-[a-zA-Z0-9_\-]{20,}"), "[API_KEY]"),
    (regex.compile(r"0x[a-fA-F0-9]{40}"), "[WALLET]"),
    (regex.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[CARD]"),
    (regex.compile(
        r"(?:mnemonic|seed\s*phrase|recovery\s*phrase|비밀\s*복구\s*구문)"
        r"[\s:]*(?:[a-z]+\s+){11,23}[a-z]+",
        regex.IGNORECASE,
    ), "[MNEMONIC]"),
    (regex.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP]"),
]

TOOL_RESULT_MAX_CHARS = 500

# Patterns indicating heartbeat/cron messages (not useful for training)
HEARTBEAT_PATTERNS = [
    regex.compile(r"^HEARTBEAT_OK$", regex.IGNORECASE),
    regex.compile(r"^\[heartbeat\]", regex.IGNORECASE),
    regex.compile(r"^heartbeat\b", regex.IGNORECASE),
]


def redact_pii(text: str) -> str:
    """Replace PII patterns with redaction tokens."""
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Session JSONL Parsing
# ---------------------------------------------------------------------------

def parse_session_file(filepath: Path) -> list[dict]:
    """Parse an OpenClaw session JSONL file into message events."""
    events: list[dict] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("Skipping invalid JSON at %s:%d — %s", filepath, line_num, e)
                continue

            if obj.get("type") != "message":
                continue

            msg = obj.get("message", {})
            role = msg.get("role")
            content = msg.get("content")
            if role is None or content is None:
                continue

            event = {"role": role, "content": content}
            if role == "toolResult":
                event["toolName"] = msg.get("toolName", "unknown")
            events.append(event)

    return events


# ---------------------------------------------------------------------------
# Content Extraction — clean text, no JSON artifacts
# ---------------------------------------------------------------------------

def extract_user_text(content) -> str:
    """Extract plain text from user message content."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block.strip())
            elif isinstance(block, dict):
                text = block.get("text", "")
                if text.strip():
                    parts.append(text.strip())
        return "\n".join(parts)
    return str(content).strip()


def extract_assistant_text(content) -> str:
    """Extract plain text from assistant message content.

    Skips thinking blocks and toolCall blocks — only keeps natural language text.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content).strip()

    parts: list[str] = []
    for block in content:
        block_type = block.get("type", "")
        if block_type == "text":
            text = block.get("text", "").strip()
            if text:
                parts.append(text)
        # Skip thinking, toolCall — we only want the natural language output
    return "\n".join(parts)


def extract_tool_result_summary(event: dict) -> str | None:
    """Extract a brief summary from tool results.

    Returns None if the result is too noisy to be useful (pure data dumps,
    very long outputs, etc.). Keeps short, meaningful results.
    """
    content = event.get("content", "")
    tool_name = event.get("toolName", "unknown")

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        result_text = "\n".join(text_parts).strip()
    elif isinstance(content, str):
        result_text = content.strip()
    else:
        return None

    if not result_text:
        return None

    # Skip very long results (data dumps, file contents, etc.)
    if len(result_text) > TOOL_RESULT_MAX_CHARS:
        # Take first portion and indicate truncation
        result_text = result_text[:TOOL_RESULT_MAX_CHARS].rsplit("\n", 1)[0]
        return f"[{tool_name} result: {result_text}...]"

    return f"[{tool_name} result: {result_text}]"


# ---------------------------------------------------------------------------
# Session to Clean Text
# ---------------------------------------------------------------------------

def _is_heartbeat(text: str) -> bool:
    """Check if text is a heartbeat/cron message (noise for training)."""
    return any(p.search(text) for p in HEARTBEAT_PATTERNS)


def session_to_text(events: list[dict]) -> str:
    """Convert a session's events into clean conversational text.

    Output format — natural conversation flow:
        Kevin: 서버 상태 확인해줘
        Assistant: 서버 상태 확인할게.
        [exec result: ● soulclaw-inference.service - active (running)]
        Assistant: 서버 정상 작동 중.
        Kevin: 고마워
        Assistant: ㅇㅇ

    Filters out heartbeat/cron messages as they are noise for training.
    """
    lines: list[str] = []

    for event in events:
        role = event["role"]

        if role == "user":
            text = extract_user_text(event["content"])
            if text and not _is_heartbeat(text):
                lines.append(f"Kevin: {text}")

        elif role == "assistant":
            text = extract_assistant_text(event["content"])
            if text and not _is_heartbeat(text):
                lines.append(f"Assistant: {text}")

        elif role == "toolResult":
            summary = extract_tool_result_summary(event)
            if summary:
                lines.append(summary)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session Filtering
# ---------------------------------------------------------------------------

def should_include_session(events: list[dict], min_turns: int = 2) -> bool:
    """Check if a session has enough content for training.

    A turn = a user message followed by at least one assistant message.
    """
    if not events:
        return False
    user_count = sum(1 for e in events if e["role"] == "user")
    assistant_count = sum(1 for e in events if e["role"] == "assistant")
    return user_count >= min_turns and assistant_count >= min_turns


# ---------------------------------------------------------------------------
# File Discovery
# ---------------------------------------------------------------------------

def discover_session_files(input_path: Path) -> list[Path]:
    """Find all .jsonl session files from input path."""
    if input_path.is_file():
        if input_path.suffix == ".jsonl" and ".deleted." not in input_path.name:
            return [input_path]
        return []

    if input_path.is_dir():
        files = sorted(
            f for f in input_path.glob("**/*.jsonl")
            if ".deleted." not in f.name
            and ".bak" not in f.name
            and ".reset." not in f.name
        )
        return files

    log.error("Input path does not exist: %s", input_path)
    return []


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def process(input_paths: list[Path], output_path: Path) -> None:
    """Run the preprocessing pipeline.

    Output: JSONL where each line is {"text": "<clean session text>"}.
    Accepts multiple input paths (files or directories).
    """
    start_time = datetime.now()
    log.info("Starting preprocessing — inputs: %s, output: %s",
             [str(p) for p in input_paths], output_path)

    files: list[Path] = []
    for input_path in input_paths:
        files.extend(discover_session_files(input_path))
    if not files:
        log.error("No input files to process. Exiting.")
        sys.exit(1)

    log.info("Found %d session file(s)", len(files))

    documents: list[str] = []
    sessions_included = 0
    sessions_skipped = 0
    total_events = 0

    for filepath in files:
        events = parse_session_file(filepath)
        total_events += len(events)

        if not should_include_session(events):
            sessions_skipped += 1
            continue

        text = session_to_text(events)
        text = redact_pii(text)

        if len(text.strip()) < 200:
            sessions_skipped += 1
            continue

        documents.append(text)
        sessions_included += 1
        log.info("  %s — %d events, %d chars",
                 filepath.name, len(events), len(text))

    log.info("Sessions: %d included, %d skipped", sessions_included, sessions_skipped)
    log.info("Total events: %d", total_events)

    if not documents:
        log.error("No valid documents produced. Exiting.")
        sys.exit(1)

    # Write as JSONL — one {"text": "..."} per session
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")

    elapsed = (datetime.now() - start_time).total_seconds()
    log.info("Done in %.1fs — %d sessions -> %s", elapsed, sessions_included, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OpenClaw session logs to clean text for causal LM training.",
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True, nargs="+",
        help="Path(s) to session JSONL file(s) or directory(ies).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output JSONL file (one {text: ...} per session).",
    )
    args = parser.parse_args()
    process(args.input, args.output)


if __name__ == "__main__":
    main()
