"""Slim Anthropic caller: one client, blunt retries, thinking-aware content extraction.

Deliberately not touchstone's AIMD-gated multi-key pool — this benchmark runs a few
hundred sequential calls, so a single key with exponential backoff on rate limits and
overloads is enough. If gloss ever fans out, lift ``lab.models.anthropic`` instead of
growing this.
"""

from __future__ import annotations

import os
import time
from typing import Any

import anthropic
from loguru import logger
from pydantic import BaseModel

RETRYABLE_STATUS = {429, 500, 503, 529}
_MAX_ATTEMPTS = 8

# Models on the adaptive-thinking API (Claude 4.6+ / 5 family): `budget_tokens` returns a
# 400 there, and `display: "summarized"` is required for the thinking text to be non-empty
# (the default is "omitted" — empty strings, which would leave this benchmark with no CoT).
_ADAPTIVE_THINKING_PREFIXES = (
    "claude-fable",
    "claude-mythos",
    "claude-opus-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)


def thinking_param(model: str, budget_tokens: int) -> dict[str, object]:
    """The correct ``thinking`` request value for ``model``.

    Adaptive-thinking models ignore the budget (depth is controlled by
    ``output_config.effort``, left at its default); older models (Haiku 4.5, Sonnet 4.5)
    take an explicit token budget, minimum 1024.
    """
    if model.startswith(_ADAPTIVE_THINKING_PREFIXES):
        return {"type": "adaptive", "display": "summarized"}
    return {"type": "enabled", "budget_tokens": max(budget_tokens, 1024)}


class CompletionBlocks(BaseModel):
    """The pieces of a response the pipeline cares about, flattened from content blocks."""

    thinking: str
    text: str
    tool_name: str | None
    tool_input: dict[str, object]  # tool_use input is schema-free JSON at this boundary
    raw_content: list[dict[str, object]]  # verbatim blocks, resent to preserve signatures
    stop_reason: str


def _client() -> anthropic.Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set — export it before running gloss")
    return anthropic.Anthropic(max_retries=0)  # retries are handled here, uniformly


def create_message(**request: Any) -> anthropic.types.Message:  # request mirrors messages.create
    """Streamed ``messages.create`` with exponential backoff on 429/500/503/529.

    Always streams and returns the final message: adaptive-thinking turns need a large
    ``max_tokens`` (thinking counts against it), and non-streaming requests that large
    hit SDK HTTP timeouts.
    """
    client = _client()
    delay = 2.0
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with client.messages.stream(**request) as stream:
                return stream.get_final_message()
        except anthropic.APIStatusError as exc:
            if exc.status_code not in RETRYABLE_STATUS or attempt == _MAX_ATTEMPTS:
                raise
            logger.warning(
                f"anthropic {exc.status_code} (attempt {attempt}/{_MAX_ATTEMPTS}), "
                f"retrying in {delay:.0f}s"
            )
            time.sleep(delay)
            delay = min(delay * 2, 60)
        except anthropic.APIConnectionError:
            if attempt == _MAX_ATTEMPTS:
                raise
            logger.warning(
                f"anthropic connection error (attempt {attempt}), retrying in {delay:.0f}s"
            )
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise AssertionError("unreachable: loop either returns or raises")


def extract_blocks(message: anthropic.types.Message) -> CompletionBlocks:
    """Flatten a response into thinking text, visible text, and the first tool call."""
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    tool_name: str | None = None
    tool_input: dict[str, object] = {}
    for block in message.content:
        if block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use" and tool_name is None:
            tool_name = block.name
            tool_input = dict(block.input)
    return CompletionBlocks(
        thinking="\n\n".join(thinking_parts),
        text="\n\n".join(text_parts),
        tool_name=tool_name,
        tool_input=tool_input,
        raw_content=[block.model_dump() for block in message.content],
        stop_reason=message.stop_reason or "",
    )
