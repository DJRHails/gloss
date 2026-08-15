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
_MAX_ATTEMPTS = 12  # overload storms on shared org keys can outlast a short budget
_MAX_DELAY = 120.0

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
    ``output_config.effort`` — see :func:`effort_param`); older models (Haiku 4.5,
    Sonnet 4.5) take an explicit token budget, minimum 1024.
    """
    if model.startswith(_ADAPTIVE_THINKING_PREFIXES):
        return {"type": "adaptive", "display": "summarized"}
    return {"type": "enabled", "budget_tokens": max(budget_tokens, 1024)}


def interleaved_thinking_param(model: str) -> dict[str, object]:
    """Extra-header kwargs so pre-4.6 models think between tool calls, not just turn 1.

    Adaptive-thinking models interleave automatically; older models (Haiku 4.5,
    Sonnet 4.5) only think after user turns unless the interleaved-thinking beta is on —
    without it every turn after the first records an empty CoT and yields no items.
    Splat into the request: ``**interleaved_thinking_param(model)``.
    """
    if model.startswith(_ADAPTIVE_THINKING_PREFIXES):
        return {}
    return {"extra_headers": {"anthropic-beta": "interleaved-thinking-2025-05-14"}}


def effort_param(model: str, effort: str) -> dict[str, object]:
    """``output_config`` kwargs for ``model``: effort on adaptive models, empty otherwise.

    ``effort`` errors on pre-4.6 models (Haiku 4.5, Sonnet 4.5), so it is only sent
    where supported. Splat the result into the request: ``**effort_param(model, "high")``.
    """
    if model.startswith(_ADAPTIVE_THINKING_PREFIXES):
        return {"output_config": {"effort": effort}}
    return {}


class ToolUse(BaseModel):
    """One ``tool_use`` block: the id a result must answer, plus the name and arguments."""

    id: str
    name: str
    input: dict[str, object]  # tool_use input is schema-free JSON at this boundary


class CompletionBlocks(BaseModel):
    """The pieces of a response the pipeline cares about, flattened from content blocks.

    ``tool_calls`` holds **every** ``tool_use`` block in response order, not just the
    first: a response may carry several (parallel tool use), and a caller that reads only
    the first both loses the others' arguments and strands their ids, which the next
    request rejects. Ask for what you want by name (:meth:`first_call`,
    :meth:`calls_named`) and answer every id you were given.
    """

    thinking: str
    text: str
    tool_calls: list[ToolUse]
    raw_content: list[dict[str, object]]  # verbatim blocks, resent to preserve signatures
    stop_reason: str

    def calls_named(self, name: str) -> list[ToolUse]:
        """Every ``tool_use`` block for tool ``name``, in response order."""
        return [call for call in self.tool_calls if call.name == name]

    def first_call(self, name: str) -> ToolUse | None:
        """The first ``tool_use`` block for tool ``name``, or ``None`` if it was not called."""
        calls = self.calls_named(name)
        return calls[0] if calls else None

    @property
    def block_signature(self) -> str:
        """The response's content-block shape, e.g. ``thinking,tool_use:scratchpad,tool_use:play``.

        Recorded per response in the transcript so the block combinations the API actually
        returned are auditable after the fact rather than assumed.
        """
        parts = [
            f"tool_use:{block['name']}" if block["type"] == "tool_use" else str(block["type"])
            for block in self.raw_content
        ]
        return ",".join(parts)


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
            delay = min(delay * 2, _MAX_DELAY)
        except anthropic.APIConnectionError:
            if attempt == _MAX_ATTEMPTS:
                raise
            logger.warning(
                f"anthropic connection error (attempt {attempt}), retrying in {delay:.0f}s"
            )
            time.sleep(delay)
            delay = min(delay * 2, _MAX_DELAY)
    raise AssertionError("unreachable: loop either returns or raises")


def _wire_block(block: anthropic.types.ContentBlock) -> dict[str, object]:
    """A block as the API accepts it back — ``model_dump()`` leaks SDK-only fields
    (e.g. ``parsed_output`` on text blocks), which the API rejects with a 400."""
    if block.type == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    if block.type == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.data}
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return block.model_dump()  # unknown block types pass through best-effort


def extract_blocks(message: anthropic.types.Message) -> CompletionBlocks:
    """Flatten a response into thinking text, visible text, and every tool call it made."""
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    tool_calls: list[ToolUse] = []
    for block in message.content:
        if block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolUse(id=block.id, name=block.name, input=dict(block.input)))
    return CompletionBlocks(
        thinking="\n\n".join(thinking_parts),
        text="\n\n".join(text_parts),
        tool_calls=tool_calls,
        raw_content=[_wire_block(block) for block in message.content],
        stop_reason=message.stop_reason or "",
    )
