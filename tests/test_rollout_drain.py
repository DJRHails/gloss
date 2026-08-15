"""Drain-loop tests: a response may carry a scratchpad *and* a play call at the same time.

The player's response is scripted here (no API), so these pin the harness contract rather
than model behaviour: every scratchpad argument in a response is captured as CoT, the
``play`` call in that same response still reaches the engine, and every tool_use id the
model was handed comes back with a ``tool_result``.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock, Usage

import gloss.rollout
from gloss.models.anthropic import extract_blocks
from gloss.rollout import run_rollout

LEGAL_OPENING = "1a"  # deal #1: 6S from cascade 1 to a free cell


def pad_block(id_: str, thoughts: str) -> ToolUseBlock:
    return ToolUseBlock(id=id_, name="scratchpad", input={"thoughts": thoughts}, type="tool_use")


def play_block(id_: str, moves: str = LEGAL_OPENING) -> ToolUseBlock:
    return ToolUseBlock(id=id_, name="play", input={"moves": moves}, type="tool_use")


StopReason = Literal["end_turn", "max_tokens", "tool_use"]


def message(content: list[Any], stop_reason: StopReason = "tool_use") -> Message:
    return Message(
        id="msg_test",
        content=content,
        model="claude-opus-5",
        role="assistant",
        stop_reason=stop_reason,
        type="message",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


class ScriptedPlayer:
    """Serves scripted responses in order and records each request's message list."""

    def __init__(self, responses: list[Message]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, object]]] = []

    def __call__(self, **request: Any) -> Message:
        self.requests.append(copy.deepcopy(request["messages"]))
        if not self.responses:
            raise AssertionError("scripted player ran out of responses")
        return self.responses.pop(0)

    def tool_results(self, request_index: int) -> dict[str, str]:
        """``{tool_use_id: result text}`` from the last user message of one request."""
        content = self.requests[request_index][-1]["content"]
        assert isinstance(content, list)
        return {
            str(block["tool_use_id"]): str(block["content"])
            for block in content
            if block["type"] == "tool_result"
        }


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):
    def install(responses: list[Message]) -> ScriptedPlayer:
        player = ScriptedPlayer(responses)
        monkeypatch.setattr(gloss.rollout, "create_message", player)
        return player

    return install


def scratchpad_rollout(max_turns: int = 1):
    return run_rollout(
        game_num=1,
        agent_model="claude-opus-5",
        max_turns=max_turns,
        cot_source="scratchpad-directed",
    )


def test_extract_blocks_records_every_tool_use_in_order() -> None:
    blocks = extract_blocks(
        message(
            [
                ThinkingBlock(thinking="native", signature="sig", type="thinking"),
                pad_block("tu_pad", "PAD"),
                play_block("tu_play"),
            ]
        )
    )
    assert [call.name for call in blocks.tool_calls] == ["scratchpad", "play"]
    assert blocks.block_signature == "thinking,tool_use:scratchpad,tool_use:play"
    assert blocks.first_call("play") is not None
    assert blocks.first_call("nope") is None


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("pad first", [pad_block("tu_pad", "PAD-TEXT"), play_block("tu_play")]),
        ("play first", [play_block("tu_play"), pad_block("tu_pad", "PAD-TEXT")]),
    ],
)
def test_pad_and_play_in_one_response_keeps_both(scripted, label: str, content: list[Any]) -> None:
    player = scripted(
        [message(content), message([TextBlock(text="done", type="text")], "end_turn")]
    )
    transcript = scratchpad_rollout(max_turns=2)
    turn = transcript.turns[0]
    # The pad is the CoT column even though the same response ended the turn with `play`...
    assert turn.thinking == "PAD-TEXT", label
    # ...and the play call still reached the engine.
    assert turn.tool_call is not None and turn.tool_call.codes == [LEGAL_OPENING], label
    assert turn.moves_applied == [LEGAL_OPENING], label
    assert turn.state_after != turn.state_before, label
    assert turn.response_signatures == [extract_blocks(message(content)).block_signature], label
    # Both ids are answered, and only the play id gets the engine's result.
    results = player.tool_results(1)
    assert set(results) == {"tu_pad", "tu_play"}, label
    assert "applied 1 move" in results["tu_play"], label
    assert "Scratchpad recorded" in results["tu_pad"], label


def test_pad_only_response_drains_then_plays(scripted) -> None:
    player = scripted(
        [
            message([pad_block("tu_pad", "FIRST")]),
            message([pad_block("tu_pad2", "SECOND"), play_block("tu_play")]),
        ]
    )
    transcript = scratchpad_rollout()
    turn = transcript.turns[0]
    assert turn.thinking == "FIRST\n\nSECOND"  # every pad written this turn, in order
    assert turn.moves_applied == [LEGAL_OPENING]
    assert turn.response_signatures == ["tool_use:scratchpad", "tool_use:scratchpad,tool_use:play"]
    assert player.tool_results(1) == {"tu_pad": gloss.rollout._SCRATCHPAD_ACK}


def test_second_play_call_is_refused_not_applied(scripted) -> None:
    player = scripted(
        [
            message([play_block("tu_play"), play_block("tu_play2", "2f")]),
            message([TextBlock(text="done", type="text")], "end_turn"),
        ]
    )
    transcript = scratchpad_rollout(max_turns=2)
    assert transcript.turns[0].moves_applied == [LEGAL_OPENING]  # the second call never ran
    assert "ignored" in player.tool_results(1)["tu_play2"]


def test_native_arm_reads_thinking_and_ignores_absent_pad(scripted) -> None:
    scripted(
        [
            message(
                [
                    ThinkingBlock(thinking="NATIVE-COT", signature="sig", type="thinking"),
                    play_block("tu_play"),
                ]
            )
        ]
    )
    transcript = run_rollout(game_num=1, agent_model="claude-opus-5", max_turns=1)
    turn = transcript.turns[0]
    assert turn.thinking == "NATIVE-COT"
    assert turn.native_thinking == "NATIVE-COT"
    assert turn.moves_applied == [LEGAL_OPENING]


def test_truncated_pad_is_acked_before_the_nudge(scripted) -> None:
    """A turn that ends on a truncated pad must still answer its tool_use id, or the next call 400s.

    This is the shape PR #1's run hit three times: native thinking ate the output budget, so
    the scratchpad argument arrived cut off and no ``play`` call followed it.
    """
    player = scripted(
        [
            message([pad_block("tu_pad", "PAD")], "max_tokens"),
            message([play_block("tu_play")]),
        ]
    )
    transcript = scratchpad_rollout(max_turns=2)
    assert transcript.turns[0].tool_call is None
    assert transcript.turns[0].truncated
    assert player.tool_results(1) == {"tu_pad": gloss.rollout._SCRATCHPAD_ACK}
    nudge_content = player.requests[1][-1]["content"]
    assert isinstance(nudge_content, list)
    assert nudge_content[-1] == {"type": "text", "text": gloss.rollout._NUDGE}


def test_native_thinking_accumulates_across_the_drain(scripted) -> None:
    """Thinking from a pad-writing response counts too, or the channel census undercounts it."""
    scripted(
        [
            message(
                [
                    ThinkingBlock(thinking="FIRST", signature="sig", type="thinking"),
                    pad_block("tu_pad", "PAD"),
                ]
            ),
            message(
                [
                    ThinkingBlock(thinking="SECOND", signature="sig", type="thinking"),
                    play_block("tu_play"),
                ]
            ),
        ]
    )
    turn = scratchpad_rollout().turns[0]
    assert turn.native_thinking == "FIRST\n\nSECOND"
    assert turn.thinking == "PAD"  # the CoT column stays the pad on this arm
