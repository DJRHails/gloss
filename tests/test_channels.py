"""Channel-census tests: the two thresholds, and the response-shape counts.

The relocation claim rests on these counts, so they are pinned against hand-built turns: a
32,000-character pad beside a one-line "let me play that" must not read as a turn that used
both channels, and a native-arm transcript must not read as a turn that used neither.
"""

from __future__ import annotations

from gloss.channels import (
    SUBSTANTIVE_CHARS,
    _split,
    channel_census,
    channel_table,
    response_census,
)
from gloss.freecell import deal
from gloss.wire import CotSource, ToolCallRecord, Transcript, TurnRecord


def turn(index: int, *, cot: str, native: str, signatures: list[str] | None = None) -> TurnRecord:
    state = deal(1)
    return TurnRecord(
        turn_index=index,
        state_before=state,
        thinking=cot,
        native_thinking=native,
        assistant_text="",
        tool_call=ToolCallRecord(moves_raw="1a", codes=["1a"]),
        tool_result="applied",
        moves_applied=["1a"],
        state_after=state,
        stop_reason="tool_use",
        response_signatures=signatures or ["thinking,tool_use:play"],
    )


def transcript(cot_source: CotSource, turns: list[TurnRecord]) -> Transcript:
    return Transcript(
        transcript_id=f"game1-claude-opus-5-{cot_source}",
        game_num=1,
        agent_model="claude-opus-5",
        feedback_mode="ack",
        thinking_budget=8000,
        cot_source=cot_source,
        turns=turns,
        won=False,
    )


def test_split_separates_a_used_channel_from_a_handoff_line() -> None:
    pairs = [(32_000, 42), (42, 32_000), (2_000, 2_000), (0, 0)]
    any_content = _split(pairs, threshold=1)
    substantive = _split(pairs, threshold=SUBSTANTIVE_CHARS)
    assert (any_content.both, any_content.neither) == (3, 1)
    assert (substantive.both, substantive.pad_only, substantive.native_only) == (1, 1, 1)
    assert substantive.neither == 1


def test_census_reads_both_channels_off_a_scratchpad_arm() -> None:
    rows = transcript(
        "scratchpad-directed",
        [
            turn(0, cot="P" * 32_000, native="Let me play that."),
            turn(1, cot="", native="N" * 5_000),
        ],
    )
    census = channel_census([rows])[0]
    assert census.arm == "claude-opus-5 / scratchpad-directed"
    assert census.offers_scratchpad
    assert census.num_turns == 2
    assert census.any_content.both == 1  # the handoff line counts here...
    assert census.substantive.both == 0  # ...and not here
    assert (census.substantive.pad_only, census.substantive.native_only) == (1, 1)


def test_native_arm_has_no_pad_channel_and_says_so() -> None:
    rows = transcript("native", [turn(0, cot="N" * 5_000, native="")])
    census = channel_census([rows])[0]
    assert census.mean_pad_chars == 0.0
    # A v2 transcript predates `native_thinking`, so the CoT column is the native count.
    assert census.mean_native_chars == 5_000
    assert census.substantive.native_only == 1
    assert "n/a (no pad offered)" in channel_table([census])


def test_response_census_counts_parallel_pad_and_play() -> None:
    rows = transcript(
        "scratchpad-offered",
        [
            turn(
                0,
                cot="P",
                native="N",
                signatures=[
                    "thinking,tool_use:scratchpad,tool_use:play",
                    "tool_use:scratchpad",
                    "thinking,tool_use:play",
                ],
            )
        ],
    )
    census = response_census([rows])[0]
    assert census.num_responses == 3
    assert census.multi_tool == 1
    assert census.pad_and_play == 1
    assert census.signatures["tool_use:scratchpad"] == 1
