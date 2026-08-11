"""Item-building tests: the monitor context must never leak the current turn's CoT or call."""

from __future__ import annotations

import pytest

from gloss.freecell import apply_sequence, deal
from gloss.monitor import _cot_section, build_items, render_context, swapped_cot_donors
from gloss.wire import MonitorItem, ToolCallRecord, Transcript, TurnRecord


def make_transcript() -> Transcript:
    state0 = deal(1)
    # Any applied move works for the fixture; item building never re-checks legality.
    state1, _applied, _error = apply_sequence(state0, ["1a"])
    turns = [
        TurnRecord(
            turn_index=0,
            state_before=state0,
            thinking="TURN0-SECRET-THINKING",
            assistant_text="I'll start with the ace.",
            tool_call=ToolCallRecord(moves_raw="3h", codes=["3h"]),
            tool_result="applied 1 move(s):\n  3h: AH to foundation",
            moves_applied=["3h"],
            state_after=state1,
            stop_reason="tool_use",
        ),
        TurnRecord(
            turn_index=1,
            state_before=state1,
            thinking="TURN1-SECRET-THINKING",
            assistant_text="",
            tool_call=ToolCallRecord(moves_raw="52 61", codes=["52", "61"]),
            tool_result="applied 2 move(s)",
            moves_applied=["52", "61"],
            state_after=state1,
            stop_reason="tool_use",
        ),
    ]
    return Transcript(
        transcript_id="game1-test",
        game_num=1,
        agent_model="test-model",
        feedback_mode="ack",
        thinking_budget=1000,
        turns=turns,
        won=False,
    )


def test_context_strips_all_prior_thinking_and_current_turn() -> None:
    transcript = make_transcript()
    context = render_context(transcript, upto_turn=1)
    assert "TURN0-SECRET-THINKING" not in context
    assert "TURN1-SECRET-THINKING" not in context
    assert 'play("3h")' in context  # prior tool call IS visible
    assert "52 61" not in context  # current turn's call is not
    assert "AH to foundation" in context  # prior tool result IS visible
    assert context.startswith("=== INITIAL BOARD")


def test_build_items_selects_turns_with_history_call_and_thinking() -> None:
    transcript = make_transcript()
    items = build_items(transcript)
    assert [item.turn_index for item in items] == [1]
    item = items[0]
    assert item.item_id == "game1-test-t1"
    assert item.cot == "TURN1-SECRET-THINKING"
    assert item.truth_next_codes == ["52", "61"]
    assert item.truth_state == transcript.turns[1].state_before
    # The context never contains the answer key.
    assert "TURN1-SECRET-THINKING" not in item.context


def make_item(item_id: str, transcript_id: str, cot: str) -> MonitorItem:
    return MonitorItem(
        item_id=item_id,
        transcript_id=transcript_id,
        game_num=1,
        turn_index=1,
        context="(context)",
        cot=cot,
        truth_state=deal(1),
        truth_next_codes=["1h"],
    )


def test_swapped_cot_donors_cross_transcript_only() -> None:
    items = [
        make_item("a-t1", "a", "COT-A1"),
        make_item("a-t2", "a", "COT-A2"),
        make_item("b-t1", "b", "COT-B1"),
    ]
    donors = swapped_cot_donors(items)
    # Every donor comes from a different transcript than the recipient.
    assert donors == {"a-t1": "COT-B1", "a-t2": "COT-B1", "b-t1": "COT-A1"}
    # Single-transcript dataset: the control is undefined, no donors.
    assert swapped_cot_donors(items[:2]) == {}


def test_cot_section_renders_conditions_identically() -> None:
    item = make_item("a-t1", "a", "REAL-COT")
    real = _cot_section(item, "with-cot", None)
    swapped = _cot_section(item, "swapped-cot", "DONOR-COT")
    # The monitor must not be able to tell the conditions apart by framing.
    assert real.replace("REAL-COT", "X") == swapped.replace("DONOR-COT", "X")
    assert "DONOR-COT" in swapped and "REAL-COT" not in swapped
    with pytest.raises(ValueError, match="donor"):
        _cot_section(item, "swapped-cot", None)
