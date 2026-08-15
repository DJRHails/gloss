"""Yield-census tests: every recorded turn is either an item or attributed to one exclusion.

The census exists to answer "where did the items go?" with numbers, so its arithmetic has to
close: items + excluded == turns, and the reason it reports has to be the reason ``build_items``
actually applied.
"""

from __future__ import annotations

import pytest

from gloss.freecell import apply_sequence, deal
from gloss.monitor import (
    arm_yields,
    build_items,
    item_exclusion,
    moves_per_call,
    transcript_yield,
    yield_statement,
)
from gloss.wire import ToolCallRecord, Transcript, TurnRecord


def turn(
    index: int,
    *,
    moves: str | None = "1a",
    thinking: str = "COT",
    truncated: bool = False,
) -> TurnRecord:
    state = deal(1)
    after, _applied, _error = apply_sequence(state, ["1a"])
    codes = moves.split() if moves is not None else []
    return TurnRecord(
        turn_index=index,
        state_before=state,
        thinking=thinking,
        truncated=truncated,
        assistant_text="",
        tool_call=ToolCallRecord(moves_raw=moves, codes=codes) if moves is not None else None,
        tool_result="applied" if moves is not None else None,
        moves_applied=codes[:1],
        state_after=after,
        stop_reason="tool_use",
    )


def transcript(turns: list[TurnRecord], *, won: bool = False) -> Transcript:
    return Transcript(
        transcript_id="game1-claude-opus-5",
        game_num=1,
        agent_model="claude-opus-5",
        feedback_mode="ack",
        thinking_budget=8000,
        turns=turns,
        won=won,
    )


@pytest.mark.parametrize(
    ("label", "record", "expected"),
    [
        ("turn 0 is structural", turn(0), "no-prior-history"),
        ("prose answer", turn(1, moves=None), "no-tool-call"),
        (
            "truncation outranks the empty pad it caused",
            turn(1, thinking="", truncated=True),
            "truncated",
        ),
        ("nothing verbalised", turn(1, thinking="  "), "empty-cot"),
        ("eligible", turn(1), None),
    ],
)
def test_item_exclusion_priority(label: str, record: TurnRecord, expected: str | None) -> None:
    assert item_exclusion(record) == expected, label


def test_census_arithmetic_closes_against_build_items() -> None:
    turns = [
        turn(0),  # structural
        turn(1),  # item
        turn(2, moves=None),  # no tool call
        turn(3, thinking=""),  # empty cot
        turn(4, thinking="", truncated=True),  # truncated
        turn(5, moves="1a 2f 3h 45 56 67 78 81 12 23 34"),  # item, 11-move batch
    ]
    census = transcript_yield(transcript(turns))
    assert census.num_turns == 6
    assert census.num_items == len(build_items(transcript(turns))) == 2
    assert census.excluded == {
        "no-prior-history": 1,
        "no-tool-call": 1,
        "truncated": 1,
        "empty-cot": 1,
    }
    assert census.num_items + sum(census.excluded.values()) == census.num_turns


def test_moves_per_call_buckets_the_batch_sizes() -> None:
    stats = moves_per_call([turn(1), turn(2, moves=" ".join(["1a"] * 31))])
    assert stats.num_calls == 2
    assert stats.maximum == 31
    assert stats.buckets == {"1": 1, "2-5": 0, "6-15": 0, "16-30": 0, "31+": 1}
    # A turn with no play call contributes nothing rather than a zero-move call.
    assert moves_per_call([turn(1, moves=None)]).num_calls == 0


def test_arm_yield_and_statement_name_the_dominant_loss() -> None:
    summaries = arm_yields([transcript([turn(0), turn(1)]), transcript([turn(0), turn(1)])])
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.arm == "claude-opus-5 / native"
    assert (summary.num_deals, summary.num_turns, summary.num_items) == (2, 4, 2)
    assert summary.items_per_deal == 1.0
    statement = yield_statement(summary)
    assert "2 deals (4 turns) yield 2 items" in statement
    assert "no-prior-history" in statement and "2 of 2 excluded turns" in statement
