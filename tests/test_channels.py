"""Guards for the channel-split measurement and the two knobs that power the run.

The load-bearing risk here is the *metric*, not the API: co-use must condition on a pad actually
being written (an unused pad tool otherwise reads as relocation), truncated turns must never be
scored, and the native arm must report zero pad use or the pad detector is firing on something else.
"""

from __future__ import annotations

from typing import cast

import pytest

from gloss.channels import classify_turn, split_for
from gloss.freecell import deal
from gloss.rollout import (
    MOVES_PER_CALL_UNCAPPED,
    SCRATCHPAD_DESCRIPTIONS,
    _cap_codes,
    _play_tool,
    _scratchpad_tool,
)
from gloss.utils.stats import wilson_ci
from gloss.wire import SCRATCHPAD_SOURCES, CotSource, Transcript, TurnRecord


def _turn(
    *, pad: str = "", native: str = "", truncated: bool = False, index: int = 0
) -> TurnRecord:
    """A turn record with only the channel fields that matter set.

    ``thinking`` is the CoT column: the pad in a scratchpad arm, the API blocks in the native arm.
    """
    state = deal(1)
    return TurnRecord(
        turn_index=index,
        state_before=state,
        thinking=pad,
        native_thinking=native,
        truncated=truncated,
        assistant_text="",
        tool_call=None,
        tool_result=None,
        moves_applied=[],
        state_after=state,
        stop_reason="tool_use",
    )


def _transcript(cot_source: str, turns: list[TurnRecord]) -> Transcript:
    return Transcript(
        transcript_id=f"t-{cot_source}",
        game_num=1,
        agent_model="claude-opus-5",
        feedback_mode="ack",
        thinking_budget=8000,
        cot_source=cast(CotSource, cot_source),
        turns=turns,
        won=False,
    )


# --- Wilson intervals ------------------------------------------------------------------------


def test_wilson_covers_the_boundary_cells_without_leaving_the_unit_interval() -> None:
    """0/n and n/n are where these rates live, and where a Wald interval breaks."""
    for successes, total in ((0, 10), (10, 10), (1, 1)):
        point, low, high = wilson_ci(successes, total)
        assert 0.0 <= low <= point <= high <= 1.0
        assert high > low, "a boundary cell must still carry width"


def test_wilson_reports_no_data_rather_than_a_rate() -> None:
    assert wilson_ci(0, 0) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("total", range(1, 60))
def test_the_wilson_interval_always_brackets_its_point_estimate(total: int) -> None:
    """A bound on the wrong side of the point is float error, and it renders as an impossible CI.

    Caught at 10/10, where the upper bound divides out to 0.9999999999999999 — just under a point
    estimate of exactly 1.0. Swept over every cell rather than the one that happened to fail.
    """
    for successes in range(total + 1):
        point, low, high = wilson_ci(successes, total)
        assert 0.0 <= low <= point <= high <= 1.0, f"{successes}/{total} -> [{low}, {high}]"


@pytest.mark.parametrize(("successes", "total"), [(-1, 5), (6, 5)])
def test_wilson_rejects_impossible_counts(successes: int, total: int) -> None:
    with pytest.raises(ValueError, match="out of range|non-negative"):
        wilson_ci(successes, total)


# --- channel classification -----------------------------------------------------------------


def test_classify_turn_separates_the_four_channel_combinations() -> None:
    assert classify_turn(_turn(pad="p", native="n"), pad_is_cot=True) == "both"
    assert classify_turn(_turn(pad="p"), pad_is_cot=True) == "pad_only"
    assert classify_turn(_turn(native="n"), pad_is_cot=True) == "native_only"
    assert classify_turn(_turn(), pad_is_cot=True) == "neither"


def test_the_native_arm_can_never_register_a_pad() -> None:
    """In the native arm `thinking` holds the API blocks, not a pad — counting it double-counts.

    This is the null control: the pad tool is never offered there, so pad use must be 0 by
    construction, and a non-zero reading means the detector is measuring something else.
    """
    turn = _turn(pad="these are API thinking blocks", native="these are API thinking blocks")
    assert classify_turn(turn, pad_is_cot=False) == "native_only"
    split = split_for([_transcript("native", [turn])], "native")
    assert split.pad_use.successes == 0
    assert split.pad_use.point == 0.0


# --- the relocation metric ------------------------------------------------------------------


def test_co_use_conditions_on_a_pad_being_written() -> None:
    """An arm that never uses the pad has no co-use to report — it must not read as relocation.

    Unconditionally, "0% of turns used both channels" is true both when the model relocates its
    reasoning into the pad AND when it ignores the pad entirely. Only the conditional separates
    those two worlds.
    """
    ignored_pad = _transcript("scratchpad_offered", [_turn(native="n", index=i) for i in range(8)])
    split = split_for([ignored_pad], "scratchpad_offered")
    assert split.mix["both"] == 0, "no turn used both"
    assert split.pad_use.point == 0.0, "because the pad was never written"
    assert split.co_use_given_pad.n == 0, "so co-use has no denominator and must be no-data"
    assert split.co_use_given_pad.render().startswith("no data")


def test_relocation_and_co_use_are_distinguishable_on_the_same_pad_rate() -> None:
    """Same pad-use rate, opposite co-use: the metric must separate these two worlds."""
    relocating = _transcript("scratchpad_urged", [_turn(pad="p", index=i) for i in range(6)])
    complementary = _transcript(
        "scratchpad_urged", [_turn(pad="p", native="n", index=i) for i in range(6)]
    )
    moved = split_for([relocating], "scratchpad_urged")
    both = split_for([complementary], "scratchpad_urged")
    assert moved.pad_use.point == both.pad_use.point == 1.0
    assert moved.co_use_given_pad.point == 0.0, "relocation: pad written, native silent"
    assert both.co_use_given_pad.point == 1.0, "complements: both channels on every turn"


def test_truncated_turns_are_excluded_not_scored_as_an_empty_pad() -> None:
    """A pad clipped by the output cap arrives as `{}` — scoring it charges our budget to the model.

    This is the bug that invalidated the first scratchpad run: every "empty pad" was a truncation.
    """
    turns = [
        _turn(pad="p", native="n", index=0),
        _turn(pad="", native="", truncated=True, index=1),
        _turn(pad="", native="", truncated=True, index=2),
    ]
    split = split_for([_transcript("scratchpad_urged", turns)], "scratchpad_urged")
    assert split.n_turns == 1, "only the intact turn is scorable"
    assert split.n_truncated == 2, "and the truncations are reported, not hidden"
    assert split.co_use_given_pad.point == 1.0


def test_a_missing_cot_source_reads_as_the_native_arm() -> None:
    """The v1/v2 transcripts predate the field, so None must mean native, not a dropped row."""
    transcript = _transcript("native", [_turn(native="n")])
    raw = transcript.model_dump()
    raw["cot_source"] = None
    revived = Transcript.model_validate({**raw, "cot_source": "native"})
    assert split_for([revived], "native").n_turns == 1


# --- the two run knobs ----------------------------------------------------------------------


def test_the_move_cap_trims_and_reports_the_overflow() -> None:
    assert _cap_codes(["3h", "27", "1a"], 2) == (["3h", "27"], 1)
    assert _cap_codes(["3h", "27"], 2) == (["3h", "27"], 0)
    assert _cap_codes(["3h"] * 30, MOVES_PER_CALL_UNCAPPED) == (["3h"] * 30, 0)


def test_the_cap_is_declared_in_the_tool_description() -> None:
    """An undeclared cap silently discards most of every call and reads as illegal play."""
    assert "At most 3 move(s)" in str(_play_tool(3)["description"])
    assert "At most" not in str(_play_tool(MOVES_PER_CALL_UNCAPPED)["description"])


def test_the_two_scratchpad_arms_differ_only_in_wording() -> None:
    """Same tool name and schema, different description — or the contrast confounds the control."""
    urged = _scratchpad_tool(SCRATCHPAD_DESCRIPTIONS["scratchpad_urged"])
    offered = _scratchpad_tool(SCRATCHPAD_DESCRIPTIONS["scratchpad_offered"])
    assert urged["name"] == offered["name"] == "scratchpad"
    assert urged["input_schema"] == offered["input_schema"]
    assert urged["description"] != offered["description"]
    assert set(SCRATCHPAD_DESCRIPTIONS) == set(SCRATCHPAD_SOURCES)


def test_the_offered_wording_makes_no_claim_about_where_reasoning_lives() -> None:
    """The urged wording asserts the pad IS the reasoning channel; the control must not."""
    offered = SCRATCHPAD_DESCRIPTIONS["scratchpad_offered"].lower()
    assert "reasoning" not in offered
    assert "optional" in offered
    assert "reasoning" in SCRATCHPAD_DESCRIPTIONS["scratchpad_urged"].lower()
