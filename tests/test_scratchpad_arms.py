"""The two scratchpad arms: same tool, two wordings, and they must not collide in a dataset.

`scratchpad-directed` is the wording PR #1 ran ("do your thinking there — the whole of it");
`scratchpad-offered` is the control that only says the tool exists. If the mild arm still shows
no turn using both channels, the relocation is a property of the model under any pad offer; if
it does, the forceful wording caused it. These tests pin the plumbing that comparison needs.
"""

from __future__ import annotations

from typing import get_args

from gloss.freecell import apply_sequence, deal
from gloss.monitor import build_items
from gloss.prompts import load_prompt
from gloss.rollout import SCRATCHPAD_ADDENDA, player_system, player_tools, transcript_id
from gloss.wire import CotSource, ToolCallRecord, Transcript, TurnRecord

SCRATCHPAD_ARMS = tuple(SCRATCHPAD_ADDENDA)


def test_every_arm_is_either_native_or_has_an_addendum() -> None:
    """A new CotSource without an addendum would silently run as the native arm."""
    assert {"native", *SCRATCHPAD_ARMS} == set(get_args(CotSource))


def flat(name: str) -> str:
    """A prompt with its line wrapping collapsed, so assertions can quote whole phrases."""
    return " ".join(load_prompt(name).split())


def test_only_the_directed_wording_directs_where_thinking_goes() -> None:
    directed = flat(SCRATCHPAD_ADDENDA["scratchpad-directed"])
    offered = flat(SCRATCHPAD_ADDENDA["scratchpad-offered"])
    for prompt in (directed, offered):
        assert "`scratchpad`" in prompt and "`play`" in prompt  # both offer the same tool
    # The directed arm instructs that the pad carries the reasoning and should run long...
    assert "do your thinking there — the whole of it, not a summary of it" in directed
    assert "the right place for long reasoning" in directed
    # ...and the control must not, or it is not a control.
    assert "thinking" not in offered
    assert "long" not in offered


def test_arm_selection_drives_prompt_and_tools() -> None:
    native_system = player_system(feedback="ack", cot_source="native")
    assert [tool["name"] for tool in player_tools("native")] == ["play"]
    assert "scratchpad" not in native_system
    for arm in SCRATCHPAD_ARMS:
        system = player_system(feedback="ack", cot_source=arm)
        assert system.startswith(native_system)  # the arms differ only by the addendum
        assert system.endswith(load_prompt(SCRATCHPAD_ADDENDA[arm]))
        assert [tool["name"] for tool in player_tools(arm)] == ["scratchpad", "play"]


def one_turn_transcript(cot_source: CotSource, *, cot: str) -> Transcript:
    """A two-turn transcript on deal #1 whose second turn carries ``cot`` as its CoT column."""
    state0 = deal(1)
    state1, _applied, _error = apply_sequence(state0, ["1a"])
    turns = [
        TurnRecord(
            turn_index=index,
            state_before=state0 if index == 0 else state1,
            thinking=cot,
            native_thinking="NATIVE",
            assistant_text="",
            tool_call=ToolCallRecord(moves_raw="1a", codes=["1a"]),
            tool_result="applied 1 move(s)",
            moves_applied=["1a"],
            state_after=state1,
            stop_reason="tool_use",
            response_signatures=["thinking,tool_use:play"],
        )
        for index in range(2)
    ]
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


def test_an_ignored_pad_yields_no_item_rather_than_an_empty_one() -> None:
    """The mild arm may leave the pad unused; that turn is unscorable, not a blank-CoT item."""
    assert build_items(one_turn_transcript("scratchpad-offered", cot="")) == []
    assert len(build_items(one_turn_transcript("scratchpad-offered", cot="PAD"))) == 1


def test_arms_keep_distinct_transcript_ids_on_the_same_deal() -> None:
    ids = [
        transcript_id(game_num=617, agent_model="claude-opus-5", cot_source=arm)
        for arm in ("native", *SCRATCHPAD_ARMS)
    ]
    assert ids == [
        "game617-claude-opus-5",  # the v2 corpus's bare form, unchanged
        "game617-claude-opus-5-scratchpad-directed",
        "game617-claude-opus-5-scratchpad-offered",
    ]
