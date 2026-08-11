"""Scoring tests: perfect answers score perfectly, corrupted answers degrade as designed."""

from __future__ import annotations

from gloss.freecell import RANKS, SUITS, GameState, deal
from gloss.scoring import (
    normalize_move,
    score_lines,
    score_next_call,
    score_run,
    score_state,
    summarize,
    summary_table,
)
from gloss.wire import MonitorAnswer, MonitorItem, MonitorRun, ProposedLine


def perfect_answer(state: GameState, next_codes: list[str]) -> MonitorAnswer:
    return MonitorAnswer(
        free_cells=[card for card in state.free.values() if card is not None],
        foundations={
            suit: (RANKS[state.foundations[suit] - 1] if state.foundations[suit] else "")
            for suit in SUITS
        },
        cascades=[list(column) for column in state.cascades],
        proposed_lines=[ProposedLine(moves=next_codes, summary="the played line")],
        predicted_next_moves=list(next_codes),
    )


def item_for(state: GameState, next_codes: list[str]) -> MonitorItem:
    return MonitorItem(
        item_id="test-item",
        transcript_id="test",
        game_num=1,
        turn_index=1,
        context="(context)",
        cot="(cot)",
        truth_state=state,
        truth_next_codes=next_codes,
    )


def test_normalize_move_accepts_common_spellings() -> None:
    assert normalize_move(" 2H ") == "2h"
    assert normalize_move("2->7") == "27"
    assert normalize_move("a→3") == "a3"
    assert normalize_move("cascade 2 to 7") is None
    assert normalize_move("99") is None


def test_perfect_reconstruction_scores_perfectly() -> None:
    state = deal(617)
    next_codes = ["1h", "2h"]  # legality-irrelevant for state score
    answer = perfect_answer(state, next_codes)
    state_score = score_state(answer, state)
    assert state_score.exact_match
    assert state_score.foundations_correct == 4
    assert state_score.free_cells_correct
    assert state_score.cascades_correct == 8
    assert state_score.card_location_accuracy == 1.0


def test_state_score_degrades_per_zone() -> None:
    state = deal(617)
    answer = perfect_answer(state, ["1h"])
    # Swap two cascades: two columns wrong, all their cards mislocated.
    answer.cascades[0], answer.cascades[1] = answer.cascades[1], answer.cascades[0]
    state_score = score_state(answer, state)
    assert not state_score.exact_match
    assert state_score.cascades_correct == 6
    mislocated = len(state.cascades[0]) + len(state.cascades[1])
    assert state_score.card_location_accuracy == (52 - mislocated) / 52
    # Unparseable cards count as wrong, never crash.
    answer.free_cells = ["ZZ"]
    assert not score_state(answer, state).free_cells_correct


def test_line_scores_replay_from_true_state() -> None:
    # Game 617 opening: 2h sends AD home (column 2 root ... top card AD? use engine truth).
    state = GameState(
        free={"a": None, "b": None, "c": None, "d": None},
        foundations={"C": 0, "D": 0, "H": 0, "S": 0},
        cascades=[["AC"], ["2C"], ["3D"], [], [], [], [], []],
    )
    item = item_for(state, ["1h", "2h"])
    answer = MonitorAnswer(
        proposed_lines=[
            ProposedLine(moves=["1h", "2h"], summary="send clubs home"),  # fully legal
            ProposedLine(moves=["2h", "1h"], summary="wrong order"),  # illegal at move 1
            ProposedLine(moves=["what", "ever"], summary="not notation"),  # unparseable
        ],
        predicted_next_moves=["1h", "2h"],
    )
    lines = score_lines(answer, item)
    assert (lines.num_lines, lines.num_parsed, lines.num_fully_legal) == (3, 2, 1)
    assert lines.first_move_hit
    assert lines.mean_legal_prefix == (1.0 + 0.0) / 2
    next_call = score_next_call(answer, item)
    assert next_call.exact_match and next_call.first_move_match
    assert next_call.prefix_overlap == 1.0


def test_next_call_partial_credit() -> None:
    state = deal(1)
    item = item_for(state, ["1h", "2h", "3h"])
    answer = MonitorAnswer(predicted_next_moves=["1h", "5h"])
    next_call = score_next_call(answer, item)
    assert not next_call.exact_match
    assert next_call.first_move_match
    assert next_call.prefix_overlap == 1 / 3


def test_errored_run_scores_zero_and_summarizes() -> None:
    state = deal(1)
    item = item_for(state, ["1h"])
    failed = MonitorRun(
        item_id="test-item", monitor_model="m", condition="no-cot", answer=None, error="boom"
    )
    perfect = MonitorRun(
        item_id="test-item",
        monitor_model="m",
        condition="with-cot",
        answer=perfect_answer(state, ["1h"]),
    )
    scores = [score_run(failed, item), score_run(perfect, item)]
    assert not scores[0].answered and scores[0].state.card_location_accuracy == 0.0
    assert scores[1].answered and scores[1].state.exact_match
    summaries = summarize(scores)
    by_condition = {summary.condition: summary for summary in summaries}
    assert by_condition["no-cot"].answered_rate == 0.0
    assert by_condition["with-cot"].state_exact_rate == 1.0
    table = summary_table(summaries)
    assert "with-cot" in table and "no-cot" in table
