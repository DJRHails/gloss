"""Engine tests: canonical Microsoft deals, move legality, supermoves, and win detection."""

from __future__ import annotations

import pytest

from gloss.freecell import (
    GameState,
    IllegalMove,
    apply_move,
    apply_sequence,
    deal,
    normalize_card,
    parse_move,
    supermove_capacity,
)

# The canonical layouts for Microsoft deals #1 and #617, row-major as dealt
# (e.g. Rosetta Code, "Deal cards for FreeCell"). Any engine sharing the MS
# numbering must reproduce these exactly.
GAME_1_ROWS = """
JD 2D 9H JC 5D 7H 7C 5H
KD KC 9S 5S AD QC KH 3H
2S KS 9D QD JS AS AH 3C
4C 5C TS QH 4H AC 4D 7S
3S TD 4S TH 8H 2C JH 7D
6D 8S 8D QS 6C 3D 8C TC
6S 9C 2H 6H
"""

GAME_617_ROWS = """
7D AD 5C 3S 5S 8C 2D AH
TD 7S QD AC 6D 8H AS KH
TH QC 3H 9D 6S 8D 3D TC
KD 5H 9S 3C 8S 7H 4D JS
4C QS 9C 9H 7C 6H 2C 2S
4S TS 2H 5D JC 6C JH QH
JD KS KC 4H
"""


def columns_from_rows(rows_text: str) -> list[list[str]]:
    rows = [line.split() for line in rows_text.strip().splitlines()]
    columns: list[list[str]] = [[] for _ in range(8)]
    for row in rows:
        for index, card in enumerate(row):
            columns[index].append(card)
    return columns


@pytest.mark.parametrize(
    ("game_num", "rows_text"), [(1, GAME_1_ROWS), (617, GAME_617_ROWS)], ids=["game-1", "game-617"]
)
def test_microsoft_deal_matches_canonical_layout(game_num: int, rows_text: str) -> None:
    assert deal(game_num).cascades == columns_from_rows(rows_text)


def test_deal_is_a_full_deck() -> None:
    state = deal(11982)
    cards = [card for column in state.cascades for card in column]
    assert len(cards) == 52
    assert len(set(cards)) == 52
    assert [len(column) for column in state.cascades] == [7, 7, 7, 7, 6, 6, 6, 6]
    assert state.free == {"a": None, "b": None, "c": None, "d": None}
    assert state.foundations == {"C": 0, "D": 0, "H": 0, "S": 0}


def test_normalize_card_accepts_common_spellings() -> None:
    assert normalize_card("10h") == "TH"
    assert normalize_card(" td ") == "TD"
    assert normalize_card("A♠") == "AS"
    with pytest.raises(ValueError, match="unrecognised card"):
        normalize_card("1H")


def test_parse_move_rejects_bad_codes() -> None:
    for bad in ("", "9a", "a", "1x", "hh", "44"):
        with pytest.raises(IllegalMove):
            parse_move(bad)
    assert parse_move(" 3H ").code == "3h"


def simple_state(**overrides: object) -> GameState:
    base: dict[str, object] = {
        "free": {"a": None, "b": None, "c": None, "d": None},
        "foundations": {"C": 0, "D": 0, "H": 0, "S": 0},
        "cascades": [[] for _ in range(8)],
    }
    base.update(overrides)
    return GameState(**base)


def test_ace_then_two_to_foundation() -> None:
    state = simple_state(cascades=[["AC"], ["2C"], [], [], [], [], [], []])
    state, description = apply_move(state, "1h")
    assert description == "AC to foundation"
    assert state.foundations["C"] == 1
    state, _ = apply_move(state, "2h")
    assert state.foundations["C"] == 2
    with pytest.raises(IllegalMove, match="cannot go home"):
        apply_move(simple_state(cascades=[["2C"]] + [[]] * 7), "1h")


def test_tableau_build_requires_alternating_descending() -> None:
    state = simple_state(cascades=[["8H"], ["9S"], ["9H"], [], [], [], [], []])
    state, _ = apply_move(state, "12")  # 8H onto 9S: ok
    assert state.cascades[1] == ["9S", "8H"]
    with pytest.raises(IllegalMove, match="fits on 9H"):
        apply_move(simple_state(cascades=[["8H"], ["9H"], [], [], [], [], [], []]), "12")


def test_free_cell_round_trip() -> None:
    state = simple_state(cascades=[["9S", "8H"], [], [], [], [], [], [], []])
    state, _ = apply_move(state, "1f")  # 8H to first empty cell (a)
    assert state.free["a"] == "8H"
    state, _ = apply_move(state, "a2")  # back down to an empty cascade
    assert state.cascades[1] == ["8H"]
    with pytest.raises(IllegalMove, match="already holds"):
        occupied = simple_state(
            free={"a": "KC", "b": None, "c": None, "d": None},
            cascades=[["9S"], [], [], [], [], [], [], []],
        )
        apply_move(occupied, "1a")


def test_supermove_moves_longest_fitting_run() -> None:
    state = simple_state(cascades=[["KC", "9H", "8S", "7H"], ["TC"], [], [], [], [], [], []])
    state, description = apply_move(state, "12")
    assert description == "9H 8S 7H to cascade 2"
    assert state.cascades[0] == ["KC"]
    assert state.cascades[1] == ["TC", "9H", "8S", "7H"]


def test_supermove_capacity_blocks_oversized_run() -> None:
    # No free cells, no empty cascades: capacity 1, so a 2-card run cannot move.
    state = simple_state(
        free={"a": "KC", "b": "KD", "c": "KH", "d": "KS"},
        cascades=[["8S", "7H"], ["2C", "9H"], ["2D"], ["2H"], ["2S"], ["3C"], ["3D"], ["3H"]],
    )
    assert supermove_capacity(state, dst_is_empty_cascade=False) == 1
    with pytest.raises(IllegalMove, match="capacity"):
        apply_move(state, "12")


def test_supermove_to_empty_cascade_excludes_destination_from_multiplier() -> None:
    state = simple_state(cascades=[["8S", "7H", "6S"], [], [], [], [], [], [], []])
    # 4 free cells, 7 empty cascades minus the destination: capacity is huge; whole run moves.
    state, _ = apply_move(state, "12")
    assert state.cascades[1] == ["8S", "7H", "6S"]
    # With no cells and only the destination empty, capacity is 1: only 6S moves.
    tight = simple_state(
        free={"a": "KC", "b": "KD", "c": "KH", "d": "KS"},
        cascades=[["8S", "7H", "6S"], [], ["2D"], ["2H"], ["2S"], ["3C"], ["3D"], ["3H"]],
    )
    moved, _ = apply_move(tight, "12")
    assert moved.cascades[0] == ["8S", "7H"]
    assert moved.cascades[1] == ["6S"]


def test_apply_move_is_pure() -> None:
    state = simple_state(cascades=[["AC"], [], [], [], [], [], [], []])
    before = state.model_dump()
    apply_move(state, "1h")
    assert state.model_dump() == before


def test_apply_sequence_stops_at_first_illegal_move() -> None:
    state = simple_state(cascades=[["AC"], ["AD"], [], [], [], [], [], []])
    final, applied, error = apply_sequence(state, ["1h", "2h", "1h"])
    assert [line.split(":")[0] for line in applied] == ["1h", "2h"]
    assert error is not None and "1h" in error
    assert final.foundations == {"C": 1, "D": 1, "H": 0, "S": 0}


def test_win_detection_and_render() -> None:
    won = simple_state(foundations={"C": 13, "D": 13, "H": 13, "S": 13})
    assert won.is_won()
    board = deal(1).render()
    assert board.splitlines()[0] == "free: a:-- b:-- c:-- d:--"
    assert board.splitlines()[1] == "home: C:- D:- H:- S:-"
    assert board.splitlines()[2].startswith("1: JD KD 2S 4C 3S 6D 6S")
