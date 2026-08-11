"""FreeCell engine: Microsoft deal numbering, legal moves, and a compact board render.

The engine is the benchmark's ground truth. Every transcript turn records the exact
:class:`GameState` before the agent's move, and every monitor answer (a reconstructed
state, a proposed line) is scored by replaying it here — so the engine must be pure and
deterministic. States are immutable from the caller's perspective: :func:`apply_move`
returns a new state and never mutates its input.

Deals use the Microsoft FreeCell numbering (the LCG every solver and UI shares), so
``deal(617)`` here is the same layout as "game #617" anywhere else.

Moves use standard FreeCell notation: two characters, source then destination.
Sources/destinations are cascades ``1``-``8``, free cells ``a``-``d``, and destinations
additionally ``f`` (first empty free cell) and ``h`` (home/foundation, suit implied by
the moved card). A cascade-to-cascade move transfers the longest legal run that fits,
capped by the standard supermove capacity ``(1 + empty cells) * 2^(empty cascades)``
(the destination, if empty, does not count toward the multiplier).
"""

from __future__ import annotations

from pydantic import BaseModel

RANKS = "A23456789TJQK"
SUITS = "CDHS"
RED_SUITS = frozenset("DH")
CELL_NAMES = "abcd"
NUM_CASCADES = 8

Card = str  # two characters, rank then suit: "AS", "TD", "9H"


class IllegalMove(ValueError):
    """A move that cannot be applied to the given state; the message says why."""


def rank_of(card: Card) -> int:
    """Numeric rank, ace low: A=1 .. K=13."""
    return RANKS.index(card[0]) + 1


def suit_of(card: Card) -> str:
    return card[1]


def is_red(card: Card) -> bool:
    return card[1] in RED_SUITS


def normalize_card(raw: str) -> Card:
    """Canonical two-char card from common spellings ("10h", "td", "T♥") or raise ValueError."""
    text = raw.strip().upper().replace("10", "T")
    for glyph, letter in (("♣", "C"), ("♦", "D"), ("♥", "H"), ("♠", "S")):
        text = text.replace(glyph, letter)
    if len(text) != 2 or text[0] not in RANKS or text[1] not in SUITS:
        raise ValueError(f"unrecognised card {raw!r} (expected e.g. 'AS', 'TD', '9H')")
    return text


def _fits_on_cascade(card: Card, below: Card) -> bool:
    """Tableau rule: one rank lower, opposite colour."""
    return rank_of(card) == rank_of(below) - 1 and is_red(card) != is_red(below)


def deal(game_num: int) -> GameState:
    """The Microsoft deal for ``game_num`` (the shared numbering of solvers and UIs)."""
    seed = game_num

    def rand() -> int:
        nonlocal seed
        seed = (seed * 214013 + 2531011) % 2**31
        return seed >> 16

    deck = [RANKS[i // 4] + SUITS[i % 4] for i in range(52)]
    cascades: list[list[Card]] = [[] for _ in range(NUM_CASCADES)]
    for position in range(52):
        pick = rand() % len(deck)
        deck[pick], deck[-1] = deck[-1], deck[pick]
        cascades[position % NUM_CASCADES].append(deck.pop())
    return GameState(
        free={name: None for name in CELL_NAMES},
        foundations=dict.fromkeys(SUITS, 0),
        cascades=cascades,
    )


class GameState(BaseModel, frozen=True):
    """Full FreeCell position: four free cells, four foundations, eight cascades.

    ``free`` maps cell name (``a``-``d``) to its card or ``None``. ``foundations`` maps
    suit to the top rank already home (0 = empty). ``cascades`` lists columns root-first.
    """

    free: dict[str, Card | None]
    foundations: dict[str, int]
    cascades: list[list[Card]]

    def is_won(self) -> bool:
        return all(self.foundations[suit] == 13 for suit in SUITS)

    def empty_cells(self) -> list[str]:
        return [name for name in CELL_NAMES if self.free[name] is None]

    def render(self) -> str:
        """Compact text board — the exact render agents see and monitors reproduce."""
        cells = " ".join(f"{name}:{self.free[name] or '--'}" for name in CELL_NAMES)
        homes = " ".join(
            f"{suit}:{RANKS[self.foundations[suit] - 1] if self.foundations[suit] else '-'}"
            for suit in SUITS
        )
        columns = "\n".join(
            f"{index + 1}: {' '.join(column) if column else '(empty)'}"
            for index, column in enumerate(self.cascades)
        )
        return f"free: {cells}\nhome: {homes}\n{columns}"


class ParsedMove(BaseModel, frozen=True):
    """A validated move code: source and destination in standard notation."""

    src: str  # '1'-'8' or 'a'-'d'
    dst: str  # '1'-'8', 'a'-'d', 'f', or 'h'

    @property
    def code(self) -> str:
        return f"{self.src}{self.dst}"


def parse_move(code: str) -> ParsedMove:
    """Validate a two-character move code or raise :class:`IllegalMove`."""
    text = code.strip().lower()
    if len(text) != 2:
        raise IllegalMove(f"move {code!r} is not two characters (e.g. '27', '3a', '5h')")
    src, dst = text[0], text[1]
    if src not in "12345678abcd":
        raise IllegalMove(f"move {code!r}: source must be a cascade 1-8 or free cell a-d")
    if dst not in "12345678abcdfh":
        raise IllegalMove(
            f"move {code!r}: destination must be a cascade 1-8, free cell a-d/f, or 'h'"
        )
    if src == dst:
        raise IllegalMove(f"move {code!r}: source and destination are the same")
    return ParsedMove(src=src, dst=dst)


def _lift_source_card(state: GameState, src: str) -> Card:
    """The card a single-card move takes from ``src``, without removing it."""
    if src in CELL_NAMES:
        card = state.free[src]
        if card is None:
            raise IllegalMove(f"free cell {src} is empty")
        return card
    column = state.cascades[int(src) - 1]
    if not column:
        raise IllegalMove(f"cascade {src} is empty")
    return column[-1]


def _remove_source_card(free: dict[str, Card | None], cascades: list[list[Card]], src: str) -> None:
    if src in CELL_NAMES:
        free[src] = None
    else:
        cascades[int(src) - 1].pop()


def _movable_run(column: list[Card]) -> list[Card]:
    """Longest properly-ordered (descending, alternating-colour) suffix of ``column``."""
    run = [column[-1]]
    for card in reversed(column[:-1]):
        if _fits_on_cascade(run[0], card):
            run.insert(0, card)
        else:
            break
    return run


def supermove_capacity(state: GameState, *, dst_is_empty_cascade: bool) -> int:
    """Max cards movable as a unit: ``(1 + empty cells) * 2^(empty cascades)``.

    An empty destination cascade does not count toward its own multiplier.
    """
    empty_cascades = sum(1 for column in state.cascades if not column)
    if dst_is_empty_cascade:
        empty_cascades -= 1
    return (1 + len(state.empty_cells())) * 2**empty_cascades


def _cascade_segment(state: GameState, src_index: int, dst_index: int) -> list[Card]:
    """The cards a cascade-to-cascade move transfers, or raise :class:`IllegalMove`."""
    src_column = state.cascades[src_index]
    if not src_column:
        raise IllegalMove(f"cascade {src_index + 1} is empty")
    run = _movable_run(src_column)
    dst_column = state.cascades[dst_index]
    capacity = supermove_capacity(state, dst_is_empty_cascade=not dst_column)
    if not dst_column:
        segment = run[-min(len(run), capacity) :]
        if not segment:
            raise IllegalMove("no capacity to move to the empty cascade")
        return segment
    top = dst_column[-1]
    for start, card in enumerate(run):
        if _fits_on_cascade(card, top):
            segment = run[start:]
            if len(segment) > capacity:
                raise IllegalMove(
                    f"moving {len(segment)} cards needs capacity {len(segment)}, "
                    f"only {capacity} available (free cells/empty cascades)"
                )
            return segment
    raise IllegalMove(
        f"no card in the run {' '.join(run)} fits on {top} "
        f"(needs rank {rank_of(top) - 1}, opposite colour)"
    )


def _apply_to_foundation(state: GameState, move: ParsedMove) -> tuple[GameState, str]:
    card = _lift_source_card(state, move.src)
    suit = suit_of(card)
    if state.foundations[suit] != rank_of(card) - 1:
        raise IllegalMove(
            f"{card} cannot go home: {suit} foundation is at "
            f"{RANKS[state.foundations[suit] - 1] if state.foundations[suit] else 'empty'}"
        )
    free = dict(state.free)
    cascades = [list(column) for column in state.cascades]
    _remove_source_card(free, cascades, move.src)
    foundations = dict(state.foundations)
    foundations[suit] = rank_of(card)
    new = GameState(free=free, foundations=foundations, cascades=cascades)
    return new, f"{card} to foundation"


def _apply_to_free_cell(state: GameState, move: ParsedMove) -> tuple[GameState, str]:
    card = _lift_source_card(state, move.src)
    cell = move.dst
    if cell == "f":
        empty = state.empty_cells()
        if not empty:
            raise IllegalMove("all free cells are occupied")
        cell = empty[0]
    elif state.free[cell] is not None:
        raise IllegalMove(f"free cell {cell} already holds {state.free[cell]}")
    free = dict(state.free)
    cascades = [list(column) for column in state.cascades]
    _remove_source_card(free, cascades, move.src)
    free[cell] = card
    new = GameState(free=free, foundations=dict(state.foundations), cascades=cascades)
    return new, f"{card} to free cell {cell}"


def _apply_to_cascade(state: GameState, move: ParsedMove) -> tuple[GameState, str]:
    dst_index = int(move.dst) - 1
    if move.src in CELL_NAMES:
        card = _lift_source_card(state, move.src)
        dst_column = state.cascades[dst_index]
        if dst_column and not _fits_on_cascade(card, dst_column[-1]):
            raise IllegalMove(f"{card} does not fit on {dst_column[-1]}")
        segment = [card]
    else:
        segment = _cascade_segment(state, int(move.src) - 1, dst_index)
    free = dict(state.free)
    cascades = [list(column) for column in state.cascades]
    if move.src in CELL_NAMES:
        free[move.src] = None
    else:
        src_column = cascades[int(move.src) - 1]
        del src_column[len(src_column) - len(segment) :]
    cascades[dst_index].extend(segment)
    new = GameState(free=free, foundations=dict(state.foundations), cascades=cascades)
    return new, f"{' '.join(segment)} to cascade {move.dst}"


def apply_move(state: GameState, code: str) -> tuple[GameState, str]:
    """Apply one move code, returning the new state and a short description.

    Raises :class:`IllegalMove` (with the reason) and leaves ``state`` untouched.
    """
    move = parse_move(code)
    if move.dst == "h":
        return _apply_to_foundation(state, move)
    if move.dst in CELL_NAMES or move.dst == "f":
        return _apply_to_free_cell(state, move)
    return _apply_to_cascade(state, move)


def apply_sequence(state: GameState, codes: list[str]) -> tuple[GameState, list[str], str | None]:
    """Apply moves in order until one fails.

    Returns the state after the legal prefix, the descriptions of the applied moves, and
    the error message of the first illegal move (``None`` if all applied).
    """
    applied: list[str] = []
    for code in codes:
        try:
            state, description = apply_move(state, code)
        except IllegalMove as exc:
            return state, applied, f"move {code!r} rejected: {exc}"
        applied.append(f"{code}: {description}")
    return state, applied, None
