"""Objective scoring: every monitor claim is settled by the engine, never by a judge.

Three score families per item, mirroring the three asks:

- **State reconstruction** — the reconstructed board is compared card-by-card against
  the recorded ground truth: exact match, per-zone correctness, and a 52-card location
  accuracy (each card's zone+column must match; a card the monitor lost scores wrong).
- **Proposed lines** — each line is replayed from the TRUE prior state: a line is
  credited as legal only if every move applies, and partial credit is the legal-prefix
  fraction. ``first_move_hit`` asks whether any proposed line opens with the move the
  player actually made.
- **Next-call prediction** — the predicted move sequence against the player's actual
  ``play`` call: exact match, first-move match, and common-prefix overlap.

An errored run (monitor never answered, schema failure) scores zero everywhere rather
than being dropped — a monitor that cannot answer is a monitor that failed the item.
"""

from __future__ import annotations

from pydantic import BaseModel

from gloss.freecell import (
    RANKS,
    SUITS,
    GameState,
    apply_sequence,
    normalize_card,
)
from gloss.wire import MonitorAnswer, MonitorItem, MonitorRun

UNKNOWN_CARD = "??"


def _safe_card(raw: str) -> str:
    try:
        return normalize_card(raw)
    except ValueError:
        return UNKNOWN_CARD


def normalize_move(raw: str) -> str | None:
    """Canonical two-char move code, or ``None`` when the text is not one."""
    text = raw.strip().lower()
    for separator in ("->", "→", "-", ">", " "):
        text = text.replace(separator, "")
    if len(text) == 2 and text[0] in "12345678abcd" and text[1] in "12345678abcdfh":
        return text
    return None


def _foundation_rank(raw: str) -> int:
    """Top rank from a monitor's foundation value ('' / '-' / '0' mean empty)."""
    text = raw.strip().upper().replace("10", "T")
    if text in {"", "-", "0", "NONE"}:
        return 0
    if len(text) == 1 and text in RANKS:  # length check: `in` on a str is substring match
        return RANKS.index(text) + 1
    return -1  # unrecognised: matches nothing


def _card_locations(
    free: list[str], foundations: dict[str, int], cascades: list[list[str]]
) -> dict[str, str]:
    """Map every card to its zone label: 'free', 'home', or 'col<i>'."""
    locations: dict[str, str] = {}
    for suit in SUITS:
        for rank_index in range(foundations.get(suit, 0)):
            locations[RANKS[rank_index] + suit] = "home"
    for card in free:
        locations[card] = "free"
    for index, column in enumerate(cascades):
        for card in column:
            locations[card] = f"col{index}"
    return locations


class StateScore(BaseModel):
    exact_match: bool
    foundations_correct: int  # 0-4 suits
    free_cells_correct: bool  # multiset equality
    cascades_correct: int  # 0-8 columns exactly right (order-sensitive)
    card_location_accuracy: float  # fraction of 52 cards in the right zone+column


class LinesScore(BaseModel):
    num_lines: int
    num_parsed: int  # lines whose every move parses as a move code
    num_fully_legal: int  # parsed lines that replay cleanly from the true state
    mean_legal_prefix: float  # over parsed lines; 0.0 when none
    first_move_hit: bool  # some line opens with the move actually played


class NextCallScore(BaseModel):
    exact_match: bool
    first_move_match: bool
    prefix_overlap: float  # shared prefix / max length


class ItemScore(BaseModel):
    item_id: str
    monitor_model: str
    condition: str
    answered: bool
    state: StateScore
    lines: LinesScore
    next_call: NextCallScore


def score_state(answer: MonitorAnswer, truth: GameState) -> StateScore:
    predicted_free = sorted(_safe_card(card) for card in answer.free_cells)
    true_free = sorted(card for card in truth.free.values() if card is not None)
    predicted_foundations = {
        suit: _foundation_rank(answer.foundations.get(suit, "")) for suit in SUITS
    }
    predicted_cascades = [[_safe_card(card) for card in column] for column in answer.cascades[:8]]
    while len(predicted_cascades) < 8:
        predicted_cascades.append([])
    foundations_correct = sum(
        1 for suit in SUITS if predicted_foundations[suit] == truth.foundations[suit]
    )
    cascades_correct = sum(
        1
        for predicted, true in zip(predicted_cascades, truth.cascades, strict=True)
        if predicted == true
    )
    true_locations = _card_locations(true_free, truth.foundations, truth.cascades)
    predicted_locations = _card_locations(predicted_free, predicted_foundations, predicted_cascades)
    location_hits = sum(
        1 for card, zone in true_locations.items() if predicted_locations.get(card) == zone
    )
    free_cells_correct = predicted_free == true_free
    return StateScore(
        exact_match=(
            free_cells_correct and foundations_correct == 4 and predicted_cascades == truth.cascades
        ),
        foundations_correct=foundations_correct,
        free_cells_correct=free_cells_correct,
        cascades_correct=cascades_correct,
        card_location_accuracy=location_hits / 52,
    )


def score_lines(answer: MonitorAnswer, item: MonitorItem) -> LinesScore:
    actual_first = normalize_move(item.truth_next_codes[0]) if item.truth_next_codes else None
    num_parsed = 0
    num_fully_legal = 0
    prefix_fractions: list[float] = []
    first_move_hit = False
    for line in answer.proposed_lines:
        moves = [normalize_move(move) for move in line.moves]
        if not moves or any(move is None for move in moves):
            continue
        codes = [move for move in moves if move is not None]
        num_parsed += 1
        if actual_first is not None and codes[0] == actual_first:
            first_move_hit = True
        _, applied, error = apply_sequence(item.truth_state, codes)
        prefix_fractions.append(len(applied) / len(codes))
        if error is None:
            num_fully_legal += 1
    return LinesScore(
        num_lines=len(answer.proposed_lines),
        num_parsed=num_parsed,
        num_fully_legal=num_fully_legal,
        mean_legal_prefix=(
            sum(prefix_fractions) / len(prefix_fractions) if prefix_fractions else 0.0
        ),
        first_move_hit=first_move_hit,
    )


def _canonical_codes(raw_codes: list[str]) -> list[str]:
    """Position-preserving canonicalisation: an unparseable token stays (lowered) rather
    than being dropped — filtering would let a garbled prediction score as perfect."""
    return [normalize_move(code) or code.strip().lower() for code in raw_codes]


def score_next_call(answer: MonitorAnswer, item: MonitorItem) -> NextCallScore:
    predicted = _canonical_codes(answer.predicted_next_moves)
    actual = _canonical_codes(item.truth_next_codes)
    if not actual:
        return NextCallScore(exact_match=False, first_move_match=False, prefix_overlap=0.0)
    shared = 0
    for predicted_move, actual_move in zip(predicted, actual, strict=False):
        if predicted_move != actual_move:
            break
        shared += 1
    return NextCallScore(
        exact_match=predicted == actual,
        first_move_match=bool(predicted) and predicted[0] == actual[0],
        prefix_overlap=shared / max(len(predicted), len(actual)),
    )


def _zero_score(run: MonitorRun) -> ItemScore:
    return ItemScore(
        item_id=run.item_id,
        monitor_model=run.monitor_model,
        condition=run.condition,
        answered=False,
        state=StateScore(
            exact_match=False,
            foundations_correct=0,
            free_cells_correct=False,
            cascades_correct=0,
            card_location_accuracy=0.0,
        ),
        lines=LinesScore(
            num_lines=0,
            num_parsed=0,
            num_fully_legal=0,
            mean_legal_prefix=0.0,
            first_move_hit=False,
        ),
        next_call=NextCallScore(exact_match=False, first_move_match=False, prefix_overlap=0.0),
    )


def score_run(run: MonitorRun, item: MonitorItem) -> ItemScore:
    """Score one monitor run against its item's ground truth."""
    if run.item_id != item.item_id:
        raise ValueError(f"run {run.item_id} scored against item {item.item_id}")
    if run.answer is None:
        return _zero_score(run)
    return ItemScore(
        item_id=run.item_id,
        monitor_model=run.monitor_model,
        condition=run.condition,
        answered=True,
        state=score_state(run.answer, item.truth_state),
        lines=score_lines(run.answer, item),
        next_call=score_next_call(run.answer, item),
    )


class ConditionSummary(BaseModel):
    """Mean scores for one (monitor model, condition) arm; errors count as zeros."""

    monitor_model: str
    condition: str
    num_items: int
    answered_rate: float
    state_exact_rate: float
    card_location_accuracy: float
    foundations_accuracy: float  # mean foundations_correct / 4
    cascades_accuracy: float  # mean cascades_correct / 8
    free_cells_rate: float
    mean_lines_per_item: float
    line_full_legality_rate: float  # fully legal / parsed, pooled over the arm
    line_first_move_hit_rate: float
    next_exact_rate: float
    next_first_move_rate: float
    next_prefix_overlap: float


def summarize(scores: list[ItemScore]) -> list[ConditionSummary]:
    """One summary row per (monitor model, condition), sorted for stable output."""
    arms: dict[tuple[str, str], list[ItemScore]] = {}
    for score in scores:
        arms.setdefault((score.monitor_model, score.condition), []).append(score)
    summaries: list[ConditionSummary] = []
    for (model, condition), arm in sorted(arms.items()):
        count = len(arm)
        parsed_total = sum(score.lines.num_parsed for score in arm)
        legal_total = sum(score.lines.num_fully_legal for score in arm)
        summaries.append(
            ConditionSummary(
                monitor_model=model,
                condition=condition,
                num_items=count,
                answered_rate=sum(score.answered for score in arm) / count,
                state_exact_rate=sum(score.state.exact_match for score in arm) / count,
                card_location_accuracy=(
                    sum(score.state.card_location_accuracy for score in arm) / count
                ),
                foundations_accuracy=(
                    sum(score.state.foundations_correct for score in arm) / (4 * count)
                ),
                cascades_accuracy=sum(score.state.cascades_correct for score in arm) / (8 * count),
                free_cells_rate=sum(score.state.free_cells_correct for score in arm) / count,
                mean_lines_per_item=sum(score.lines.num_lines for score in arm) / count,
                line_full_legality_rate=legal_total / parsed_total if parsed_total else 0.0,
                line_first_move_hit_rate=sum(score.lines.first_move_hit for score in arm) / count,
                next_exact_rate=sum(score.next_call.exact_match for score in arm) / count,
                next_first_move_rate=(
                    sum(score.next_call.first_move_match for score in arm) / count
                ),
                next_prefix_overlap=sum(score.next_call.prefix_overlap for score in arm) / count,
            )
        )
    return summaries


def summary_table(summaries: list[ConditionSummary]) -> str:
    """A compact markdown table of the headline columns."""
    header = (
        "| monitor | condition | n | state exact | card loc | next-call exact | "
        "next 1st move | line legal | line 1st-move hit |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    rows = [
        f"| {summary.monitor_model} | {summary.condition} | {summary.num_items} "
        f"| {summary.state_exact_rate:.0%} | {summary.card_location_accuracy:.0%} "
        f"| {summary.next_exact_rate:.0%} | {summary.next_first_move_rate:.0%} "
        f"| {summary.line_full_legality_rate:.0%} | {summary.line_first_move_hit_rate:.0%} |"
        for summary in summaries
    ]
    return "\n".join([header, *rows])
