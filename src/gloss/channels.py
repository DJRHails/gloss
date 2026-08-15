"""Does a scratchpad *relocate* reasoning, or reveal reasoning that was happening anyway?

The first scratchpad run found zero of 19 Opus 5 turns using both channels: where the player wrote
a pad it did no native thinking, and where it thought natively the pad was empty. Read as
relocation — same work, different pipe — that is a stronger claim than either side of the original
"is elicited reasoning real" argument, and it is also exactly what a forcefully-worded pad
instruction would produce on its own. This module measures the split so the two readings separate.

The headline is **co-use**: P(native thinking present | the player wrote a pad). Conditioning on a
pad matters — an arm that simply never uses the pad has no co-use to measure, and reporting an
unconditional "0% used both" there would look like relocation while meaning "the tool went unused".
So pad-use rate is reported alongside it as the denominator, never folded into one number.

The ``native`` arm is the null control: the pad tool is never offered, so pad-use must be 0.0. If it
is not, the pad detector is firing on something else and every number here is meaningless.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict

from gloss.utils.stats import wilson_ci
from gloss.wire import Transcript, TurnRecord

ChannelUse = Literal["both", "pad_only", "native_only", "neither"]


class Rate(BaseModel):
    """A binomial rate with its 95% Wilson interval and the denominator it rests on."""

    model_config = ConfigDict(frozen=True)

    point: float
    low: float
    high: float
    successes: int
    n: int

    @classmethod
    def of(cls, successes: int, total: int) -> Rate:
        point, low, high = wilson_ci(successes, total)
        return cls(point=point, low=low, high=high, successes=successes, n=total)

    def render(self) -> str:
        if not self.n:
            return "no data (n=0)"
        return f"{self.point:.2f} [{self.low:.2f}, {self.high:.2f}] {self.successes}/{self.n}"


class ChannelSplit(BaseModel):
    """One arm's channel usage over its scorable turns."""

    model_config = ConfigDict(frozen=True)

    cot_source: str
    n_turns: int
    n_truncated: int
    # Turns that used at least one channel. The meaningful denominator for the *use* rates, because
    # a capped-move run is dominated by turns that reason in neither channel: under a per-call cap
    # the player plans once on turn 0 (an 18k-43k character pad) then emits bare 3-move calls at ~52
    # output tokens for every remaining turn. Those turns are real but carry nothing to measure, and
    # dividing by them makes pad use look like 0.06 when it is 1.00 of the turns that reasoned.
    n_reasoning_turns: int
    mix: dict[str, int]  # ChannelUse -> count; a plain tally, not a validated boundary
    pad_use: Rate
    # Pad use over reasoning-bearing turns only — the arms' comparable figure.
    pad_use_given_reasoning: Rate
    # P(native thinking present | pad present) — the relocation test. Relocation predicts ~0.
    co_use_given_pad: Rate
    native_use: Rate
    median_pad_chars: int
    median_native_chars: int
    median_native_thinking_tokens: int | None


def classify_turn(turn: TurnRecord, *, pad_is_cot: bool) -> ChannelUse:
    """Which channels this turn used.

    ``pad_is_cot`` says whether this arm sourced its CoT column from the pad. In a scratchpad arm
    ``thinking`` holds the pad and ``native_thinking`` the API blocks; in the native arm both hold
    the API blocks, so the pad is empty by construction and must be reported as such.
    """
    pad = turn.thinking if pad_is_cot else ""
    native = turn.native_thinking
    if pad and native:
        return "both"
    if pad:
        return "pad_only"
    if native:
        return "native_only"
    return "neither"


def _median(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def split_for(transcripts: list[Transcript], cot_source: str) -> ChannelSplit:
    """Channel usage for every scorable turn of one arm.

    Truncated turns are excluded and counted separately: a pad clipped by the output cap arrives as
    ``{}``, so scoring it would record "wrote no pad" for what is really a harness limit — the bug
    that invalidated the first scratchpad run.
    """
    arm = [t for t in transcripts if (t.cot_source or "native") == cot_source]
    all_turns = [turn for transcript in arm for turn in transcript.turns]
    turns = [turn for turn in all_turns if not turn.truncated]
    pad_is_cot = cot_source != "native"
    uses = [classify_turn(turn, pad_is_cot=pad_is_cot) for turn in turns]
    tally = Counter(uses)
    with_pad = [turn for turn, use in zip(turns, uses, strict=True) if use in ("both", "pad_only")]
    reported_tokens = [
        turn.native_thinking_tokens for turn in turns if turn.native_thinking_tokens is not None
    ]
    reasoning_turns = len(turns) - tally["neither"]
    return ChannelSplit(
        cot_source=cot_source,
        n_turns=len(turns),
        n_truncated=len(all_turns) - len(turns),
        n_reasoning_turns=reasoning_turns,
        mix={use: tally.get(use, 0) for use in ("both", "pad_only", "native_only", "neither")},
        pad_use=Rate.of(tally["both"] + tally["pad_only"], len(turns)),
        pad_use_given_reasoning=Rate.of(tally["both"] + tally["pad_only"], reasoning_turns),
        co_use_given_pad=Rate.of(tally["both"], len(with_pad)),
        native_use=Rate.of(tally["both"] + tally["native_only"], len(turns)),
        median_pad_chars=_median(
            [len(turn.thinking) for turn in turns] if pad_is_cot else [0] * len(turns)
        ),
        median_native_chars=_median([len(turn.native_thinking) for turn in turns]),
        median_native_thinking_tokens=_median(reported_tokens) if reported_tokens else None,
    )


def render_table(splits: list[ChannelSplit]) -> str:
    """A text table of the arms, headline metric last so it reads as the conclusion."""
    lines = [
        f"{'arm':20s} {'turns':>5s} {'reas':>5s} {'trunc':>5s}  "
        f"{'pad use | reasoning':>22s}  {'native use':>22s}  {'co-use | pad':>22s}",
    ]
    for split in splits:
        lines.append(
            f"{split.cot_source:20s} {split.n_turns:5d} {split.n_reasoning_turns:5d} "
            f"{split.n_truncated:5d}  "
            f"{split.pad_use_given_reasoning.render():>22s}  {split.native_use.render():>22s}  "
            f"{split.co_use_given_pad.render():>22s}"
        )
    lines.append("")
    lines.append("channel mix (both / pad-only / native-only / neither), and median sizes:")
    for split in splits:
        mix = split.mix
        tokens = (
            "unreported"
            if split.median_native_thinking_tokens is None
            else str(split.median_native_thinking_tokens)
        )
        lines.append(
            f"  {split.cot_source:20s} {mix['both']:3d} / {mix['pad_only']:3d} / "
            f"{mix['native_only']:3d} / {mix['neither']:3d}   "
            f"pad~{split.median_pad_chars} chars, native~{split.median_native_chars} chars, "
            f"native thinking tokens~{tokens}"
        )
    return "\n".join(lines)
