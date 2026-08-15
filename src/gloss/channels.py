"""Channel audit over recorded transcripts: which pipe carried the player's reasoning.

The scratchpad arms hand the player a `scratchpad` tool *and* leave native thinking on, so a
turn can put its reasoning in either channel, both, or neither. Two censuses answer that from
the transcript alone, no model in the loop:

- :func:`channel_census` — per arm, how many turns used the pad, native thinking, both, or
  neither. "Both" is the number that decides whether a scratchpad *reveals* reasoning that was
  happening anyway or merely *relocates* it.
- :func:`response_census` — per arm, the content-block shape of every API response
  (``TurnRecord.response_signatures``), including how often one response carried two tool_use
  blocks. The drain in :mod:`gloss.rollout` handles that case; this is how we know at what rate
  it actually occurs rather than assuming it never does.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from gloss.utils.stats import rate_with_ci
from gloss.wire import Transcript, TurnRecord

PAD_TOOL_BLOCK = "tool_use:scratchpad"
PLAY_TOOL_BLOCK = "tool_use:play"
# A channel that carries fewer characters than this is doing handoff, not reasoning: the
# directed arm's turns that "use both channels" include ones whose native thinking is a
# single line ("Let me play those moves") against a 32,000-character pad. Counting those as
# both-channel turns would answer the relocation question yes on a technicality, so the census
# reports the strict count (any non-empty channel, comparable to PR #1's tally) and this
# substantive one side by side.
SUBSTANTIVE_CHARS = 1_000


class ChannelSplit(BaseModel):
    """Turns partitioned by which channels cleared one character threshold."""

    threshold: int  # a channel counts as used at this many characters or more
    both: int
    pad_only: int
    native_only: int
    neither: int


class ChannelCensus(BaseModel):
    """Per-arm turn counts by which reasoning channel carried content.

    Two splits over the same turns: ``any_content`` counts a channel as used at one character
    (PR #1's tally), ``substantive`` at :data:`SUBSTANTIVE_CHARS`. The pair is the whole point —
    a turn can look like it used both channels and still have put all of its reasoning in one.
    """

    arm: str
    offers_scratchpad: bool  # a native arm has no pad channel, so "both" is not an estimate there
    num_turns: int
    any_content: ChannelSplit
    substantive: ChannelSplit
    truncated: int
    mean_pad_chars: float
    mean_native_chars: float


class ResponseCensus(BaseModel):
    """Per-arm census of API response shapes, keyed by content-block signature."""

    arm: str
    num_responses: int
    multi_tool: int  # responses carrying two or more tool_use blocks
    pad_and_play: int  # responses carrying both a scratchpad and a play call
    signatures: dict[str, int]


def arm_label(transcript: Transcript) -> str:
    """``'claude-opus-5 / scratchpad-directed'`` — the (player, CoT source) pair being censused."""
    return f"{transcript.agent_model} / {transcript.cot_source}"


def _channel_chars(transcript: Transcript, turn: TurnRecord) -> tuple[int, int]:
    """``(pad chars, native-thinking chars)`` for one turn, whichever arm recorded it.

    The CoT column (``thinking``) is the pad on a scratchpad arm and the API's thinking blocks on
    a native arm; ``native_thinking`` always holds the API blocks but is empty on transcripts
    written before that field existed, so a native arm reads its count off the CoT column. A
    native arm's pad count is 0 by construction — the tool was never offered.
    """
    if transcript.cot_source == "native":
        return 0, len(turn.thinking.strip())
    return len(turn.thinking.strip()), len(turn.native_thinking.strip())


def _group_by_arm(transcripts: list[Transcript]) -> dict[str, list[Transcript]]:
    groups: dict[str, list[Transcript]] = {}
    for transcript in transcripts:
        groups.setdefault(arm_label(transcript), []).append(transcript)
    return dict(sorted(groups.items()))


def _split(chars: list[tuple[int, int]], *, threshold: int) -> ChannelSplit:
    """Partition ``(pad, native)`` character pairs by which channels reach ``threshold``."""
    used = [(pad >= threshold, native >= threshold) for pad, native in chars]
    return ChannelSplit(
        threshold=threshold,
        both=sum(1 for pad, native in used if pad and native),
        pad_only=sum(1 for pad, native in used if pad and not native),
        native_only=sum(1 for pad, native in used if native and not pad),
        neither=sum(1 for pad, native in used if not pad and not native),
    )


def channel_census(transcripts: list[Transcript]) -> list[ChannelCensus]:
    """One :class:`ChannelCensus` per arm, over every recorded turn.

    A truncated turn is counted in the four-way split by what it actually contains (usually
    native thinking with an empty pad, since a tool argument cut off by ``max_tokens`` arrives
    as ``{}``) and also reported in its own column, rather than being folded into "neither".
    """
    censuses: list[ChannelCensus] = []
    for arm, group in _group_by_arm(transcripts).items():
        turns = [(transcript, turn) for transcript in group for turn in transcript.turns]
        chars = [_channel_chars(transcript, turn) for transcript, turn in turns]
        pads = [pad for pad, _native in chars]
        natives = [native for _pad, native in chars]
        count = len(turns) or 1  # means over an empty arm are 0.0, not a ZeroDivisionError
        censuses.append(
            ChannelCensus(
                arm=arm,
                offers_scratchpad=any(t.cot_source != "native" for t in group),
                num_turns=len(turns),
                any_content=_split(chars, threshold=1),
                substantive=_split(chars, threshold=SUBSTANTIVE_CHARS),
                truncated=sum(1 for _transcript, turn in turns if turn.truncated),
                mean_pad_chars=sum(pads) / count,
                mean_native_chars=sum(natives) / count,
            )
        )
    return censuses


def response_census(transcripts: list[Transcript]) -> list[ResponseCensus]:
    """One :class:`ResponseCensus` per arm, over every API response recorded in a turn."""
    censuses: list[ResponseCensus] = []
    for arm, group in _group_by_arm(transcripts).items():
        signatures = [
            signature
            for transcript in group
            for turn in transcript.turns
            for signature in turn.response_signatures
        ]
        censuses.append(
            ResponseCensus(
                arm=arm,
                num_responses=len(signatures),
                multi_tool=sum(1 for signature in signatures if signature.count("tool_use:") > 1),
                pad_and_play=sum(
                    1
                    for signature in signatures
                    if PAD_TOOL_BLOCK in signature and PLAY_TOOL_BLOCK in signature
                ),
                signatures=dict(Counter(signatures).most_common()),
            )
        )
    return censuses


def channel_table(censuses: list[ChannelCensus]) -> str:
    """Markdown table of the four-way channel split, with a Wilson interval on "both"."""
    header = (
        "| arm | turns | a channel counts as used at | both channels | pad only | native only "
        "| neither | truncated | mean pad chars | mean native chars |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    rows: list[str] = []
    for census in censuses:
        for split in (census.any_content, census.substantive):
            both = (
                rate_with_ci(split.both, census.num_turns)
                if census.offers_scratchpad
                else "n/a (no pad offered)"
            )
            label = "1 char" if split.threshold == 1 else f"{split.threshold:,} chars"
            rows.append(
                f"| {census.arm} | {census.num_turns} | {label} | {both} "
                f"| {split.pad_only} | {split.native_only} | {split.neither} | {census.truncated} "
                f"| {census.mean_pad_chars:,.0f} | {census.mean_native_chars:,.0f} |"
            )
    return "\n".join([header, *rows])


def response_table(censuses: list[ResponseCensus]) -> str:
    """Markdown table of response block shapes per arm, most common first."""
    header = "| arm | responses | 2+ tool_use | scratchpad+play | block signature | count |\n"
    header += "|---|---|---|---|---|---|"
    rows: list[str] = []
    for census in censuses:
        for index, (signature, count) in enumerate(census.signatures.items()):
            lead = (
                f"| {census.arm} | {census.num_responses} "
                f"| {rate_with_ci(census.multi_tool, census.num_responses)} "
                f"| {rate_with_ci(census.pad_and_play, census.num_responses)} "
                if index == 0
                else "|  |  |  |  "
            )
            rows.append(f"{lead}| `{signature}` | {count} |")
    return "\n".join([header, *rows])
