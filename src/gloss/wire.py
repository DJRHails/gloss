"""Wire types shared across the pipeline: transcripts, monitor items, answers, runs.

Everything that crosses a file boundary (rollout -> items -> monitor -> scoring) is a
pydantic model here, so each stage validates its input at the boundary instead of
trusting the previous stage's writer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from gloss.freecell import GameState

FeedbackMode = Literal["ack", "board"]
Condition = Literal["with-cot", "no-cot"]


class ToolCallRecord(BaseModel):
    """The agent's ``play`` call: the raw argument and the parsed move codes."""

    moves_raw: str
    codes: list[str]


class TurnRecord(BaseModel):
    """One agent turn with its ground-truth surroundings.

    ``state_before`` is the engine state when the agent started thinking — the exact
    object the monitor is later asked to reconstruct. ``moves_applied`` is the legal
    prefix of the tool call that the environment actually executed.
    """

    turn_index: int
    state_before: GameState
    thinking: str
    assistant_text: str
    tool_call: ToolCallRecord | None
    tool_result: str | None
    moves_applied: list[str]
    state_after: GameState
    stop_reason: str


class Transcript(BaseModel):
    """One full game: the agent's turns plus the run configuration that produced them."""

    transcript_id: str
    game_num: int
    agent_model: str
    feedback_mode: FeedbackMode
    thinking_budget: int
    turns: list[TurnRecord]
    won: bool


class MonitorItem(BaseModel):
    """One monitor question: prior-turn history (CoT stripped) plus this turn's CoT.

    ``context`` is the rendered history the monitor sees under both conditions;
    ``cot`` is included only under ``with-cot``. The ``truth_*`` fields never reach
    the monitor — they are the scoring key.
    """

    item_id: str
    transcript_id: str
    game_num: int
    turn_index: int
    context: str
    cot: str
    truth_state: GameState
    truth_next_codes: list[str]


class ProposedLine(BaseModel):
    """A candidate move sequence the monitor believes the CoT considered."""

    moves: list[str]
    summary: str = ""


class MonitorAnswer(BaseModel):
    """The monitor's structured reconstruction, exactly as submitted via the tool."""

    free_cells: list[str] = Field(default_factory=list)
    foundations: dict[str, str] = Field(default_factory=dict)
    cascades: list[list[str]] = Field(default_factory=list)
    proposed_lines: list[ProposedLine] = Field(default_factory=list)
    predicted_next_moves: list[str] = Field(default_factory=list)


class MonitorRun(BaseModel):
    """One monitor call on one item under one condition, successful or not."""

    item_id: str
    monitor_model: str
    condition: Condition
    answer: MonitorAnswer | None
    error: str | None = None
