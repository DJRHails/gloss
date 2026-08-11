"""Rollout harness: an agent model plays FreeCell via a ``play`` tool, thinking enabled.

Every turn records the ground-truth :class:`gloss.freecell.GameState` before the agent
moved — the object monitors are later scored on reconstructing. Feedback comes in two
modes: ``ack`` (the default) confirms each applied move but never re-renders the board,
so the CoT is the richest surviving record of the agent's state tracking; ``board``
appends a full render to every tool result, the easy mode where the log alone gives the
state away.
"""

from __future__ import annotations

from loguru import logger

from gloss.freecell import GameState, apply_sequence, deal
from gloss.models.anthropic import create_message, extract_blocks
from gloss.prompts import load_prompt, rules_block
from gloss.wire import FeedbackMode, ToolCallRecord, Transcript, TurnRecord

PLAY_TOOL = {
    "name": "play",
    "description": (
        "Submit a space-separated sequence of FreeCell move codes, applied in order "
        "until the first illegal move (the legal prefix stays applied)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "moves": {
                "type": "string",
                "description": "Space-separated move codes, e.g. '3h 27 1a'",
            }
        },
        "required": ["moves"],
    },
}

_FEEDBACK_CLAUSE = {
    "ack": ", but does NOT show the board again",
    "board": " and shows the updated board",
}
_NUDGE = "Please continue playing: submit your next moves with the `play` tool."
_MAX_NUDGES = 2


def _tool_result_text(
    state: GameState, applied: list[str], error: str | None, feedback: FeedbackMode
) -> str:
    lines = [f"applied {len(applied)} move(s):"] if applied else []
    lines.extend(f"  {line}" for line in applied)
    if error:
        lines.append(f"REJECTED: {error} (subsequent moves discarded)")
    if state.is_won():
        lines.append("GAME WON — all foundations complete.")
    if feedback == "board":
        lines.append("")
        lines.append(state.render())
    return "\n".join(lines) if lines else "no moves applied"


def _record_turn(  # noqa: PLR0913 — pure assembly of one wire row
    turn_index: int,
    state_before: GameState,
    blocks_thinking: str,
    blocks_text: str,
    tool_call: ToolCallRecord | None,
    tool_result: str | None,
    moves_applied: list[str],
    state_after: GameState,
    stop_reason: str,
) -> TurnRecord:
    return TurnRecord(
        turn_index=turn_index,
        state_before=state_before,
        thinking=blocks_thinking,
        assistant_text=blocks_text,
        tool_call=tool_call,
        tool_result=tool_result,
        moves_applied=moves_applied,
        state_after=state_after,
        stop_reason=stop_reason,
    )


def run_rollout(
    *,
    game_num: int,
    agent_model: str,
    max_turns: int = 24,
    thinking_budget: int = 8000,
    feedback: FeedbackMode = "ack",
) -> Transcript:
    """Play one game, returning the full transcript with per-turn ground truth."""
    state = deal(game_num)
    system = load_prompt(
        "agent_system", rules=rules_block(), feedback_clause=_FEEDBACK_CLAUSE[feedback]
    )
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": f"Here is the deal (game #{game_num}):\n\n{state.render()}\n\nPlay to win.",
        }
    ]
    turns: list[TurnRecord] = []
    nudges = 0
    while len(turns) < max_turns and not state.is_won():
        message = create_message(
            model=agent_model,
            system=system,
            messages=messages,
            tools=[PLAY_TOOL],
            max_tokens=thinking_budget + 4000,
            thinking={"type": "enabled", "budget_tokens": thinking_budget},
        )
        blocks = extract_blocks(message)
        messages.append({"role": "assistant", "content": blocks.raw_content})
        state_before = state
        if blocks.tool_name is None:
            turns.append(
                _record_turn(
                    len(turns),
                    state_before,
                    blocks.thinking,
                    blocks.text,
                    None,
                    None,
                    [],
                    state,
                    blocks.stop_reason,
                )
            )
            if nudges >= _MAX_NUDGES:
                logger.warning(f"game {game_num}: agent stopped calling the tool; ending rollout")
                break
            nudges += 1
            messages.append({"role": "user", "content": _NUDGE})
            continue
        moves_raw = str(blocks.tool_input.get("moves", ""))
        codes = moves_raw.split()
        state, applied, error = apply_sequence(state, codes)
        result_text = _tool_result_text(state, applied, error, feedback)
        tool_use_id = next(
            str(block["id"]) for block in blocks.raw_content if block["type"] == "tool_use"
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tool_use_id, "content": result_text}
                ],
            }
        )
        turns.append(
            _record_turn(
                len(turns),
                state_before,
                blocks.thinking,
                blocks.text,
                ToolCallRecord(moves_raw=moves_raw, codes=codes),
                result_text,
                [line.split(":")[0] for line in applied],
                state,
                blocks.stop_reason,
            )
        )
        logger.info(
            f"game {game_num} turn {len(turns) - 1}: {len(codes)} moves submitted, "
            f"{len(applied)} applied{' (rejection)' if error else ''}"
        )
    return Transcript(
        transcript_id=f"game{game_num}-{agent_model}",
        game_num=game_num,
        agent_model=agent_model,
        feedback_mode=feedback,
        thinking_budget=thinking_budget,
        turns=turns,
        won=state.is_won(),
    )
