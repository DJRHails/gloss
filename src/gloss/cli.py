"""The ``gloss`` CLI: deal -> rollout -> items -> monitor -> score.

Each stage reads and writes JSONL under ``data/`` by default, so a run is resumable at
any stage boundary and every artifact is inspectable.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from gloss.freecell import deal as ms_deal
from gloss.monitor import build_items, run_monitor
from gloss.rollout import run_rollout
from gloss.scoring import score_run, summarize, summary_table
from gloss.utils.jsonl import read_jsonl_rows, write_jsonl
from gloss.wire import Condition, MonitorItem, MonitorRun, Transcript

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

DEFAULT_AGENT = "claude-fable-5"
DEFAULT_MONITORS = "claude-sonnet-5,claude-haiku-4-5-20251001"
CONDITIONS: tuple[Condition, Condition] = ("with-cot", "no-cot")


@app.command()
def deal(game_num: Annotated[int, typer.Argument(help="Microsoft deal number")]) -> None:
    """Print the board for a Microsoft deal (sanity check / eyeballing)."""
    typer.echo(ms_deal(game_num).render())


@app.command()
def rollout(
    games: Annotated[str, typer.Option(help="Comma-separated Microsoft deal numbers")] = "1,617",
    agent_model: Annotated[str, typer.Option()] = DEFAULT_AGENT,
    max_turns: Annotated[int, typer.Option()] = 24,
    thinking_budget: Annotated[int, typer.Option()] = 8000,
    feedback: Annotated[str, typer.Option(help="'ack' (hard) or 'board' (easy)")] = "ack",
    out: Annotated[Path, typer.Option()] = Path("data/transcripts.jsonl"),
) -> None:
    """Play each deal with the agent model, recording per-turn ground truth + CoT."""
    if feedback not in ("ack", "board"):
        raise typer.BadParameter("feedback must be 'ack' or 'board'")
    game_nums = [int(part) for part in games.split(",")]
    transcripts: list[Transcript] = []
    with ThreadPoolExecutor(max_workers=min(len(game_nums), 4)) as pool:
        futures = [
            pool.submit(
                run_rollout,
                game_num=game_num,
                agent_model=agent_model,
                max_turns=max_turns,
                thinking_budget=thinking_budget,
                feedback=feedback,
            )
            for game_num in game_nums
        ]
        for future in as_completed(futures):
            transcript = future.result()
            transcripts.append(transcript)
            logger.info(
                f"game {transcript.game_num}: {len(transcript.turns)} turns, won={transcript.won}"
            )
            write_jsonl(transcripts, out, atomic=True)  # checkpoint after every game
    transcripts.sort(key=lambda transcript: transcript.game_num)
    write_jsonl(transcripts, out, atomic=True)
    typer.echo(f"{len(transcripts)} transcripts -> {out}")


@app.command()
def items(
    transcripts: Annotated[Path, typer.Option()] = Path("data/transcripts.jsonl"),
    out: Annotated[Path, typer.Option()] = Path("data/items.jsonl"),
) -> None:
    """Build monitor items (one per eligible turn) from recorded transcripts."""
    rows: list[MonitorItem] = []
    for transcript in read_jsonl_rows(transcripts, Transcript):
        rows.extend(build_items(transcript))
    write_jsonl(rows, out, atomic=True)
    typer.echo(f"{len(rows)} items -> {out}")


@app.command()
def monitor(
    items_path: Annotated[Path, typer.Option("--items")] = Path("data/items.jsonl"),
    monitor_models: Annotated[str, typer.Option(help="Comma-separated")] = DEFAULT_MONITORS,
    thinking_budget: Annotated[int, typer.Option()] = 4000,
    out: Annotated[Path, typer.Option()] = Path("data/runs.jsonl"),
) -> None:
    """Run each monitor on every item under both conditions (with-cot and no-cot)."""
    monitor_items = read_jsonl_rows(items_path, MonitorItem)
    calls = [
        (item, model.strip(), condition)
        for model in monitor_models.split(",")
        for condition in CONDITIONS
        for item in monitor_items
    ]
    runs: list[MonitorRun] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                run_monitor,
                item,
                monitor_model=model,
                condition=condition,
                thinking_budget=thinking_budget,
            ): (item.item_id, model, condition)
            for item, model, condition in calls
        }
        for future in as_completed(futures):
            runs.append(future.result())
            item_id, model, condition = futures[future]
            logger.info(f"{item_id} [{model} / {condition}] done ({len(runs)}/{len(calls)})")
            write_jsonl(runs, out, atomic=True)  # checkpoint after every call
    typer.echo(f"{len(runs)} monitor runs -> {out}")


@app.command()
def score(
    items_path: Annotated[Path, typer.Option("--items")] = Path("data/items.jsonl"),
    runs_path: Annotated[Path, typer.Option("--runs")] = Path("data/runs.jsonl"),
    out: Annotated[Path, typer.Option()] = Path("data/scores.json"),
) -> None:
    """Score every run against ground truth and print the per-arm summary table."""
    items_by_id = {item.item_id: item for item in read_jsonl_rows(items_path, MonitorItem)}
    scores = [
        score_run(run, items_by_id[run.item_id]) for run in read_jsonl_rows(runs_path, MonitorRun)
    ]
    summaries = summarize(scores)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "per_item": [item_score.model_dump() for item_score in scores],
                "summaries": [summary.model_dump() for summary in summaries],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    typer.echo(summary_table(summaries))
    typer.echo(f"\nfull scores -> {out}")


if __name__ == "__main__":
    app()
