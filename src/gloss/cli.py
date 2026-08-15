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

from gloss.channels import render_table, split_for
from gloss.freecell import deal as ms_deal
from gloss.monitor import build_items, run_monitor, swapped_cot_donors
from gloss.rollout import run_rollout
from gloss.scoring import score_run, summarize, summary_table
from gloss.utils.jsonl import read_jsonl_rows, write_jsonl
from gloss.wire import SCRATCHPAD_SOURCES, Condition, MonitorItem, MonitorRun, Transcript

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

DEFAULT_AGENT = "claude-fable-5"
DEFAULT_MONITORS = "claude-sonnet-5,claude-haiku-4-5-20251001"
CONDITIONS: tuple[Condition, ...] = ("with-cot", "no-cot", "swapped-cot")


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
    effort: Annotated[str, typer.Option(help="Adaptive-model effort level")] = "medium",
    max_output_tokens: Annotated[
        int, typer.Option(help="Output cap; a scratchpad arm needs room for thinking AND a pad")
    ] = 128000,
    max_moves_per_call: Annotated[
        int,
        typer.Option(
            help="Cap moves per play call (0 = uncapped). A cap raises turns, so items, per deal"
        ),
    ] = 0,
    cot_source: Annotated[
        str,
        typer.Option(help=f"'native' thinking blocks, or one of {', '.join(SCRATCHPAD_SOURCES)}"),
    ] = "native",
    out: Annotated[Path, typer.Option()] = Path("data/transcripts.jsonl"),
) -> None:
    """Play each deal with the agent model, recording per-turn ground truth + CoT."""
    if feedback not in ("ack", "board"):
        raise typer.BadParameter("feedback must be 'ack' or 'board'")
    if cot_source not in ("native", *SCRATCHPAD_SOURCES):
        raise typer.BadParameter(f"cot-source must be 'native' or one of {SCRATCHPAD_SOURCES}")
    if max_moves_per_call < 0:
        raise typer.BadParameter("max-moves-per-call must be >= 0 (0 means uncapped)")
    game_nums = [int(part) for part in games.split(",")]
    # Append: the two cot_source arms are separate rollouts that must land in one dataset.
    existing = read_jsonl_rows(out, Transcript) if out.exists() else []
    keep = {t.transcript_id for t in existing}
    transcripts: list[Transcript] = list(existing)
    with ThreadPoolExecutor(max_workers=min(len(game_nums), 4)) as pool:
        futures = [
            pool.submit(
                run_rollout,
                game_num=game_num,
                agent_model=agent_model,
                max_turns=max_turns,
                thinking_budget=thinking_budget,
                feedback=feedback,
                effort=effort,
                cot_source=cot_source,  # type: ignore[arg-type]
                max_output_tokens=max_output_tokens,
                max_moves_per_call=max_moves_per_call,
            )
            for game_num in game_nums
        ]
        for future in as_completed(futures):
            transcript = future.result()
            if transcript.transcript_id in keep:
                transcripts = [
                    t for t in transcripts if t.transcript_id != transcript.transcript_id
                ]
            transcripts.append(transcript)
            logger.info(
                f"game {transcript.game_num}: {len(transcript.turns)} turns, won={transcript.won}"
            )
            write_jsonl(transcripts, out, atomic=True)  # checkpoint after every game
    transcripts.sort(key=lambda transcript: transcript.transcript_id)
    write_jsonl(transcripts, out, atomic=True)
    typer.echo(f"{len(transcripts)} transcripts -> {out}")


@app.command()
def channels(
    transcripts: Annotated[Path, typer.Option()] = Path("data/transcripts.jsonl"),
    out: Annotated[Path, typer.Option()] = Path("data/channels.json"),
) -> None:
    """Report per-arm channel usage: does a scratchpad relocate reasoning or reveal it?

    Headline is co-use — P(native thinking present | the player wrote a pad) — with pad-use rate as
    its denominator, and the ``native`` arm as the null control (pad use there must be 0.00).
    """
    rows = read_jsonl_rows(transcripts, Transcript)
    present = [(t.cot_source or "native") for t in rows]
    arms = [arm for arm in ("native", *SCRATCHPAD_SOURCES) if arm in present]
    if not arms:
        raise typer.BadParameter(f"no recognised cot_source in {transcripts}")
    splits = [split_for(rows, arm) for arm in arms]
    typer.echo(render_table(splits))
    control = next((s for s in splits if s.cot_source == "native"), None)
    if control is not None and control.pad_use.successes:
        # Loud, not silent: the pad detector firing on an arm that was never offered the tool
        # invalidates every other number in this table.
        logger.error(
            f"NULL CONTROL FAILED: native arm shows {control.pad_use.successes} pad turns; "
            "the pad detector is firing on something else"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([split.model_dump() for split in splits], indent=2), encoding="utf-8")
    typer.echo(f"\n-> {out}")


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
    effort: Annotated[str, typer.Option(help="Adaptive-model effort level")] = "medium",
    out: Annotated[Path, typer.Option()] = Path("data/runs.jsonl"),
) -> None:
    """Run each monitor on every item under both conditions (with-cot and no-cot).

    Resumable: runs already checkpointed in ``out`` are skipped, and a call that fails
    even after retries is logged and left missing (so a re-run picks it up) rather than
    recorded as a monitor failure — infra faults must not score as zeros.
    """
    monitor_items = read_jsonl_rows(items_path, MonitorItem)
    donors = swapped_cot_donors(monitor_items)
    runs: list[MonitorRun] = read_jsonl_rows(out, MonitorRun) if out.exists() else []
    done = {(run.item_id, run.monitor_model, run.condition) for run in runs}
    calls = [
        (item, model.strip(), condition)
        for model in monitor_models.split(",")
        for condition in CONDITIONS
        for item in monitor_items
        if (item.item_id, model.strip(), condition) not in done
        # swapped-cot is undefined when no cross-transcript donor exists
        and not (condition == "swapped-cot" and item.item_id not in donors)
    ]
    if done:
        logger.info(f"resuming: {len(done)} runs already checkpointed, {len(calls)} to go")
    failures = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                run_monitor,
                item,
                monitor_model=model,
                condition=condition,
                thinking_budget=thinking_budget,
                effort=effort,
                donor_cot=donors.get(item.item_id),
            ): (item.item_id, model, condition)
            for item, model, condition in calls
        }
        for future in as_completed(futures):
            item_id, model, condition = futures[future]
            try:
                runs.append(future.result())
            except Exception as exc:  # noqa: BLE001 — contain one call; re-run resumes it
                failures += 1
                logger.warning(f"{item_id} [{model} / {condition}] failed, will resume: {exc}")
                continue
            logger.info(
                f"{item_id} [{model} / {condition}] done ({len(runs)}/{len(done) + len(calls)})"
            )
            write_jsonl(runs, out, atomic=True)  # checkpoint after every call
    write_jsonl(runs, out, atomic=True)  # ensure the file exists even with zero items
    if failures:
        logger.warning(f"{failures} calls failed on API faults — re-run `gloss monitor` to resume")
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
