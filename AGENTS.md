# gloss — CoT legibility benchmark

A monitor model reconstructs FreeCell game state, proposed lines, and the next tool
call from another model's chain of thought; every claim is scored by the engine
(`src/gloss/freecell.py`), never a judge. See README.md for the design and provenance.

## Conventions

- `uv run pytest` / `uv run ruff format . && uv run ruff check .` / `uv run ty check` —
  all three must pass before any commit.
- Prompts are `.md` files under `src/gloss/prompts/`, rendered with `string.Template`
  (`$var`) — never string literals in code, never `str.format` (prompts contain JSON
  braces).
- Everything that crosses a file boundary is a pydantic model in `src/gloss/wire.py`;
  stages validate their inputs with `read_jsonl_rows`, never trust the writer.
- The engine is pure: `apply_move` returns a new state. Ground truth in transcripts is
  produced only by the engine — never hand-edit `data/*.jsonl`.
- Scoring changes must keep `tests/test_scoring.py`'s invariant: a perfect answer
  scores perfectly, an errored run scores zero (dropped runs would flatter monitors).
- Model ids are version-qualified everywhere (`claude-haiku-4-5-20251001`, not
  "haiku"), per touchstone's rule.

## Layout

```
src/gloss/freecell.py   # engine: MS deals, legal moves, render — the ground truth
src/gloss/wire.py       # transcript/item/answer/run pydantic wire types
src/gloss/rollout.py    # player harness (thinking + play tool)
src/gloss/monitor.py    # item building (CoT stripping) + monitor calls
src/gloss/scoring.py    # engine-settled scores + per-arm summaries
src/gloss/cli.py        # gloss deal|rollout|items|monitor|score
src/gloss/utils/        # copied from touchstone lab.utils (fs, jsonl)
data/                   # committed benchmark artifacts (JSONL per stage)
```
