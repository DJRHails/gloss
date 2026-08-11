"""JSONL wire IO — the one writer and two readers the pipeline shares.

Copied from DJRHails/touchstone (``lab.utils.jsonl``), minus the checkpoint writer this
repo has no loop for. ``write_jsonl`` serialises each row by its wire type: a pydantic
model via ``model_dump_json()``, a mapping via ``json.dumps``, and a ``str`` passes
through as a pre-serialised line. ``read_jsonl`` is the lenient reader for raw wire rows
(blank lines skipped); ``read_jsonl_rows`` is the typed twin, validating every line into
the caller's row model at the boundary and treating a blank line as corruption rather
than padding. ``atomic=True`` stages the write through ``gloss.utils.fs`` for the
tmp+rename hand-off contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from gloss.utils.fs import atomic_write_text

WireRow = BaseModel | Mapping[str, object] | str


def write_jsonl(rows: Sequence[WireRow], out_path: Path, *, atomic: bool = False) -> None:
    """Write ``rows`` as one JSON line each, creating parent dirs.

    ``atomic=True`` stages the payload through a ``.tmp`` sibling + rename
    (:func:`gloss.utils.fs.atomic_write_text`), so a mid-write kill never leaves a
    truncated JSONL at the final path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(_wire_line(row) + "\n" for row in rows)
    if atomic:
        atomic_write_text(out_path, text)
    else:
        out_path.write_text(text, encoding="utf-8")
    logger.info(f"wrote {len(rows)} rows -> {out_path}")


def _wire_line(row: WireRow) -> str:
    if isinstance(row, BaseModel):
        return row.model_dump_json()
    if isinstance(row, str):
        return row
    return json.dumps(row)


def read_jsonl(path: Path) -> list[dict]:  # noqa: ANN401 — raw JSONL wire rows
    """Parse one JSON object per non-blank line of ``path``."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def read_jsonl_rows[RowT: BaseModel](path: Path, row_type: type[RowT]) -> list[RowT]:
    """Read a JSONL written by :func:`write_jsonl` back into typed rows.

    Every line is validated into ``row_type`` at the boundary, so a malformed or
    wrong-schema file fails loudly here rather than deep in a consumer. A blank line is
    corruption, not padding — :func:`write_jsonl` never emits one — so this raises
    instead of skipping. A file with no lines at all legitimately returns ``[]``.
    """
    rows: list[RowT] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(
                f"{path}: line {lineno} is blank — write_jsonl never emits blank lines, so this "
                "file was truncated, concatenated, or hand-edited; regenerate it rather than "
                "reading past the gap"
            )
        rows.append(row_type.model_validate_json(line))
    return rows
