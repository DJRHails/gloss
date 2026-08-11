"""Atomic file writes — the one tmp-sibling + rename helper the scripts share.

A mid-write kill (OOM, SIGKILL, node preemption) must never leave a truncated file that a
resume pass would read as complete: the full content goes to a ``<name>.tmp`` sibling first,
then ``Path.replace`` moves it over the target — an atomic rename on POSIX filesystems, so a
reader sees either the old file or the complete new one, never a torn write.
"""

from __future__ import annotations

from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` (UTF-8) via a tmp sibling + rename so a mid-write kill can't tear it."""
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` via a tmp sibling + rename so a mid-write kill can't tear it."""
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
