"""Prompt loading: templates are ``.md`` files next to this module, never string literals.

Templates use ``string.Template`` (``$name``) substitution — chosen over ``str.format``
because the prompts themselves contain JSON braces. ``load_prompt`` fails loudly on a
missing placeholder rather than emitting a half-rendered prompt.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from string import Template

_PROMPT_DIR = Path(__file__).parent


@cache
def _template(name: str) -> Template:
    return Template((_PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8"))


def load_prompt(name: str, **values: str) -> str:
    """Render ``prompts/<name>.md``, substituting every ``$placeholder`` in ``values``."""
    return _template(name).substitute(**values)


def rules_block() -> str:
    """The shared FreeCell rules-and-notation block both the agent and monitor see."""
    return (_PROMPT_DIR / "rules_common.md").read_text(encoding="utf-8").strip()
