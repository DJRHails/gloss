"""CLI help tests: the documented surface must match what the commands actually run."""

from __future__ import annotations

from typer.testing import CliRunner

from gloss.cli import CONDITIONS, app


def test_monitor_help_names_every_condition() -> None:
    # A wide terminal keeps rich from wrapping a condition name across lines.
    result = CliRunner().invoke(app, ["monitor", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    for condition in CONDITIONS:
        assert condition in result.output
    assert "both conditions" not in result.output
