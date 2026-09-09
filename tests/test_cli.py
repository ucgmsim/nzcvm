"""Command-line smoke tests.

These tests check one thing: every top-level subcommand accepts ``--help``
and exits cleanly (exit code 0).  These tests don't read a data file or load
a model configuration.
"""

import pytest
from typer.testing import CliRunner

from nzcvm.scripts.nzcvm_cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["generate"],
        ["basin"],
        ["tomography"],
        ["surface"],
        ["tree-stats"],
        ["view"],
    ],
)
def test_help_exits_cleanly(args: list[str]) -> None:
    result = runner.invoke(app, args + ["--help"])
    assert result.exit_code == 0
