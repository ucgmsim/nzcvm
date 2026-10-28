"""Command-line smoke tests.

These tests check that every top-level subcommand accepts ``--help`` and
exits cleanly (exit code 0), and that commands reject out-of-range options
as usage errors before any work starts.  These tests don't read a data file
or load a model configuration.
"""

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nzcvm.scripts.nzcvm_cli import app

runner = CliRunner()

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(output: str) -> str:
    """Strip the colour codes Rich adds when CI forces colour."""
    return ANSI.sub("", output)


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
        ["synthetic"],
        ["plugins"],
    ],
)
def test_help_exits_cleanly(args: list[str]) -> None:
    result = runner.invoke(app, args + ["--help"])
    assert result.exit_code == 0


@pytest.mark.parametrize("spacing", ["0", "-5"])
def test_surface_convert_rejects_non_positive_spacing(
    tmp_path: Path, spacing: str
) -> None:
    surface = tmp_path / "surface.h5"
    surface.touch()
    result = runner.invoke(
        app,
        ["surface", "convert", str(surface), str(tmp_path / "out.zarr")]
        + ["--spacing", spacing],
    )
    assert result.exit_code == 2
    assert "--spacing" in plain(result.output)


@pytest.mark.parametrize("downsample", ["0", "-2"])
def test_convert_tiff_rejects_non_positive_downsample(
    tmp_path: Path, downsample: str
) -> None:
    result = runner.invoke(
        app,
        ["convert-tiff", "main", str(tmp_path / "in.tif"), "1"]
        + [str(tmp_path / "out.zarr"), "--downsample", downsample],
    )
    assert result.exit_code == 2
    assert "--downsample" in plain(result.output)
