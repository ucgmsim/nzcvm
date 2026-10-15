"""Tests for the ``nzcvm view model`` command.

The tests mock the plotter, so these tests don't need a display.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from typer.testing import CliRunner

from nzcvm.config.metadata import ModelMetadata
from nzcvm.formats import datatree
from nzcvm.scripts import view
from nzcvm.velocity_model import VelocityModel
from tests.test_sfile import _grid, _qualities

pv = pytest.importorskip("pyvista")

runner = CliRunner()


@pytest.fixture
def zarr_model(tmp_path: Path) -> Path:
    path = tmp_path / "model.zarr"
    model = VelocityModel(
        grids={"g0": _grid("g0", 100.0, 0.0)},
        qualities={"g0": _qualities()},
        metadata=ModelMetadata(),
    )
    datatree.to_zarr(model, path)
    return path


@pytest.fixture
def plotter(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    plotter = MagicMock()
    monkeypatch.setattr(pv, "Plotter", MagicMock(return_value=plotter))
    return plotter


def _invoke(*args: str) -> None:
    result = runner.invoke(
        view.app,
        ["model", *args, "--no-axes", "--slice-mode", "none", "--stride", "1"],
    )
    assert result.exit_code == 0, result.output


def test_zero_colour_limits_are_respected(zarr_model: Path, plotter: MagicMock):
    # The model's vs is well above 0, so a dropped --min-val 0 would show.
    _invoke(str(zarr_model), "--min-val", "0", "--max-val", "0")
    assert plotter.add_mesh.call_args.kwargs["clim"] == [0.0, 0.0]


def test_colour_limits_default_to_data_range(zarr_model: Path, plotter: MagicMock):
    _invoke(str(zarr_model))
    lo, hi = plotter.add_mesh.call_args.kwargs["clim"]
    assert lo > 0
    assert hi > lo


def test_log_diff_on_dask_grids():
    # The test grids are dask-backed, like grids read from disk.
    grid, qualities = _grid("g0", 100.0, 0.0), _qualities()
    other = _qualities(offset=100.0)
    mesh = view._build_structured_grid(
        0, grid, qualities, "vs", 1, grid, other, view.DiffMode.LOG
    )
    expected = np.log10(qualities.vs.values) - np.log10(other.vs.values)
    np.testing.assert_allclose(mesh.point_data["vs"], expected.ravel(order="F"))


def test_log_diff_from_cli(zarr_model: Path, plotter: MagicMock):
    _invoke(str(zarr_model), "--compare-to", str(zarr_model), "--diff-mode", "log")
    lo, hi = plotter.add_mesh.call_args.kwargs["clim"]
    assert lo == hi == 0.0
