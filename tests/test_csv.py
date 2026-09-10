"""Tests for the flat CSV writer.

The interesting property is the header. A grid may hold coordinates past the
``(i, j, k)`` index, and the writer has to turn each one into a label column,
which is what lets a reader tell one borehole profile from another. The rest
checks that the table lists each grid point once, and that a round trip
through :func:`pandas.read_csv` recovers the float32 values exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr

from nzcvm.components import Component
from nzcvm.config.metadata import ModelMetadata
from nzcvm.coordinates import Coordinate
from nzcvm.formats import Format, from_path, write_velocity_model
from nzcvm.formats.csv import GRID_COLUMN, to_csv
from nzcvm.grids.grid import Grid, GridSchema
from nzcvm.qualities import QualitiesSchema
from nzcvm.velocity_model import VelocityModel

SHAPE = (2, 1, 3)


def _grid(name: str = "boreholes", sites: list[str] | None = None) -> Grid:
    """A small concrete grid, optionally labelled with a ``site`` coordinate."""
    ni, nj, nk = SHAPE
    i, j, k = np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij")
    grid = GridSchema.new(
        # Values distinct per point, so a misordered table shows up.
        x=(1_500_000.0 + i).astype(np.float32),
        y=(5_100_000.0 + j).astype(np.float32),
        z=(100.0 * i + k).astype(np.float32),
        depth=(25.0 * k).astype(np.float32),
        name=name,
        resolution=25.0,
        geometry=shapely.box(171.9, -43.6, 172.1, -43.4),
        origin_lon=np.float32(172.0),
        origin_lat=np.float32(-43.5),
        azimuth=np.float32(0.0),
        grid_azimuth=np.float32(0.0),
        bottom_left_lon=np.float32(172.0),
        bottom_left_lat=np.float32(-43.5),
    )
    if sites is not None:
        grid = grid.assign_coords({Coordinate.SITE: (Coordinate.I, sites)})
    return grid


def _model(*grids: Grid) -> VelocityModel:
    """Pair each grid with qualities that vary point by point."""
    qualities = {}
    for n, grid in enumerate(grids):
        ramp = xr.zeros_like(grid.x) + grid.depth + 1000.0 * n
        qualities[grid.name] = QualitiesSchema.new(
            rho=ramp + 1.0,
            vp=ramp + 2.0,
            vs=ramp + 3.0,
            qp=ramp + 4.0,
            qs=ramp + 5.0,
            alpha=xr.ones_like(grid.x),
        )
    return VelocityModel(
        grids={grid.name: grid for grid in grids},
        metadata=ModelMetadata(),
        qualities=qualities,
    )


@pytest.fixture()
def labelled(tmp_path: Path) -> pd.DataFrame:
    """The table written for a grid that holds site labels."""
    path = tmp_path / "boreholes.csv"
    to_csv(_model(_grid(sites=["GULL", "TERR"])), path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Format selection
# ---------------------------------------------------------------------------


def test_csv_is_inferred_from_the_extension() -> None:
    assert from_path(Path("boreholes.csv")) is Format.CSV


def test_quantisation_is_rejected(tmp_path: Path) -> None:
    """ZFP applies to the array stores, so it can't mean anything here."""
    with pytest.raises(ValueError, match="quantisation"):
        write_velocity_model(
            _model(_grid()), tmp_path / "out.csv", Format.CSV, quantise_arrays=True
        )


def test_write_velocity_model_dispatches_to_csv(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    write_velocity_model(_model(_grid()), path, Format.INFERRED, quantise_arrays=False)
    assert path.read_text().startswith(f"{GRID_COLUMN},")


# ---------------------------------------------------------------------------
# The header
# ---------------------------------------------------------------------------


def test_extra_coordinates_become_label_columns(labelled: pd.DataFrame) -> None:
    assert labelled.columns.tolist() == [
        GRID_COLUMN,
        Coordinate.SITE,
        Coordinate.I,
        Coordinate.J,
        Coordinate.K,
        Coordinate.X,
        Coordinate.Y,
        Coordinate.Z,
        Coordinate.DEPTH,
        *Component,
    ]


def test_a_grid_without_extra_coordinates_has_no_label_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / "plain.csv"
    to_csv(_model(_grid()), path)
    assert Coordinate.SITE not in pd.read_csv(path).columns


def test_labels_follow_their_own_axis(labelled: pd.DataFrame) -> None:
    """`site` lives on i, so every row of a column shares one label."""
    by_site = labelled.groupby(Coordinate.SITE.value)[Coordinate.I.value].unique()
    assert by_site["GULL"].tolist() == [0]
    assert by_site["TERR"].tolist() == [1]


def test_every_label_gets_a_full_profile(labelled: pd.DataFrame) -> None:
    counts = labelled[Coordinate.SITE].value_counts()
    assert counts.to_dict() == {"GULL": SHAPE[2], "TERR": SHAPE[2]}


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_one_row_per_grid_point(labelled: pd.DataFrame) -> None:
    assert len(labelled) == np.prod(SHAPE)
    index = [Coordinate.I.value, Coordinate.J.value, Coordinate.K.value]
    assert not labelled.duplicated(subset=index).any()


def test_values_round_trip_losslessly(tmp_path: Path) -> None:
    """The written precision recovers the float32 exactly."""
    grid = _grid(sites=["GULL", "TERR"])
    model = _model(grid)
    path = tmp_path / "boreholes.csv"
    to_csv(model, path)
    table = pd.read_csv(path)

    qualities = model.qualities[grid.name]
    for name, expected in [*grid.data_vars.items(), *qualities.data_vars.items()]:
        assert np.array_equal(
            table[name].to_numpy().astype(np.float32), expected.values.ravel()
        ), name


def test_grid_column_separates_multiple_grids(tmp_path: Path) -> None:
    """An SW4 domain writes one grid per refinement into the same table."""
    path = tmp_path / "refinements.csv"
    to_csv(_model(_grid("coarse"), _grid("fine")), path)

    table = pd.read_csv(path)
    assert table[GRID_COLUMN].value_counts().to_dict() == {
        "coarse": np.prod(SHAPE),
        "fine": np.prod(SHAPE),
    }
    # The second grid's qualities are offset by 1000, so the split is real.
    assert table.groupby(GRID_COLUMN).vs.min().to_dict() == {
        "coarse": 3.0,
        "fine": 1003.0,
    }


def test_eastings_avoid_scientific_notation(tmp_path: Path) -> None:
    """A seven-digit float32 easting reads as 1.5315095e+06 unless asked not to."""
    path = tmp_path / "boreholes.csv"
    to_csv(_model(_grid(sites=["GULL", "TERR"])), path)
    assert "e+06" not in path.read_text()
