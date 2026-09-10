"""Tests for the synthetic dataset and the commands that write it out.

Closed-form data is data a reader can check by hand. These tests assert the
analytic properties the rest of the suite relies on: the shoreline sits where
elevation crosses zero, and a basin closes to zero thickness on its own
outline. They then read each written file back through the production reader
that consumes it.

``just synthetic`` builds the basin meshes themselves, since that goes through
gmsh and runs far too slowly for a unit test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import shapely
from hypothesis import given
from hypothesis import strategies as st
from typer.testing import CliRunner

from nzcvm import synthetic
from nzcvm.layers.coastline import _read_compressed_shapely_wkb
from nzcvm.scripts import construct_mesh
from nzcvm.scripts.convert_tomography import (
    MODEL_COLUMNS,
    ModelType,
    data_frame_to_mesh,
)
from nzcvm.scripts.synthetic import app

runner = CliRunner()

# Longitudes and latitudes comfortably inside the synthetic domain.
lons = st.floats(min_value=synthetic.DOMAIN.lon_min, max_value=synthetic.DOMAIN.lon_max)
lats = st.floats(min_value=synthetic.DOMAIN.lat_min, max_value=synthetic.DOMAIN.lat_max)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------


@given(lon=lons, lat=lats)
def test_normalise_round_trips(lon: float, lat: float) -> None:
    u, v = synthetic.DOMAIN.normalise(lon, lat)
    back_lon, back_lat = synthetic.DOMAIN.denormalise(u, v)
    assert back_lon == pytest.approx(lon)
    assert back_lat == pytest.approx(lat)


@given(lon=lons, lat=lats)
def test_normalise_stays_in_unit_square(lon: float, lat: float) -> None:
    u, v = synthetic.DOMAIN.normalise(lon, lat)
    assert 0.0 <= u <= 1.0
    assert 0.0 <= v <= 1.0


def test_sample_spans_the_domain() -> None:
    lon, lat = synthetic.DOMAIN.sample(8, 5)
    assert (lon[0], lon[-1]) == (synthetic.DOMAIN.lon_min, synthetic.DOMAIN.lon_max)
    assert (lat[0], lat[-1]) == (synthetic.DOMAIN.lat_min, synthetic.DOMAIN.lat_max)
    assert (len(lon), len(lat)) == (8, 5)


# ---------------------------------------------------------------------------
# Elevation, Vs30 and the shoreline
# ---------------------------------------------------------------------------


@given(v=st.floats(min_value=0.0, max_value=1.0))
def test_shoreline_is_the_zero_contour(v: float) -> None:
    """`coastline_u` has to agree with where `elevation` actually crosses zero."""
    lon, lat = synthetic.DOMAIN.denormalise(synthetic.coastline_u(v), v)
    assert float(synthetic.elevation(lon, lat)) == pytest.approx(0.0, abs=1e-9)


@given(v=st.floats(min_value=0.0, max_value=1.0))
def test_land_lies_west_of_the_shoreline(v: float) -> None:
    shore = float(synthetic.coastline_u(v))
    inland_lon, lat = synthetic.DOMAIN.denormalise(shore - 0.1, v)
    offshore_lon, _ = synthetic.DOMAIN.denormalise(shore + 0.1, v)
    assert synthetic.elevation(inland_lon, lat) > 0.0
    assert synthetic.elevation(offshore_lon, lat) < 0.0


def test_land_polygon_matches_the_elevation_sign() -> None:
    polygon = shapely.Polygon(synthetic.land())
    assert polygon.is_valid
    for u, v in [(0.1, 0.2), (0.5, 0.5), (0.6, 0.9)]:
        lon, lat = synthetic.DOMAIN.denormalise(u, v)
        inside = polygon.contains(shapely.Point(lon, lat))
        assert inside == bool(synthetic.elevation(lon, lat) > 0.0)


def test_land_polygon_extends_beyond_the_domain() -> None:
    """Only the seaward edge may come near the domain, or the coastline layer
    would measure a distance to the edge of the polygon instead of to a coast."""
    polygon = shapely.Polygon(synthetic.land())
    min_lon, min_lat, _, max_lat = polygon.bounds
    assert min_lon < synthetic.DOMAIN.lon_min
    assert min_lat < synthetic.DOMAIN.lat_min
    assert max_lat > synthetic.DOMAIN.lat_max


@given(lon=lons, lat=lats)
def test_vs30_stays_within_bounds(lon: float, lat: float) -> None:
    assert synthetic.VS30_MIN <= synthetic.vs30(lon, lat) <= synthetic.VS30_MAX


# ---------------------------------------------------------------------------
# Basins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(synthetic.BasinName))
def test_basin_closes_on_its_outline(name: synthetic.BasinName) -> None:
    basin = synthetic.BASINS[name]
    outline = basin.outline()
    sediment = basin.sediment(outline[:, 0], outline[:, 1])
    assert sediment == pytest.approx(np.zeros(len(outline)), abs=1e-9)


@pytest.mark.parametrize("name", list(synthetic.BasinName))
def test_basin_is_deepest_at_its_centre(name: synthetic.BasinName) -> None:
    basin = synthetic.BASINS[name]
    assert basin.sediment(basin.centre_lon, basin.centre_lat) == pytest.approx(
        basin.thickness
    )


@pytest.mark.parametrize("name", list(synthetic.BasinName))
def test_basement_never_rises_above_topography(name: synthetic.BasinName) -> None:
    basin = synthetic.BASINS[name]
    lon, lat = synthetic.DOMAIN.sample(21, 21)
    mesh_lon, mesh_lat = np.meshgrid(lon, lat)
    assert np.all(
        basin.basement(mesh_lon, mesh_lat) <= synthetic.elevation(mesh_lon, mesh_lat)
    )


@pytest.mark.parametrize("name", list(synthetic.BasinName))
def test_basin_sits_inside_the_domain(name: synthetic.BasinName) -> None:
    outline = shapely.Polygon(synthetic.BASINS[name].outline())
    assert synthetic.DOMAIN.polygon.contains(outline)


# ---------------------------------------------------------------------------
# Tomography
# ---------------------------------------------------------------------------


def test_tomography_covers_the_domain() -> None:
    frame = synthetic.tomography()
    corners = np.asarray(synthetic.DOMAIN.polygon.exterior.coords)
    assert frame["Longitude"].min() < corners[:, 0].min()
    assert frame["Longitude"].max() > corners[:, 0].max()
    assert frame["Latitude"].min() < corners[:, 1].min()
    assert frame["Latitude"].max() > corners[:, 1].max()


def test_tomography_starts_above_sea_level() -> None:
    """The topography rises well over sea level, so the block has to as well."""
    frame = synthetic.tomography()
    assert frame["Depth(km_BSL)"].min() < 0.0


def test_tomography_converts_to_a_mesh() -> None:
    """The converter rejects anything that isn't a full rectilinear grid."""
    frame = synthetic.tomography(n_horizontal=4, n_depth=3)
    mesh = data_frame_to_mesh("synthetic", frame, MODEL_COLUMNS[ModelType.EP2020])
    assert mesh.sizes["i"] == 4 * 4 * 3
    assert np.all(mesh.vs.values > 0.0)
    assert np.all(mesh.vp.values > mesh.vs.values)


def test_tomography_speeds_up_with_depth() -> None:
    frame = synthetic.tomography().sort_values("Depth(km_BSL)")
    shallowest = frame[frame["Depth(km_BSL)"] == frame["Depth(km_BSL)"].min()]
    deepest = frame[frame["Depth(km_BSL)"] == frame["Depth(km_BSL)"].max()]
    assert deepest["Vp"].min() > shallowest["Vp"].max()


# ---------------------------------------------------------------------------
# Command output, read back through the production readers
# ---------------------------------------------------------------------------


def _invoke(*args: str | Path) -> None:
    result = runner.invoke(app, [str(arg) for arg in args])
    assert result.exit_code == 0, result.output


def test_dem_reads_back_as_a_surface(tmp_path: Path) -> None:
    path = tmp_path / "dem.h5"
    _invoke("dem", path, "--samples", "16")

    x, y, z = construct_mesh.read_surface_file(path)
    assert x.shape == y.shape == z.shape == (16, 16)
    # read_surface_file flips to the +z down convention used everywhere else.
    lon, lat = synthetic.DOMAIN.sample(16, 16)
    mesh_lon, mesh_lat = np.meshgrid(lon, lat)
    assert z == pytest.approx(-synthetic.elevation(mesh_lon, mesh_lat))


def test_vs30_reads_back_unflipped(tmp_path: Path) -> None:
    path = tmp_path / "vs30.h5"
    _invoke("vs30", path, "--samples", "16")

    from nzcvm.scripts.surface_cli import read_surface_file

    _, _, values = read_surface_file(path, scalar_key="vs30", flip=False)
    assert values.min() >= synthetic.VS30_MIN
    assert values.max() <= synthetic.VS30_MAX


def test_coastline_reads_back_projected(tmp_path: Path) -> None:
    path = tmp_path / "coastline.wkb.gz"
    _invoke("coastline", path)

    polygon = _read_compressed_shapely_wkb(path)
    assert polygon.is_valid
    # NZTM eastings and northings, not degrees.
    _, _, max_x, max_y = polygon.bounds
    assert 1_000_000 < max_x < 3_000_000
    assert 4_000_000 < max_y < 7_000_000


def test_profile_reads_back_as_a_layered_model(tmp_path: Path) -> None:
    path = tmp_path / "profile.fd_modfile"
    _invoke("profile", path)

    model = construct_mesh.read_layered_model(path)
    assert len(model) == len(synthetic.PROFILE)
    # read_layered_model rescales the km/s columns to SI.
    assert model["vs"].to_numpy() == pytest.approx(
        synthetic.PROFILE["vs"].to_numpy() * 1000.0
    )
    assert model["thickness"].min() > 0.0


def test_basin_writes_an_outline_and_a_basement(tmp_path: Path) -> None:
    _invoke("basin", "gully", tmp_path, "--samples", "16")

    outline_path = tmp_path / "gully_outline.geojson"
    basement_path = tmp_path / "gully_basement.h5"
    assert basement_path.exists()

    # construct_mesh reads the first geometry out of the collection.
    collection = shapely.from_geojson(outline_path.read_text())
    outline = collection.geoms[0]
    assert isinstance(outline, shapely.Polygon)
    basin = synthetic.BASINS[synthetic.BasinName.GULLY]
    assert outline.contains(shapely.Point(basin.centre_lon, basin.centre_lat))


def test_basement_surface_is_below_the_dem(tmp_path: Path) -> None:
    _invoke("dem", tmp_path / "dem.h5", "--samples", "16")
    _invoke("basin", "gully", tmp_path, "--samples", "16")

    _, _, topography = construct_mesh.read_surface_file(tmp_path / "dem.h5")
    _, _, basement = construct_mesh.read_surface_file(tmp_path / "gully_basement.h5")
    # Both are +z down, so the basement has the larger of the two values.
    assert np.all(basement >= topography)
    assert np.any(basement > topography)
