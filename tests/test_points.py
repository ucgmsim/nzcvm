"""Tests for the points grid and for running layer pipelines on it.

The topography is a plane, which bilinear interpolation reproduces exactly,
so every expected depth follows by hand.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
import shapely
import xarray as xr
from pyproj import CRS

from nzcvm.components import Component
from nzcvm.config.grids.model import Projection
from nzcvm.config.layers.clamp import Bound, ClampLayerConfig, Floor
from nzcvm.config.layers.coastline import CoastlineConfig
from nzcvm.config.layers.ely import ElyLayerConfig
from nzcvm.coordinates import Coordinate
from nzcvm.ely_taper import ely_vs_profile
from nzcvm.grids.grid import GridSchema
from nzcvm.grids.points import points_grid
from nzcvm.layers.dummy import ConstantLayer
from nzcvm.layers.pipeline import build_pipeline
from nzcvm.models.surface import Surface

#: An NZTM point near Christchurch, the centre of the test surfaces.
X0, Y0 = 1_570_000.0, 5_180_000.0
NODES = np.array([-1000.0, 0.0, 1000.0])
#: The surfaces' frame: NZTM shifted so (X0, Y0) is the origin.
FRAME = [1.0, 0.0, -X0, 0.0, 1.0, -Y0]

NZTM = Projection(crs=CRS.from_epsg(2193))
GEOMETRY = shapely.box(X0 - 10_000, Y0 - 10_000, X0 + 10_000, Y0 + 10_000)


def ground(x: np.ndarray) -> np.ndarray:
    """Elevation of the topography, positive down: 100 m up, rising eastward."""
    return -100.0 - 0.05 * (np.asarray(x) - X0)


#: Points, the elevation of each (positive down) and the depth it should get.
POINTS_X = np.array([X0, X0 + 500.0, X0 - 500.0, X0 + 5000.0])
POINTS_Y = np.array([Y0, Y0, Y0 + 200.0, Y0 - 300.0])
POINTS_Z = np.array([400.0, -200.0, 0.0, 850.0])
#: In order: deep, above the ground, shallow, and off the surface's east edge,
#: where the ground extends the edge at X0 + 1000.
DEPTHS = np.array([500.0, 0.0, 75.0, 1000.0])
GROUND = np.array([-100.0, -125.0, -75.0, -150.0])


def _surface(values: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {"z": (("y", "x"), values.astype(np.float32))},
        coords={"x": NODES, "y": NODES},
        attrs={"frame": FRAME},
    )


@pytest.fixture(scope="module")
def topography() -> Surface:
    x = np.broadcast_to(NODES + X0, (3, 3))
    return Surface.from_dataset(_surface(ground(x)))


def _grid(topography: Surface, **kwargs):
    return points_grid(POINTS_X, POINTS_Y, POINTS_Z, topography, NZTM, **kwargs)


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def test_each_point_is_a_one_sample_column(topography: Surface) -> None:
    grid = _grid(topography)
    n = len(POINTS_X)
    assert dict(grid.sizes) == {Coordinate.I: n, Coordinate.J: 1, Coordinate.K: 1}
    for name in (Coordinate.X, Coordinate.Y, Coordinate.Z, Coordinate.DEPTH):
        assert grid[name].dims == (Coordinate.I, Coordinate.J, Coordinate.K)
        assert grid[name].dtype == np.float32
    np.testing.assert_array_equal(grid[Coordinate.I], np.arange(n))
    np.testing.assert_array_equal(grid[Coordinate.K], [0])
    np.testing.assert_allclose(grid.x.values.ravel(), POINTS_X)
    np.testing.assert_allclose(grid.y.values.ravel(), POINTS_Y)


def test_depth_is_below_the_topography(topography: Surface) -> None:
    grid = _grid(topography)
    np.testing.assert_allclose(grid.depth.values.ravel(), DEPTHS, atol=1e-3)


def test_a_point_above_the_ground_moves_onto_it(topography: Surface) -> None:
    """The point at z = -200 sits 75 m over the ground at -125."""
    grid = _grid(topography)
    expected = np.where(DEPTHS > 0, POINTS_Z, GROUND)
    np.testing.assert_allclose(grid.z.values.ravel(), expected, atol=1e-3)
    # The grid keeps the invariant every builder does: z is the ground plus
    # the depth.
    np.testing.assert_allclose((grid.z - grid.depth).values.ravel(), GROUND, atol=1e-3)


def test_fills_the_schema(topography: Surface) -> None:
    grid = _grid(topography, name="fault")
    # Round-trips through the schema, so no field is missing.
    GridSchema.from_dataset(grid)
    assert grid.attrs["name"] == "fault"
    assert grid.attrs["solver"] == "physical"
    assert np.isnan(grid.attrs["resolution"])
    assert np.isnan(grid[Coordinate.NOMINAL_DEPTH]).all()
    assert grid.attrs["geometry"].equals(
        shapely.MultiPoint(np.column_stack((POINTS_X, POINTS_Y)))
    )
    # Christchurch, give or take.
    assert grid.attrs["origin_lon"] == pytest.approx(172.6, abs=0.2)
    assert grid.attrs["origin_lat"] == pytest.approx(-43.5, abs=0.2)
    assert grid.attrs["bottom_left_lon"] < grid.attrs["origin_lon"]
    assert grid.attrs["bottom_left_lat"] < grid.attrs["origin_lat"]


def test_flattens_broadcast_inputs(topography: Surface) -> None:
    x = POINTS_X[:2].reshape(2, 1) + np.array([0.0, 100.0, 200.0])
    grid = points_grid(x, Y0, 300.0, topography, NZTM)
    assert grid.sizes[Coordinate.I] == 6
    np.testing.assert_allclose(grid.x.values.ravel(), x.ravel())
    np.testing.assert_allclose(grid.y.values.ravel(), Y0)
    np.testing.assert_allclose(
        grid.depth.values.ravel(), 300.0 - ground(x.ravel()), atol=1e-3
    )


def test_rejects_no_points(topography: Surface) -> None:
    with pytest.raises(ValueError, match="at least one point"):
        points_grid([], [], [], topography, NZTM)


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def _constant(**kwargs: float):
    return ConstantLayer.config_cls(**kwargs)


def test_constant_pipeline(topography: Surface) -> None:
    grid = _grid(topography)
    qualities = build_pipeline(GEOMETRY, [_constant(vs=3000.0)])(grid)
    assert qualities.vs.shape == (len(POINTS_X), 1, 1)
    np.testing.assert_allclose(qualities.vs.values, 3000.0)


def test_clamp_depth_floor_reads_depth_below_the_ground(topography: Surface) -> None:
    """The floor holds at and below 200 m, so it catches the two deep points."""
    clamp = ClampLayerConfig(
        clamps={Component.VS: Bound(floors=[Floor(min=4000.0, depth=200.0)])}
    )
    pipeline = build_pipeline(GEOMETRY, [clamp, _constant(vs=3000.0)])
    vs = pipeline(_grid(topography)).vs.values.ravel()
    np.testing.assert_allclose(vs, np.where(DEPTHS >= 200.0, 4000.0, 3000.0))


VS30 = 400.0
DEPTH_T = 350.0
VP_T, VS_T = 4000.0, 2000.0


@pytest.fixture()
def ely_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """A constant Vs30 surface and a coastline with everything on land."""
    vs30 = tmp_path / "vs30.zarr"
    _surface(np.full((3, 3), VS30)).to_zarr(vs30, mode="w")
    coastline = tmp_path / "coastline.wkb.gz"
    with gzip.open(coastline, "wb") as handle:
        handle.write(shapely.to_wkb(GEOMETRY))
    return vs30, coastline


def test_ely_pipeline_tapers_shallow_points(
    topography: Surface, ely_inputs: tuple[Path, Path]
) -> None:
    """The Ely layer reads each column's top by ``k``, which the grid provides.

    The constant layer below has alpha 0, so it isn't a basin. The taper then
    applies above ``depth_t`` and the constant stands below it.
    """
    vs30, coastline = ely_inputs
    pipeline = build_pipeline(
        GEOMETRY,
        [
            CoastlineConfig(coastline=coastline),
            ElyLayerConfig(vs30=vs30, depth_t=DEPTH_T),
            _constant(vp=VP_T, vs=VS_T, alpha=0.0),
        ],
    )
    # Only the first three points lie on the Vs30 surface.
    grid = _grid(topography).isel({Coordinate.I: slice(0, 3)})
    qualities = pipeline(grid)
    assert qualities.vs.shape == (3, 1, 1)

    depth = xr.DataArray(DEPTHS[:3].reshape(-1, 1, 1), dims=grid.depth.dims)
    profile = ely_vs_profile(
        depth.clip(max=DEPTH_T),
        xr.full_like(depth, VS30),
        xr.full_like(depth, VP_T),
        xr.full_like(depth, VS_T),
        depth_t=DEPTH_T,
    )
    tapered = DEPTHS[:3] < DEPTH_T
    np.testing.assert_allclose(
        qualities.vs.values.ravel(),
        np.where(tapered, profile.vs.values.ravel(), VS_T),
        rtol=1e-5,
    )
    # The point lifted onto the ground reads the surface value, half the Vs30
    # relation's Vs.
    assert qualities.vs.values.ravel()[1] == pytest.approx(0.5 * VS30, rel=1e-5)
