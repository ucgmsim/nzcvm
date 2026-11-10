"""Tests for filling a land DEM's sea with a coarser bathymetry."""

import numpy as np
import pytest
import xarray as xr
from typer.testing import CliRunner

from nzcvm.models.mesh import UPRIGHT_FRAME, surface_mesh
from nzcvm.scripts.merge_bathymetry import (
    app,
    coast_weight,
    merge_bathymetry,
    round_mantissa,
)

SPACING = 30.0
TAPER = 300.0


def _dem(nx: int = 80, ny: int = 40, coast: int = 30) -> xr.DataArray:
    """Depth: land (100 m up) west of column *coast*, sea (0) east of it."""
    z = np.zeros((ny, nx), dtype=np.float32)
    z[:, :coast] = -100.0
    return xr.DataArray(
        z,
        dims=("y", "x"),
        coords={
            "x": 1.0e6 + SPACING * np.arange(nx),
            "y": 5.0e6 - SPACING * np.arange(ny),
        },
    ).chunk(16)


def _bathymetry(
    elevation: float = -200.0, step: float = 250.0, extra: float = 0.0
) -> xr.DataArray:
    """A flat seafloor on a coarse lattice, overlapping the DEM and *extra* m east."""
    x = np.arange(1.0e6 - 2 * step, 1.0e6 + 80 * SPACING + extra + 2 * step, step)
    y = np.arange(5.0e6 + 2 * step, 5.0e6 - 40 * SPACING - 2 * step, -step)
    return xr.DataArray(
        np.full((len(y), len(x)), elevation, dtype=np.float32),
        dims=("y", "x"),
        coords={"x": x, "y": y},
    )


def test_coast_weight_is_a_smoothstep_of_the_distance() -> None:
    sea = np.ones((1, 20), dtype=bool)
    sea[0, 0] = False
    w = coast_weight(sea, SPACING, TAPER)
    s = np.clip(SPACING * np.arange(20) / TAPER, 0, 1)
    np.testing.assert_allclose(w[0], s * s * (3 - 2 * s), atol=1e-6)


def test_coast_weight_without_land_is_one() -> None:
    np.testing.assert_array_equal(
        coast_weight(np.ones((4, 4), dtype=bool), SPACING, TAPER), 1.0
    )


@pytest.mark.parametrize("order", [1, 3])
def test_land_is_kept_and_the_sea_tapers_into_the_bathymetry(order: int) -> None:
    dem = _dem()
    merged = merge_bathymetry(dem, _bathymetry(), taper=TAPER, order=order)
    z = merged.sel(x=dem.x, y=dem.y).values

    np.testing.assert_array_equal(z[:, :30], -100.0)
    # Depth grows monotonically offshore, from near 0 at the coast...
    row = z[20, 30:]
    assert np.all(np.diff(row) >= -1e-3)
    assert row[0] < 200.0 * 0.05
    # ...to the bathymetry once past the taper, across chunk boundaries.
    far = 30 + int(TAPER / SPACING) + 1
    np.testing.assert_allclose(z[:, far:], 200.0, rtol=1e-4)


def test_the_taper_reaches_across_chunks_exactly() -> None:
    dem = _dem()
    lazy = merge_bathymetry(dem, _bathymetry(), taper=TAPER).values
    whole = merge_bathymetry(dem.compute(), _bathymetry(), taper=TAPER).values
    np.testing.assert_allclose(lazy, whole, atol=1e-4)


def test_land_in_the_bathymetry_stays_at_sea_level() -> None:
    dem = _dem()
    merged = merge_bathymetry(dem, _bathymetry(elevation=50.0), taper=TAPER)
    z = merged.sel(x=dem.x, y=dem.y).values
    np.testing.assert_array_equal(z[:, 30:], 0.0)


def test_the_result_is_on_the_dem_lattice() -> None:
    dem = _dem()
    merged = merge_bathymetry(dem, _bathymetry(), taper=TAPER)
    np.testing.assert_array_equal(merged.x, dem.x)
    np.testing.assert_array_equal(merged.y, dem.y)


def _sloping_raster(crs: str, step: float = 250.0) -> xr.DataArray:
    """Elevation -100 - (x - 1e6) / 10 in NZTM x, on a *crs* lattice.

    It covers the DEM 10 km past each side, in the raster's own projection.
    """
    from pyproj import Transformer

    to_crs = Transformer.from_crs("EPSG:2193", crs, always_xy=True)
    to_nztm = Transformer.from_crs(crs, "EPSG:2193", always_xy=True)
    cx, cy = to_crs.transform(
        [1.0e6 - 1e4, 1.0e6 + 80 * SPACING + 1e4] * 2,
        [5.0e6 + 1e4] * 2 + [5.0e6 - 40 * SPACING - 1e4] * 2,
    )
    x = np.arange(min(cx), max(cx) + step, step)
    y = np.arange(max(cy), min(cy) - step, -step)
    gx, gy = np.meshgrid(x, y)
    nx, _ = to_nztm.transform(gx, gy)
    values = (-100.0 - (nx - 1.0e6) / 10.0).astype(np.float32)
    raster = xr.DataArray(values, dims=("y", "x"), coords={"x": x, "y": y})
    return raster.rio.write_crs(crs)


@pytest.mark.parametrize("crs", ["EPSG:2193", "EPSG:3994"])
def test_cli_warps_the_bathymetry_under_the_dem(tmp_path, crs: str) -> None:
    pytest.importorskip("rioxarray")
    dem = _dem()
    land = tmp_path / "land.zarr"
    surface_mesh("land", dem.data, dem.x.values, dem.y.values).to_zarr(land)
    tif = tmp_path / "bathy.tif"
    _sloping_raster(crs).rio.to_raster(tif)

    out = tmp_path / "dem.zarr"
    result = CliRunner().invoke(
        app, [str(land), str(tif), str(out), "--taper", str(TAPER), "--keep-bits", "23"]
    )
    assert result.exit_code == 0, result.output
    merged = xr.open_zarr(out)
    assert merged.attrs["frame"] == list(UPRIGHT_FRAME)
    np.testing.assert_array_equal(merged.x, dem.x)
    np.testing.assert_array_equal(merged.y, dem.y)
    # Past the taper, the depth is the slope, up to the bilinear warp.
    far = 30 + int(TAPER / SPACING) + 1
    expected = 100.0 + (dem.x.values[far:] - 1.0e6) / 10.0
    np.testing.assert_allclose(
        merged.z.values[:, far:],
        np.broadcast_to(expected, (40, len(expected))),
        atol=0.1,
    )


@pytest.mark.parametrize("keep", [0, 7, 10, 23])
def test_round_mantissa_bounds_the_relative_error(keep: int) -> None:
    rng = np.random.default_rng(1)
    values = rng.uniform(-6000.0, 3000.0, 1000).astype(np.float32)
    rounded = round_mantissa(values, keep)
    if keep == 23:
        np.testing.assert_array_equal(rounded, values)
    else:
        bound = np.abs(values) * 2.0 ** -(keep + 1)
        assert np.all(np.abs(rounded - values) <= bound)
    assert np.all(round_mantissa(np.zeros(3, np.float32), keep) == 0.0)


def test_rounding_leaves_land_exact() -> None:
    dem = _dem()
    dem[:, :30] = -123.456789
    merged = merge_bathymetry(dem, _bathymetry(-1234.5678), taper=TAPER, keep_bits=7)
    z = merged.sel(x=dem.x, y=dem.y).values
    np.testing.assert_array_equal(z[:, :30], np.float32(-123.456789))
    np.testing.assert_allclose(z[:, -5:], 1234.5678, rtol=2.0**-8)
