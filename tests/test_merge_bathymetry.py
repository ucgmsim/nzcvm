"""Tests for filling a land DEM's sea with a coarser bathymetry."""

import numpy as np
import pytest
import xarray as xr
from typer.testing import CliRunner

from nzcvm.models.mesh import UPRIGHT_FRAME, surface_mesh
from nzcvm.scripts.merge_bathymetry import app, coast_weight, merge_bathymetry

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
    # The DEM's own nodes; the lattice also extends over the bathymetry.
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


def test_the_lattice_extends_over_the_bathymetry() -> None:
    merged = merge_bathymetry(_dem(), _bathymetry(extra=3000.0), taper=TAPER)
    x = merged.x.values
    np.testing.assert_allclose(np.diff(x), SPACING)
    assert x[0] == pytest.approx(1.0e6 - 2 * 250.0, abs=SPACING)
    assert x[-1] > 1.0e6 + 80 * SPACING + 3000.0 - SPACING
    # The DEM's nodes keep their positions and values.
    np.testing.assert_array_equal(
        merged.sel(x=_dem().x, y=_dem().y).values[:, :30], -100.0
    )
    np.testing.assert_allclose(merged.values[:, -5:], 200.0, rtol=1e-4)


def test_cli_writes_a_surface(tmp_path) -> None:
    pytest.importorskip("rioxarray")
    dem = _dem()
    land = tmp_path / "land.zarr"
    surface_mesh("land", dem.data, dem.x.values, dem.y.values).to_zarr(land)

    raster = _bathymetry().rio.write_crs("EPSG:2193")
    tif = tmp_path / "bathy.tif"
    raster.rio.to_raster(tif)

    out = tmp_path / "dem.zarr"
    result = CliRunner().invoke(
        app, [str(land), str(tif), str(out), "--taper", str(TAPER)]
    )
    assert result.exit_code == 0, result.output
    merged = xr.open_zarr(out)
    assert merged.attrs["frame"] == list(UPRIGHT_FRAME)
    np.testing.assert_allclose(merged.z.values[:, -5:], 200.0, rtol=1e-4)


def test_cli_rejects_a_bathymetry_not_in_nztm(tmp_path) -> None:
    pytest.importorskip("rioxarray")
    dem = _dem()
    land = tmp_path / "land.zarr"
    surface_mesh("land", dem.data, dem.x.values, dem.y.values).to_zarr(land)
    tif = tmp_path / "bathy.tif"
    _bathymetry().rio.write_crs("EPSG:3994").rio.to_raster(tif)

    result = CliRunner().invoke(app, [str(land), str(tif), str(tmp_path / "o.zarr")])
    assert result.exit_code != 0
