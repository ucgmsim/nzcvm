"""Tests for converting a raster into a structured surface mesh."""

import numpy as np
import pytest
import xarray as xr

from nzcvm.scripts.convert_tiff import convert_tiff


def _raster(values: np.ndarray, step: float = 100.0) -> xr.DataArray:
    """A north-up raster on a uniform grid, like rasterio reads a GeoTIFF."""
    ny, nx = values.shape
    return xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "x": 1.0e6 + step * np.arange(nx),
            "y": 5.0e6 - step * np.arange(ny),
        },
    )


def _assert_uniform(mesh: xr.Dataset, step_x: float, step_y: float) -> None:
    x = mesh.x.values
    y = mesh.y.values
    np.testing.assert_allclose(np.diff(x, axis=1), step_x)
    np.testing.assert_allclose(np.diff(y, axis=0), step_y)


def test_crops_to_the_data_and_keeps_inner_no_data_rows() -> None:
    values = np.full((8, 6), np.nan, dtype=np.float32)
    values[2:7, 1:5] = 1.0
    values[4, :] = np.nan  # a no-data row inside the data
    mesh = convert_tiff("vs30", _raster(values), downsample=1)

    assert mesh.sizes == {"i": 5, "j": 4}
    _assert_uniform(mesh, 100.0, -100.0)
    assert np.isnan(mesh.z.values[2]).all()


@pytest.mark.parametrize("downsample", [2, 3])
def test_downsampled_grid_stays_uniform(downsample: int) -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(100.0, 800.0, (11, 7)).astype(np.float32)
    values[5, :] = np.nan
    mesh = convert_tiff("vs30", _raster(values), downsample=downsample)

    _assert_uniform(mesh, 100.0 * downsample, -100.0 * downsample)
    # The first node is the centre of the first full block.
    assert mesh.x.values[0, 0] == pytest.approx(1.0e6 + 50.0 * (downsample - 1))
    assert mesh.z.values[0, 0] == pytest.approx(
        np.nanmean(values[:downsample, :downsample]), rel=1e-6
    )
