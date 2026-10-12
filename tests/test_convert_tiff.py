"""Tests for converting a raster into a surface on an upright lattice."""

import numpy as np
import pytest
import xarray as xr

from nzcvm.models.mesh import UPRIGHT_FRAME
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


def test_crops_to_the_data_and_keeps_inner_no_data_rows() -> None:
    values = np.full((8, 6), np.nan, dtype=np.float32)
    values[2:7, 1:5] = 1.0
    values[4, :] = np.nan  # a no-data row inside the data
    mesh = convert_tiff("vs30", _raster(values), downsample=1)

    assert mesh.sizes == {"y": 5, "x": 4}
    np.testing.assert_allclose(mesh.x.values, 1.0e6 + 100.0 * np.arange(1, 5))
    np.testing.assert_allclose(mesh.y.values, 5.0e6 - 100.0 * np.arange(2, 7))
    assert mesh.attrs["frame"] == list(UPRIGHT_FRAME)
    assert np.isnan(mesh.z.values[2]).all()


@pytest.mark.parametrize("downsample", [2, 3])
def test_downsampling_averages_whole_blocks(downsample: int) -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(100.0, 800.0, (11, 7)).astype(np.float32)
    values[5, :] = np.nan
    mesh = convert_tiff("vs30", _raster(values), downsample=downsample)

    np.testing.assert_allclose(np.diff(mesh.x.values), 100.0 * downsample)
    np.testing.assert_allclose(np.diff(mesh.y.values), -100.0 * downsample)
    # The first node is the centre of the first block. Partial blocks at
    # the far edges drop out.
    half = 50.0 * (downsample - 1)
    assert mesh.x.values[0] == pytest.approx(1.0e6 + half)
    assert mesh.y.values[0] == pytest.approx(5.0e6 - half)
    assert mesh.sizes == {"y": 11 // downsample, "x": 7 // downsample}
    assert mesh.z.values[0, 0] == pytest.approx(
        np.nanmean(values[:downsample, :downsample]), rel=1e-6
    )


def test_flip_negates_the_values() -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    mesh = convert_tiff("dem", _raster(values), downsample=1, flip=True)
    np.testing.assert_array_equal(mesh.z.values, -values)
