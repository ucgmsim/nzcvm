import dask.array as da
import numpy as np
import shapely
import xarray as xr

from nzcvm import coordinates
from nzcvm.coordinates import Affine, Coordinate
from nzcvm.models.surface import Surface


def compute_surface_elevation(
    topography: Surface,
    x: xr.DataArray,
    y: xr.DataArray,
) -> xr.DataArray:
    """Evaluate *topography* at each (x, y) grid point.

    Parameters
    ----------
    topography :
        Loaded topographic surface.
    x, y :
        Projected coordinates of the grid points.

    Returns
    -------
    xarray.DataArray
        Elevation values with the same shape and chunks as *x*.
    """
    return xr.apply_ufunc(
        topography.transform,
        x,
        y,
        dask="parallelized",
        output_dtypes=[x.dtype],
    )


def ensure_chunks(*dsets: xr.DataArray) -> list[xr.DataArray]:
    """Rechunk all arrays to the finest common chunk spec across all inputs.

    For each dimension the target is the chunk tuple with the most pieces, the
    finest split any input array carries.  Every output array then has chunks
    along *all* dimensions that any input chunked along, which avoids the
    single-chunk fallback when a 1-D array (a depth coordinate, say) broadcasts
    into a higher-dimensional space.
    """
    target: dict = {}
    for dset in dsets:
        for dim, sizes in dset.chunksizes.items():
            if dim not in target or len(sizes) > len(target[dim]):
                target[dim] = sizes
    return [dset.chunk(target) for dset in dsets]


def outline(transform: Affine, extent_x: float, extent_y: float) -> shapely.Geometry:
    dx = extent_x / 2
    dy = extent_y / 2
    # shape: (2, -1)
    corners_x_local = np.array([-dx, dx, dx, -dx])
    corners_y_local = np.array([-dy, -dy, dy, dy])
    corners_x, corners_y = coordinates.apply_affine_transform(
        transform, corners_x_local, corners_y_local
    )
    return shapely.Polygon(np.c_[corners_x, corners_y])


def raw_coordinates(
    ni: int,
    nj: int,
    resolution: float,
    offset: float,
    chunks: dict[Coordinate, int],
) -> tuple[xr.DataArray, xr.DataArray]:

    i = np.arange(ni)
    j = np.arange(nj)
    xi_raw = (offset + (i - ni / 2) * resolution).astype(np.float32)
    yi_raw = (offset + (j - nj / 2) * resolution).astype(np.float32)
    xi = da.from_array(xi_raw, chunks=(chunks[Coordinate.I]))
    xj = da.from_array(yi_raw, chunks=(chunks[Coordinate.J]))

    x_raw, y_raw = da.meshgrid(
        xi,
        xj,
        indexing="ij",
    )
    x_da = xr.DataArray(
        x_raw,
        dims=[Coordinate.I, Coordinate.J],
        coords={Coordinate.I: i, Coordinate.J: j},
    )
    y_da = xr.DataArray(
        y_raw,
        dims=[Coordinate.I, Coordinate.J],
        coords={Coordinate.I: i, Coordinate.J: j},
    )
    return x_da, y_da
