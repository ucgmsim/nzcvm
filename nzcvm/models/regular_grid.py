"""Readers for modeller-style HDF5 surface files on regular grids.

A surface file holds 1-D ``latitude`` and ``longitude`` axes and one or more
2-D scalar datasets (``elevation``, ``vs30``, ...) indexed ``(latitude,
longitude)``. :func:`read_surface_file` projects the axes into NZTM and wraps
the chosen scalar in a :class:`SurfaceGrid` for bilinear interpolation.
"""

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pyproj

from nzcvm.nzcvm import (  # ty: ignore[unresolved-import]
    StructuredGrid,
    structured_grid,
)

TRANSFORMER = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)


@dataclass
class SurfaceGrid:
    """Values on a structured grid of Cartesian nodes, interpolated bilinearly.

    The values are depths (positive down) for a surface, or mesh sizes for a
    sizing field. The cells can be any quadrilaterals, such as those of a
    latitude/longitude grid projected into a Cartesian frame. Interpolation is
    bilinear within each cell. Outside the grid, the kernel clamps the query
    point onto the edge cell, which extends the edge row or column outward.
    """

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    _grid: StructuredGrid = field(init=False, repr=False)

    def __post_init__(self):
        self._grid = structured_grid(
            np.asarray(self.x, dtype=np.float64),
            np.asarray(self.y, dtype=np.float64),
            np.asarray(self.values, dtype=np.float64),
        )

    def __call__(
        self, x: np.ndarray, y: np.ndarray, clamp: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate the grid at points in the grid's frame.

        Parameters
        ----------
        x, y : np.ndarray
            Coordinates of the query points, in the same frame as the grid.
        clamp : bool
            If True, clamp points outside the grid onto its edge. If False,
            they take NaN.

        Returns
        -------
        values : np.ndarray
            Interpolated values at each query point.
        outside : np.ndarray
            Mask of the query points that lie outside the grid.
        """
        return self._grid.query_many(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), clamp
        )

    def query(self, x: float, y: float, clamp: bool = True) -> tuple[float, bool]:
        """Interpolate the grid at one point, as for :meth:`__call__`."""
        return self._grid.query(x, y, clamp)


def read_surface_file(
    surface_path: Path,
    scalar_key: str = "elevation",
    flip: bool = True,
    bbox: tuple[float, float, float, float] | None = None,
) -> SurfaceGrid:
    """Read one scalar of an HDF5 surface file onto an NZTM grid.

    Parameters
    ----------
    surface_path : Path
        HDF5 file with ``latitude`` and ``longitude`` axes.
    scalar_key : str
        Name of the 2-D dataset to read.
    flip : bool
        If True, negate the values, converting elevation (+z up) to depth
        (+z down).
    bbox : tuple of float, optional
        ``(min_lat, max_lat, min_lon, max_lon)``. If given, read only the
        nodes inside it. A box that misses the grid reads the whole file.

    Returns
    -------
    SurfaceGrid
        The values, indexed ``(latitude, longitude)``, at projected nodes.
    """
    with h5py.File(surface_path, "r") as f:
        latitude = f["latitude"][:]
        longitude = f["longitude"][:]
        lat_slice = lon_slice = slice(None)

        if bbox is not None:
            min_lat, max_lat, min_lon, max_lon = bbox
            lat_idxs = np.where((latitude >= min_lat) & (latitude <= max_lat))[0]
            lon_idxs = np.where((longitude >= min_lon) & (longitude <= max_lon))[0]

            if len(lat_idxs) > 0 and len(lon_idxs) > 0:
                lat_slice = slice(lat_idxs.min(), lat_idxs.max() + 1)
                lon_slice = slice(lon_idxs.min(), lon_idxs.max() + 1)
                latitude = latitude[lat_slice]
                longitude = longitude[lon_slice]

        values = f[scalar_key][lat_slice, lon_slice]

    if flip:
        values = -values

    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    x, y = TRANSFORMER.transform(lon_grid, lat_grid)
    return SurfaceGrid(x, y, values)
