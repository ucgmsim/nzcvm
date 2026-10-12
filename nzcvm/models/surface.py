"""Surface interpolation for topography-based depth transforms.

A :class:`Surface` wraps a surface sampled on a regular grid and provides
point-query interpolation, used to convert depth-below-surface coordinates
into absolute elevations.

A :class:`SurfaceGrid` holds values on a regular latitude/longitude grid
projected into NZTM, interpolated bilinearly. :func:`read_surface_file` reads
one from a modeller-style HDF5 surface file, which holds 1-D ``latitude`` and
``longitude`` axes and 2-D scalar datasets indexed ``(latitude, longitude)``.
"""

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Self

import h5py
import numpy as np
import pyproj
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.nzcvm import (  # ty: ignore[unresolved-import]
    StructuredGrid,
    structured_grid,
)

DEFAULT_TOLERANCE = 1e-4

TRANSFORMER = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)
logger = logging.getLogger(__name__)


#: Tolerance on the fractional cell coordinates when deciding a point is
#: outside a :class:`Surface`.
OUTSIDE_TOLERANCE = 1e-9


@dataclass
class Surface:
    """A surface interpolator over a grid that's regular in its own frame.

    Given a set of (x, y) query points, returns the interpolated elevation
    (z) value at each location.  Used by grid builders to convert
    depth-below-surface coordinates to absolute elevations.

    A query maps its points into the surface's frame and finds each one's
    cell from the coordinate indexes, then gathers the cell's corners with
    one pointwise selection and interpolates bilinearly. A surface from
    :meth:`load` is lazily indexed: the selection reads only the chunks
    holding those corners.
    """

    #: ``z(y, x)`` with increasing 1-D frame coordinates.
    values: xr.DataArray
    #: The affine map from NZTM into the frame, as a 2 x 3 matrix.
    frame: np.ndarray

    @classmethod
    def from_dataset(cls, mesh: xr.Dataset) -> Self:
        """Build a surface over *mesh* (see :class:`~nzcvm.models.mesh.StructuredMeshSchema`)."""
        z = mesh.z
        # A plain reversed slice, unlike sortby, keeps lazy reads basic.
        for axis in ("y", "x"):
            if not z.indexes[axis].is_monotonic_increasing:
                z = z.isel({axis: slice(None, None, -1)})
        # The frame acts on (X, Y, 1).
        frame = np.asarray(mesh.attrs["frame"], dtype=np.float64).reshape(2, 3)
        return cls(z, frame)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Open the surface at *path*, lazily indexed.

        Parameters
        ----------
        path :
            Path to the surface's Zarr store.

        Returns
        -------
        Surface
            The surface, ready to interpolate.
        """
        # cache=False, or one read of the whole array would keep it all.
        dataset = xr.open_dataset(path, engine="zarr", chunks=None, cache=False)
        return cls.from_dataset(dataset)

    def to_frame(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """NZTM points ``(x, y)`` in the surface's frame."""
        frame_x, frame_y = self.frame @ np.stack([x, y, np.ones_like(x)])
        return frame_x, frame_y

    def from_frame(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Frame points ``(x, y)`` in NZTM."""
        linear, offset = self.frame[:, :2], self.frame[:, 2]
        nztm_x, nztm_y = np.linalg.solve(
            linear, np.stack([x - offset[0], y - offset[1]])
        )
        return nztm_x, nztm_y

    def __dask_tokenize__(self) -> object:
        """A token for dask that doesn't read the surface.

        Grid builders hand :meth:`transform` to dask, which tokenizes it to
        name tasks. Left to itself, dask hashes every value of a lazily
        indexed surface, reading all of it. A surface doesn't change once
        loaded. The object itself identifies it.
        """
        return type(self).__name__, id(self)

    @property
    def n_points(self) -> int:
        """Number of nodes."""
        return int(self.values.size)

    @property
    def corners(self) -> np.ndarray:
        """``(4, 2)`` NZTM positions of the grid's corner nodes."""
        x, y = self.values.x.values, self.values.y.values
        frame_x = np.array([x[0], x[-1], x[0], x[-1]])
        frame_y = np.array([y[0], y[0], y[-1], y[-1]])
        return np.column_stack(self.from_frame(frame_x, frame_y))

    @cached_property
    def bounds(self) -> np.ndarray:
        """``[xmin, ymin, zmin, xmax, ymax, zmax]`` of the surface.

        The z range reads every value. A lazy surface works it out only when
        asked.
        """
        corners = self.corners
        values = self.values.values
        return np.array(
            [
                corners[:, 0].min(),
                corners[:, 1].min(),
                float(np.nanmin(values)),
                corners[:, 0].max(),
                corners[:, 1].max(),
                float(np.nanmax(values)),
            ]
        )

    def transform(
        self, x: np.ndarray, y: np.ndarray, extrapolate: bool = False
    ) -> np.ndarray:
        """Interpolate surface elevation at query (x, y) locations.

        Parameters
        ----------
        x, y :
            Query point coordinates in the same projected CRS as the mesh.
        extrapolate :
            If True, a point outside the surface takes the edge value at its
            position clamped onto the grid, which extends the edge row or
            column outward. If False, it takes NaN. A point inside the
            surface where the surface has no value stays NaN either way.

        Returns
        -------
        numpy.ndarray
            Elevation (z) values with the same shape as *x*, NaN where a
            point falls outside the surface, unless *extrapolate* fills it.
        """
        logger.debug(f"Calculating z values for x, y (size = {x.size}).")
        frame_x, frame_y = self.to_frame(
            np.asarray(x, dtype=np.float64).ravel(),
            np.asarray(y, dtype=np.float64).ravel(),
        )
        if frame_x.size == 0:
            return np.empty(x.shape, dtype=x.dtype)

        def cell(axis: str, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            # The node at or before each position, and the fraction of the
            # way to the next node.
            index = self.values.indexes[axis]
            nodes = index.to_numpy()
            lower = np.clip(
                index.get_indexer(position, method="pad"), 0, len(nodes) - 2
            )
            fraction = (position - nodes[lower]) / (nodes[lower + 1] - nodes[lower])
            return lower, fraction

        ix, tx = cell("x", frame_x)
        iy, ty = cell("y", frame_y)
        outside = (
            (tx < -OUTSIDE_TOLERANCE)
            | (tx > 1 + OUTSIDE_TOLERANCE)
            | (ty < -OUTSIDE_TOLERANCE)
            | (ty > 1 + OUTSIDE_TOLERANCE)
        )
        tx, ty = np.clip(tx, 0.0, 1.0), np.clip(ty, 0.0, 1.0)
        corners = (
            self.values.isel(
                x=xr.DataArray(np.concatenate([ix, ix + 1, ix, ix + 1]), dims="point"),
                y=xr.DataArray(np.concatenate([iy, iy, iy + 1, iy + 1]), dims="point"),
            )
            .values.reshape(4, -1)
            .astype(np.float64)
        )
        z = (
            (1 - tx) * (1 - ty) * corners[0]
            + tx * (1 - ty) * corners[1]
            + (1 - tx) * ty * corners[2]
            + tx * ty * corners[3]
        )
        if not extrapolate:
            z[outside] = np.nan
        logger.debug("Query complete.")
        return z.reshape(x.shape).astype(x.dtype)

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render surface metadata as a rich tree.

        Yields
        ------
        rich.tree.Tree
            The metadata tree rich should display for this surface.
        """
        tree = Tree("Surface Interpolation")
        tree.add("Kind: Bilinear")
        # The corners alone, since the value range would read every value.
        (x_min, y_min), (x_max, y_max) = self.corners.min(0), self.corners.max(0)
        tree.add(f"Bounds: [X: {x_min:.0f}-{x_max:.0f}, Y: {y_min:.0f}-{y_max:.0f}]")
        tree.add(f"Number of points in surface: {self.n_points:,}")
        yield tree


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

    @property
    def inner(self) -> StructuredGrid:
        """The compiled grid, for Rust functions that take one."""
        return self._grid

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


def median_spacing(grid: SurfaceGrid) -> float:
    """The median distance between neighbouring nodes of *grid*, in metres."""
    x = np.asarray(grid.x, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    steps = np.concatenate(
        [
            np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)).ravel(),
            np.hypot(np.diff(x, axis=1), np.diff(y, axis=1)).ravel(),
        ]
    )
    return float(np.median(steps))


def _bbox_frame_range(
    surface: Surface, bbox: tuple[float, float, float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The x and y ranges, in *surface*'s frame, covering a lat/lon *bbox*."""
    min_lat, max_lat, min_lon, max_lon = bbox
    # Densified, since a latitude/longitude box curves in NZTM.
    x0, y0, x1, y1 = TRANSFORMER.transform_bounds(
        min_lon, min_lat, max_lon, max_lat, densify_pts=21
    )
    frame_x, frame_y = surface.to_frame(
        np.array([x0, x1, x0, x1]), np.array([y0, y0, y1, y1])
    )
    return (frame_x.min(), frame_x.max()), (frame_y.min(), frame_y.max())


def _node_spacing(z: xr.DataArray) -> float:
    """The smaller of an upright surface's two node spacings."""
    return float(min(np.diff(z.x.values[:2])[0], np.diff(z.y.values[:2])[0]))


def _window(
    surface: Surface, bbox: tuple[float, float, float, float] | None, factor: int = 1
) -> xr.DataArray:
    """The lazy window of *surface* covering *bbox*, on whole blocks of *factor*."""
    z = surface.values
    if bbox is None:
        return z
    (x_lo, x_hi), (y_lo, y_hi) = _bbox_frame_range(surface, bbox)
    margin = factor * _node_spacing(z)

    def window(axis: str, low: float, high: float) -> slice:
        # Start and stop on multiples of the block size, so blocks line
        # up with those of every other window.
        index = z.indexes[axis]
        start = max(int(index.searchsorted(low - margin)) - 1, 0)
        stop = min(int(index.searchsorted(high + margin)) + 1, len(index))
        return slice(
            start // factor * factor, min(-(-stop // factor) * factor, len(index))
        )

    return z.isel(x=window("x", x_lo, x_hi), y=window("y", y_lo, y_hi))


def surface_range(
    surface_path: Path, bbox: tuple[float, float, float, float]
) -> tuple[float, float]:
    """The least and greatest depth of a surface around *bbox*, at full resolution.

    Parameters
    ----------
    surface_path :
        Zarr surface or HDF5 elevation file.
    bbox :
        ``(min_lat, max_lat, min_lon, max_lon)`` to read.

    Returns
    -------
    tuple of float
        ``(min, max)`` depth over the nodes covering *bbox*.
    """
    if surface_path.suffix != ".zarr":
        z = np.asarray(read_surface_file(surface_path, bbox=bbox).values)
    else:
        # Lazily indexed, so this reads only the window.
        z = _window(Surface.load(surface_path), bbox).values
    return float(np.nanmin(z)), float(np.nanmax(z))


def read_surface_window(
    surface_path: Path,
    bbox: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
) -> SurfaceGrid:
    """Read a surface, as depth (+z down), around *bbox* onto an NZTM grid.

    A Zarr store contains a surface already stored as depth. This reads the
    window covering the box, started and stopped on whole blocks so blocks
    line up whichever window reads them, and averages it in blocks to about
    *spacing*. It reads anything else
    as an HDF5 elevation file with :func:`read_surface_file`, which negates
    it.

    Parameters
    ----------
    surface_path :
        Zarr surface or HDF5 elevation file.
    bbox :
        ``(min_lat, max_lat, min_lon, max_lon)``. If given, read only the
        nodes covering it.
    spacing :
        Node spacing, in metres, to average a Zarr surface to. The nearest
        whole number of nodes makes a block. Defaults to the surface's own
        spacing. It has no effect on an HDF5 surface.

    Returns
    -------
    SurfaceGrid
        The depths at projected nodes.
    """
    if surface_path.suffix != ".zarr":
        return read_surface_file(surface_path, bbox=bbox)
    surface = Surface.load(surface_path)
    node_spacing = _node_spacing(surface.values)
    factor = 1 if spacing is None else max(1, round(spacing / node_spacing))
    z = _window(surface, bbox, factor)
    # Lazily indexed, so this reads only the window.
    z = z.load()
    if factor > 1:
        z = z.coarsen(x=factor, y=factor, boundary="trim").mean()
    frame_x, frame_y = np.meshgrid(z.x.values, z.y.values)
    nztm_x, nztm_y = surface.from_frame(frame_x.ravel(), frame_y.ravel())
    return SurfaceGrid(
        nztm_x.reshape(frame_x.shape),
        nztm_y.reshape(frame_x.shape),
        np.asarray(z.values, dtype=np.float64),
    )
