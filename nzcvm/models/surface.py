"""Surface interpolation for topography-based depth transforms.

A :class:`Surface` wraps a surface sampled on a regular lattice and provides
point-query interpolation, used to convert depth-below-surface coordinates
into absolute elevations.

A :class:`SurfaceGrid` holds values on a regular latitude/longitude grid
projected into NZTM, interpolated bilinearly. :func:`read_surface_file` reads
one from a modeller-style HDF5 surface file, which holds 1-D ``latitude`` and
``longitude`` axes and 2-D scalar datasets indexed ``(latitude, longitude)``.
"""

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Self

import h5py
import numpy as np
import pyproj
import zarr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.models.mesh import StructuredMesh
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
    """A surface interpolator over a regular lattice of nodes.

    Given a set of (x, y) query points, returns the interpolated elevation
    (z) value at each location.  Used by grid builders to convert
    depth-below-surface coordinates to absolute elevations.

    The lattice places node ``[i, j]`` at ``origin + i * step_i + j *
    step_j``. A query maps the point back to fractional ``(i, j)``, which
    names its cell directly. The surface splits each cell into two triangles
    along the diagonal from node ``[i, j]`` to node ``[i + 1, j + 1]``, and
    a point's elevation interpolates barycentrically over the triangle
    containing it. An affine map keeps barycentric coordinates, so the
    weights come straight from the fractional index.

    A surface from :meth:`load` reads its values from the Zarr store as
    queries need them, a chunk at a time. A large surface costs memory only
    for the chunks a query touches.
    """

    #: ``(m, n)`` node values: a NumPy array, or a Zarr array read lazily.
    values: np.ndarray | zarr.Array
    origin: np.ndarray
    step_i: np.ndarray
    step_j: np.ndarray

    @classmethod
    def from_dataset(cls, mesh: StructuredMesh) -> Self:
        """Build a surface holding *mesh*'s values in memory."""
        return cls._placed(np.asarray(mesh.z.values, dtype=np.float32), mesh.attrs)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Open the surface at *path*, reading its values lazily.

        Parameters
        ----------
        path :
            Path to the surface's Zarr store.

        Returns
        -------
        Surface
            The surface, ready to interpolate.
        """
        group = zarr.open_group(path, mode="r")
        values = zarr.open_array(store=path, path="z", mode="r")
        return cls._placed(values, group.attrs)

    @classmethod
    def _placed(cls, values: np.ndarray | zarr.Array, attrs: Mapping) -> Self:
        origin, step_i, step_j = (
            np.asarray(attrs[key], dtype=np.float64)
            for key in ("origin", "step_i", "step_j")
        )
        return cls(values, origin, step_i, step_j)

    @property
    def n_points(self) -> int:
        """Number of nodes in the lattice."""
        return int(np.prod(self.values.shape))

    @property
    def corners(self) -> np.ndarray:
        """``(4, 2)`` positions of the lattice's four corner nodes."""
        ni, nj = self.values.shape
        corner_index = np.array([[0, 0], [ni - 1, 0], [0, nj - 1], [ni - 1, nj - 1]])
        return self.origin + corner_index @ np.stack([self.step_i, self.step_j])

    @cached_property
    def bounds(self) -> np.ndarray:
        """``[xmin, ymin, zmin, xmax, ymax, zmax]`` of the surface.

        The z range reads every value. A lazy surface works it out only when
        asked.
        """
        corners = self.corners
        values = np.asarray(self.values[:])
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

    def _gather(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """The values at nodes ``(i[k], j[k])``."""
        if isinstance(self.values, np.ndarray):
            return self.values[i, j]
        return np.asarray(self.values.vindex[i, j])

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
            position clamped onto the nearest edge cell, which extends the
            edge row or column outward. If False, it takes NaN. A point
            inside the surface where the surface has no value stays NaN
            either way.

        Returns
        -------
        numpy.ndarray
            Elevation (z) values with the same shape as *x*, NaN where a
            point falls outside the surface, unless *extrapolate* fills it.
        """
        logger.debug(f"Calculating z values for x, y (size = {x.size}).")
        x_flat = np.asarray(x, dtype=np.float64).ravel()
        y_flat = np.asarray(y, dtype=np.float64).ravel()
        if x_flat.size == 0:
            return np.empty(x.shape, dtype=x.dtype)

        inverse = np.linalg.inv(np.column_stack([self.step_i, self.step_j]))
        fi, fj = inverse @ np.stack([x_flat - self.origin[0], y_flat - self.origin[1]])
        ni, nj = self.values.shape
        i = np.clip(np.floor(fi), 0, ni - 2).astype(np.int64)
        j = np.clip(np.floor(fj), 0, nj - 2).astype(np.int64)
        t, s = fi - i, fj - j
        outside = (
            (t < -OUTSIDE_TOLERANCE)
            | (t > 1 + OUTSIDE_TOLERANCE)
            | (s < -OUTSIDE_TOLERANCE)
            | (s > 1 + OUTSIDE_TOLERANCE)
        )
        # Clamped onto the edge of its cell, a point takes the line between
        # the edge's two nodes, which both triangles agree on.
        t, s = np.clip(t, 0.0, 1.0), np.clip(s, 0.0, 1.0)

        # In (t, s), the triangle [i, j], [i + 1, j], [i + 1, j + 1] is
        # t >= s and the triangle [i, j], [i + 1, j + 1], [i, j + 1] is
        # t < s. Its barycentric coordinates at (t, s) weight the corners.
        upper = t >= s
        side_i = np.where(upper, i + 1, i)
        side_j = np.where(upper, j, j + 1)
        corner, side, diagonal = (
            self._gather(
                np.concatenate([i, side_i, i + 1]), np.concatenate([j, side_j, j + 1])
            )
            .reshape(3, -1)
            .astype(np.float64)
        )
        w_side = np.where(upper, t - s, s - t)
        w_diagonal = np.where(upper, s, t)
        z = (1.0 - w_side - w_diagonal) * corner + w_side * side + w_diagonal * diagonal
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
        tree.add("Kind: Linear/Sample")
        tree.add(
            f"Bounds: [X: {self.bounds[0]:.0f}-{self.bounds[3]:.0f}, Y: {self.bounds[1]:.0f}-{self.bounds[4]:.0f}]"
        )
        tree.add(f"Value Range: {self.bounds[2]:.0f}-{self.bounds[5]:.0f}")
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


@dataclass
class LatticeWindow:
    """A rectangular window of a lattice surface, held in memory.

    Node ``[i, j]`` of the window is at ``origin + i * step_i + j *
    step_j``, like the surface it came from.
    """

    #: ``(m, n)`` float32 node values, as depth (+z down).
    values: np.ndarray
    origin: np.ndarray
    step_i: np.ndarray
    step_j: np.ndarray

    @property
    def spacing(self) -> float:
        """The shorter of the two node steps, in metres."""
        return float(min(np.linalg.norm(self.step_i), np.linalg.norm(self.step_j)))

    def fractional_index(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """``(2, N)`` fractional ``(i, j)`` of the points ``(x, y)``."""
        inverse = np.linalg.inv(np.column_stack([self.step_i, self.step_j]))
        return inverse @ np.stack([x - self.origin[0], y - self.origin[1]])

    def coarsened(self, factor: int) -> Self:
        """Average blocks of *factor* x *factor* nodes into one node each.

        A block at the edge that runs past the window averages the nodes it
        has. The coarse node is at the centre of its block.
        """
        if factor <= 1:
            return self
        m, n = self.values.shape
        padded = np.full(
            (-(-m // factor) * factor, -(-n // factor) * factor), np.nan, np.float32
        )
        padded[:m, :n] = self.values
        blocks = padded.reshape(
            padded.shape[0] // factor, factor, padded.shape[1] // factor, factor
        )
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            values = np.nanmean(blocks, axis=(1, 3)).astype(np.float32)
        centre = (factor - 1) / 2
        return type(self)(
            values,
            self.origin + centre * (self.step_i + self.step_j),
            self.step_i * factor,
            self.step_j * factor,
        )

    def grid(self) -> SurfaceGrid:
        """The window as a :class:`SurfaceGrid` over its projected nodes."""
        i, j = np.meshgrid(
            np.arange(self.values.shape[0]),
            np.arange(self.values.shape[1]),
            indexing="ij",
        )
        x = self.origin[0] + i * self.step_i[0] + j * self.step_j[0]
        y = self.origin[1] + i * self.step_i[1] + j * self.step_j[1]
        return SurfaceGrid(x, y, self.values.astype(np.float64))


def read_lattice_window(
    surface_path: Path,
    bbox: tuple[float, float, float, float] | None = None,
    align: int = 1,
) -> LatticeWindow:
    """Read the nodes of a Zarr lattice surface covering *bbox*.

    It reads only the chunks under the box.

    Parameters
    ----------
    surface_path :
        Zarr lattice surface, stored as depth (+z down).
    bbox :
        ``(min_lat, max_lat, min_lon, max_lon)``. If given, read only the
        nodes covering it.
    align :
        Start and stop the window on node indices that are multiples of this.

    Returns
    -------
    LatticeWindow
        The nodes in the window.
    """
    surface = Surface.load(surface_path)
    ni, nj = surface.values.shape
    i_window = j_window = slice(None)
    if bbox is not None:
        min_lat, max_lat, min_lon, max_lon = bbox
        # A latitude/longitude box curves in NZTM, so project its whole
        # boundary rather than its corners.
        edge = np.linspace(0.0, 1.0, 64)
        lon = np.concatenate(
            [
                min_lon + (max_lon - min_lon) * edge,
                np.full(64, max_lon),
                max_lon - (max_lon - min_lon) * edge,
                np.full(64, min_lon),
            ]
        )
        lat = np.concatenate(
            [
                np.full(64, min_lat),
                min_lat + (max_lat - min_lat) * edge,
                np.full(64, max_lat),
                max_lat - (max_lat - min_lat) * edge,
            ]
        )
        x, y = TRANSFORMER.transform(lon, lat)
        whole = LatticeWindow(
            np.empty((0, 0), np.float32), surface.origin, surface.step_i, surface.step_j
        )
        fi, fj = whole.fractional_index(np.asarray(x), np.asarray(y))

        # Start and stop on multiples of *align*, so blocks of that many
        # nodes averaged from any window line up and are whole, except at the
        # surface's own edge.
        def aligned(low: float, high: float, size: int) -> slice:
            start = int(np.clip(np.floor(low) - 1, 0, size)) // align * align
            stop = -(-int(np.clip(np.ceil(high) + 2, 0, size)) // align) * align
            return slice(start, min(stop, size))

        i_window = aligned(fi.min(), fi.max(), ni)
        j_window = aligned(fj.min(), fj.max(), nj)
    i0 = i_window.start or 0
    j0 = j_window.start or 0
    return LatticeWindow(
        np.asarray(surface.values[i_window, j_window], dtype=np.float32),
        surface.origin + i0 * surface.step_i + j0 * surface.step_j,
        surface.step_i,
        surface.step_j,
    )


def read_surface_window(
    surface_path: Path,
    bbox: tuple[float, float, float, float] | None = None,
    spacing: float | None = None,
) -> SurfaceGrid:
    """Read a surface, as depth (+z down), around *bbox* onto an NZTM grid.

    A Zarr store contains a lattice surface already stored as depth. This
    reads only the chunks under the box and averages them in blocks to about
    *spacing*. It reads anything else as an HDF5 elevation file with
    :func:`read_surface_file`, which negates it.

    Parameters
    ----------
    surface_path :
        Zarr lattice surface or HDF5 elevation file.
    bbox :
        ``(min_lat, max_lat, min_lon, max_lon)``. If given, read only the
        nodes covering it.
    spacing :
        Node spacing, in metres, to average a lattice surface to. The
        nearest whole number of nodes makes a block. Defaults to the
        surface's own spacing. It has no effect on an HDF5 surface.

    Returns
    -------
    SurfaceGrid
        The depths at projected nodes.
    """
    if surface_path.suffix != ".zarr":
        return read_surface_file(surface_path, bbox=bbox)
    surface = Surface.load(surface_path)
    node_spacing = min(np.linalg.norm(surface.step_i), np.linalg.norm(surface.step_j))
    factor = 1 if spacing is None else max(1, round(spacing / node_spacing))
    # Blocks aligned to the whole surface average the same nodes whichever
    # basin reads them, so basins that share a surface agree on it.
    window = read_lattice_window(surface_path, bbox, align=factor)
    return window.coarsened(factor).grid()
