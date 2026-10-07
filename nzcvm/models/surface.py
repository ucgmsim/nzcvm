"""Surface interpolation for topography-based depth transforms.

A :class:`Surface` wraps a surface mesh and provides point-query
interpolation, used to convert depth-below-surface coordinates into
absolute elevations.

A :class:`SurfaceGrid` holds values on a regular latitude/longitude grid
projected into NZTM, interpolated bilinearly. :func:`read_surface_file` reads
one from a modeller-style HDF5 surface file, which holds 1-D ``latitude`` and
``longitude`` axes and 2-D scalar datasets indexed ``(latitude, longitude)``.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import h5py
import numpy as np
import pyproj
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm import registry
from nzcvm.models.mesh import StructuredMesh, StructuredMeshSchema, triangulate
from nzcvm.nzcvm import (  # ty: ignore[unresolved-import]
    PySurfaceModel,
    StructuredGrid,
    structured_grid,
    surface_model,
)

DEFAULT_TOLERANCE = 1e-4

TRANSFORMER = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)
logger = logging.getLogger(__name__)


@dataclass
class Surface:
    """A surface interpolator backed by a triangulated mesh.

    Given a set of (x, y) query points, returns the interpolated elevation
    (z) value at each location.  Used by grid builders to convert
    depth-below-surface coordinates to absolute elevations.
    """

    inner: PySurfaceModel
    bounds: np.ndarray
    n_points: int

    @classmethod
    def from_dataset(cls, mesh: StructuredMesh) -> Self:
        points = np.c_[mesh.x.values.ravel(), mesh.y.values.ravel()]
        z = mesh.z.values.ravel()
        faces = triangulate(mesh)
        logger.debug("Constructing inner surface model")
        inner = surface_model(points, faces, z)
        logger.debug("Inner model constructed.")

        bounds = np.array(
            [
                points[..., 0].min(),
                points[..., 1].min(),
                float(z.min()),
                points[..., 0].max(),
                points[..., 1].max(),
                float(z.max()),
            ]
        )

        return cls(inner, bounds=bounds, n_points=len(points))

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a surface mesh from *surface_path* and return a :class:`Surface`.

        Parameters
        ----------
        path :
            Path to the surface mesh file.

        Returns
        -------
        Surface
            The loaded surface, ready to interpolate.
        """
        with xr.open_dataset(path) as dset:
            mesh = StructuredMeshSchema.from_dataset(dset)
            return cls.from_dataset(mesh)

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Interpolate surface elevation at query (x, y) locations.

        Parameters
        ----------
        x, y :
            Query point coordinates in the same projected CRS as the mesh.

        Returns
        -------
        numpy.ndarray
            Elevation (z) values with the same shape as *x*, NaN where a
            point falls outside the surface.
        """
        logger.debug(f"Calculating z values for x, y (size = {x.size}).")
        pts = np.stack((x.flatten(), y.flatten()), axis=-1).astype(
            np.float32, copy=False
        )

        z = self.inner.query_many(pts)
        logger.debug("Query complete.")
        return z.reshape(x.shape).astype(x.dtype)

    def __getstate__(self):
        # When standard pickle hits this object, bypass pickling the Rust object
        state = self.__dict__.copy()

        state["inner"] = registry.pickle_pass(self.inner)
        return state

    def __setstate__(self, state):
        # When unpickling, swap the key back for the live object reference
        self.__dict__.update(state)
        key = state["inner"]
        self.inner = registry.REGISTRY[key]

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
