"""Surface interpolation for topography-based depth transforms.

A :class:`Surface` wraps a surface mesh and provides point-query
interpolation, used to convert depth-below-surface coordinates into
absolute elevations.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.models.mesh import StructuredMesh, StructuredMeshSchema, triangulate
from nzcvm.models.reconstruct import Reconstructable, cached
from nzcvm.nzcvm import PySurfaceModel, surface_model  # ty: ignore[unresolved-import]

DEFAULT_TOLERANCE = 1e-4

logger = logging.getLogger(__name__)


@dataclass
class Surface(Reconstructable):
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

        surface = cls(inner, bounds=bounds, n_points=len(points))
        return surface.built_by(cls.from_dataset, mesh)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a surface mesh from *path* and return a :class:`Surface`.

        Repeated loads of the same path return the same object: a worker
        process rebuilding the surface after unpickling pays for the read and
        the triangulation once rather than once per task.

        Parameters
        ----------
        path :
            Path to the surface mesh file.

        Returns
        -------
        Surface
            The loaded surface, ready to interpolate.
        """
        return _load(Path(path))  # ty: ignore[invalid-return-type]

    def transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Interpolate surface elevation at query (x, y) locations.

        Parameters
        ----------
        x, y :
            Query point coordinates in the same projected CRS as the mesh.

        Returns
        -------
        numpy.ndarray
            Elevation (z) values with the same shape as *x*.
        """
        logger.debug(f"Calculating z values for x, y (size = {x.size}).")
        pts = np.stack((x.flatten(), y.flatten()), axis=-1)

        z = self.inner.query_many(pts)
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


@cached
def _load(path: Path) -> Surface:
    """Read and index the surface at *path*."""
    with xr.open_dataset(path) as dset:
        mesh = StructuredMeshSchema.from_dataset(dset)
    # Recorded against the path rather than the mesh `from_dataset` recorded,
    # so pickling sends a filename instead of the whole surface.
    return Surface.from_dataset(mesh).built_by(Surface.load, path)
