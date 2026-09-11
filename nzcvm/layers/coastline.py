from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import shapely
import shapely.ops
import xarray as xr
from shapely import Geometry

from nzcvm.config.layers.coastline import CoastlineConfig
from nzcvm.coordinates import Coordinate
from nzcvm.layers.core import Layer
from nzcvm.models.reconstruct import Reconstructable, cached
from nzcvm.nzcvm import coastline as build_coastline  # ty: ignore[unresolved-import]
from nzcvm.query import ModelRange

if TYPE_CHECKING:
    from nzcvm.grids.grid import Grid
    from nzcvm.qualities import Qualities


logger = logging.getLogger(__name__)


def _read_compressed_shapely_wkb(path: Path) -> shapely.Geometry:
    with gzip.open(path) as handle:
        return shapely.from_wkb(handle.read())


def _extract_segments(geometry: shapely.Geometry) -> np.ndarray:
    boundary = geometry.boundary
    lines = boundary.geoms if hasattr(boundary, "geoms") else [boundary]

    extracted_segments = []
    for line in lines:
        coords = np.array(line.coords)
        for i in range(len(coords) - 1):
            extracted_segments.append([coords[i], coords[i + 1]])

    return np.array(extracted_segments)


@dataclass
class Coastline(Reconstructable):
    """A coastline indexed for signed-distance queries.

    Wraps the Rust index so that a layer can cross a process boundary with
    one attached. The index itself has no Python representation, but the file
    it came from does.
    """

    inner: Any

    @classmethod
    def load(cls, path: Path) -> Self:
        """Read and index the coastline polygon stored at *path*.

        Repeated loads of the same path return the same object, so a worker
        process rebuilding it after unpickling indexes once rather than once
        per task.

        Parameters
        ----------
        path :
            Path to a gzipped WKB polygon in the projected CRS.

        Returns
        -------
        Coastline
            The indexed coastline.
        """
        return _load(Path(path))  # ty: ignore[invalid-return-type]

    def signed_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Distance from each point to the coastline, negative onshore."""
        return self.inner.signed_distance(_borrowable(x), _borrowable(y))

    def __len__(self) -> int:
        return len(self.inner)


def _borrowable(array: np.ndarray) -> np.ndarray:
    """Coerce *array* into something the Rust side can borrow as a slice.

    It needs float32, contiguous *and* aligned.  `ascontiguousarray` promises
    only the first two: a buffer that arrived over the wire can start at an
    offset that leaves the array contiguous and misaligned, and it would pass
    through untouched and then fail in the binding.
    """
    return np.require(array, dtype=np.float32, requirements=["C", "A"])


@cached
def _load(path: Path) -> Coastline:
    polygon = shapely.ops.orient(_read_compressed_shapely_wkb(path), sign=1.0)
    segments = _extract_segments(polygon).astype(np.float32)
    return Coastline(build_coastline(segments)).built_by(Coastline.load, path)


class CoastlineLayer(Layer[CoastlineConfig], config_cls=CoastlineConfig):
    def __init__(
        self, config: CoastlineConfig, geometry: Geometry, next_layer: Layer
    ) -> None:
        super().__init__(config, geometry, next_layer)
        self.coastline = Coastline.load(config.coastline)
        logger.debug("Indexed %d coastline segments", len(self.coastline))

    def _distance(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:

        def _compute_chunk_dist(x_chunk, y_chunk):
            distance = self.coastline.signed_distance(x_chunk.ravel(), y_chunk.ravel())
            return distance.reshape(x_chunk.shape)

        return xr.apply_ufunc(
            _compute_chunk_dist,
            x,
            y,
            input_core_dims=[[], []],
            output_core_dims=[[]],
            output_dtypes=[x.dtype],
        )

    def __call__(
        self, grid: Grid, model_range: ModelRange = ModelRange.ALL
    ) -> Qualities:
        grid[Coordinate.COASTLINE] = self._distance(
            grid.x.isel({Coordinate.K: 0}), grid.y.isel({Coordinate.K: 0})
        )
        return self.next_layer(grid, model_range)
