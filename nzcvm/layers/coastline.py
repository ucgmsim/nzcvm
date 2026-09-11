from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import shapely
import shapely.ops
import xarray as xr
from shapely import Geometry

from nzcvm.config.layers.coastline import CoastlineConfig
from nzcvm.coordinates import Coordinate
from nzcvm.layers.core import Layer
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


class CoastlineLayer(Layer[CoastlineConfig], config_cls=CoastlineConfig):
    def __init__(
        self, config: CoastlineConfig, geometry: Geometry, next_layer: Layer
    ) -> None:
        super().__init__(config, geometry, next_layer)
        coastline = shapely.ops.orient(
            _read_compressed_shapely_wkb(config.coastline), sign=1.0
        )
        segments = _extract_segments(coastline).astype(np.float32)
        self.coastline = build_coastline(segments)
        logger.debug("Indexed %d coastline segments", len(self.coastline))

    def _distance(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:

        def _compute_chunk_dist(x_chunk, y_chunk):
            # The Rust side takes contiguous float32. A grid holds float32
            # already, but a caller passing anything else should convert here
            # rather than hit a binding type error.
            distance = self.coastline.signed_distance(
                np.ascontiguousarray(x_chunk.ravel(), dtype=np.float32),
                np.ascontiguousarray(y_chunk.ravel(), dtype=np.float32),
            )
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
