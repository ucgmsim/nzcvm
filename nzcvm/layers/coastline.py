from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import shapely
import shapely.ops
import xarray as xr
from numba import guvectorize
from shapely import Geometry

from nzcvm.config.layers.coastline import CoastlineConfig
from nzcvm.coordinates import Coordinate
from nzcvm.layers.core import Layer
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


@guvectorize(
    [
        "void(float32[:], float32[:], int64[:], int64[:], float32[:,:,:], b1[:])",
    ],
    "(n), (n), (m), (m), (s, d, c) -> (n)",
    nopython=True,
    fastmath=True,
)
def _ray_cast_intersections(
    x: np.ndarray,
    y: np.ndarray,
    ray_indices: np.ndarray,
    segment_indices: np.ndarray,
    segments: np.ndarray,
    out: np.ndarray,
):
    # Initialise buffer to all zeros because guvectorize initialises out with
    # something equivalent to np.empty.
    out[:] = 0

    for i in range(len(ray_indices)):
        # Roughly the algorithm is grab the possibly intersecting segment, find
        # the x-value of the intersection point with a horizontal ray (extending
        # in both the + & - directions from the test point). If that x-value is
        # greater than px then the +x ray cast from (px, py) intersects the
        # boundary and counts toward the intersection parity.
        p_idx = ray_indices[i]
        s_idx = segment_indices[i]

        px = x[p_idx]
        py = y[p_idx]

        Ax = segments[s_idx, 0, 0]
        Ay = segments[s_idx, 0, 1]
        Bx = segments[s_idx, 1, 0]
        By = segments[s_idx, 1, 1]

        # This check is another bounding box check in the y-direction only,
        # which sounds redundant given the earlier bounding box intersections
        # with STRTree. It matters anyway. Suppose that the ray intersects at a
        # vertex of a polygon, then the ray would count as intersecting twice
        # -- once on each incident segment. That would put a point inside the
        # polygon erroneously *outside* it, so this check acts as a tie
        # breaker. In the event that a ray cast passes through a polygon
        # vertex, the tie breaker counts only one of the two incident segments.
        if min(Ay, By) <= py < max(Ay, By):
            x_intersect = Ax + (py - Ay) * (Bx - Ax) / (By - Ay)

            if x_intersect > px:
                out[p_idx] = not out[p_idx]


class CoastlineLayer(Layer[CoastlineConfig], config_cls=CoastlineConfig):
    def __init__(
        self, config: CoastlineConfig, geometry: Geometry, next_layer: Layer
    ) -> None:
        super().__init__(config, geometry, next_layer)
        coastline = shapely.ops.orient(
            _read_compressed_shapely_wkb(config.coastline), sign=1.0
        )

        self.segments = _extract_segments(coastline).astype(np.float32)
        # TODO: Replace with geo-index to remove overhead creating points in the _compute_chunk_dist
        # Blocked on https://github.com/georust/geo-index/issues/150.

        lines = [shapely.geometry.LineString(seg) for seg in self.segments]
        self.tree = shapely.STRtree(lines)
        self.poly_max_x = (
            coastline.bounds[2] + 10.0
        )  # Add a buffer to ensure it exits the polygon

    def _distance(self, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:

        def _compute_chunk_dist(x_chunk, y_chunk):
            orig_shape = x_chunk.shape
            x_flat = x_chunk.ravel()
            y_flat = y_chunk.ravel()

            pts = shapely.points(x_flat, y_flat)

            _, distance = self.tree.query_nearest(
                pts, return_distance=True, all_matches=False
            )
            distance = distance.astype(np.float32)

            # That returns the distance to the nearest segment, but its
            # sign (in or out) doesn't follow from distance alone: the test
            # needs to know whether a point lies inside the geometry. Ray
            # casting answers that. If the cast ray (arbitrary, so a +x ray)
            # crosses the geometry an odd number of times, it lies inside.
            # Making that fast takes two techniques:
            #
            # 1. Reusing the STRTree to query all *possibly* intersecting
            # polygon segments by ray. Possibly intersecting is per the
            # definition of `STRTree.query`: a segment is possibly intersecting
            # a ray if the bounding box of the ray intersects the bounding box
            # of the segment. This necessarily returns more segments than
            # actually intersect.
            ray_ends_x = np.full_like(x_flat, self.poly_max_x)
            ray_coords = np.column_stack((x_flat, y_flat, ray_ends_x, y_flat)).reshape(
                -1, 2, 2
            )
            rays = shapely.linestrings(ray_coords)
            ray_idx, tree_idx = self.tree.query(rays)

            # 2. A tight numba kernel iterates over the possible segments
            # returned from (1) and performs an exact intersection test. It also
            # counts the parity of the intersections and sets a mask is_inside
            # to 1 where the points are inside the geometry.
            is_inside = _ray_cast_intersections(
                x_flat, y_flat, ray_idx, tree_idx, self.segments
            )

            distance[is_inside] *= -1

            return distance.reshape(orig_shape)

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
