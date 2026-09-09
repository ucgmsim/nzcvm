"""Pipeline layer for applying the offshore taper.

The offshore layer fills near-surface velocities in ocean regions and at the
coast-to-ocean transition.  It is the seaward counterpart of the Ely GTL
layer: rather than using a Vs30 surface, it interpolates a 1-D depth–velocity
profile parametrised by horizontal distance from the coastline.

Requires the ``coastline`` coordinate to be present on the grid, which is
provided by :class:`~nzcvm.layers.coastline.CoastlineLayer`.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
import xarray as xr
from shapely import Geometry

from nzcvm import qualities
from nzcvm.components import Component
from nzcvm.config.layers.offshore import (
    DepthModel,
    OffshoreBasinConfig,
    VelocityModel1D,
)
from nzcvm.coordinates import Coordinate
from nzcvm.grids.grid import grid_like_at_depth
from nzcvm.layers.core import Layer
from nzcvm.qualities import QualitiesSchema
from nzcvm.query import ModelRange

if TYPE_CHECKING:
    from nzcvm.grids.grid import Grid
    from nzcvm.qualities import Qualities

logger = logging.getLogger(__name__)


# TODO: An efficiency can be made by observing that exact distances don't have
# to be calculated for segments > max distance from shoreline. Come back and
# clean this up with a BVHTree implementation from the rust side.


def step_interpolator(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    idx = np.searchsorted(xp, x, side="right") - 1
    idx = np.clip(idx, 0, len(xp) - 1)
    return fp[idx]


def _extract_qualities(layers: list[VelocityModel1D]) -> np.ndarray:
    qualities = []

    for layer in layers:
        layer_dict = layer.to_dict()
        qualities.append([layer_dict[component] for component in Component])

    return np.array(qualities).astype(np.float32)


def _build_model_interpolator(
    model: list[VelocityModel1D], absolute_bottom: float
) -> tuple[np.ndarray, np.ndarray]:
    layers = sorted(model, key=lambda model: model.bottom_depth)
    # Trim excess layers from velocity model
    end_idx = max(
        i for i in range(len(layers)) if layers[i].bottom_depth < absolute_bottom
    )
    layers = layers[: end_idx + 1]

    top_depths = [0.0]
    top_depths.extend(layer.bottom_depth for layer in layers[:-1])
    top_depths: np.ndarray = np.array(top_depths).astype(np.float32)

    model_qualities = _extract_qualities(layers)

    return top_depths, model_qualities


@dataclass(frozen=True)
class OffshoreModel:
    """Encapsulates the geometry, segments, and spatial index for distance calculations."""

    distances: np.ndarray
    depths: np.ndarray
    absolute_bottom: float

    model_top_depths: np.ndarray
    model_qualities: np.ndarray

    @classmethod
    def build(cls, basin_depth: list[DepthModel], model: list[VelocityModel1D]) -> Self:
        distance_layers = sorted(basin_depth, key=lambda layer: layer.distance)
        distances = np.array([layer.distance for layer in distance_layers]).astype(
            np.float32
        )
        depths = np.array([layer.bottom_depth for layer in distance_layers]).astype(
            np.float32
        )
        absolute_bottom = depths.max()
        model_top_depths, model_qualities = _build_model_interpolator(
            model, absolute_bottom
        )
        return cls(
            distances, depths, absolute_bottom, model_top_depths, model_qualities
        )

    def depth(self, distance: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            functools.partial(np.interp, xp=self.distances, fp=self.depths),
            distance,
            input_core_dims=[[]],
            output_core_dims=[[]],
            output_dtypes=[distance.dtype],
        )

    def qualities(self, depths: xr.DataArray) -> Qualities:
        darr = xr.apply_ufunc(
            step_interpolator,
            depths,
            input_core_dims=[[]],
            output_core_dims=[["component"]],
            kwargs={"xp": self.model_top_depths, "fp": self.model_qualities},
            output_sizes={"component": len(list(Component))},
            output_dtypes=[depths.dtype],
        )
        dset = darr.assign_coords(component=list(Component)).to_dataset(dim="component")

        return QualitiesSchema.from_dataset(dset)


class OffshoreBasinLayer(Layer[OffshoreBasinConfig], config_cls=OffshoreBasinConfig):
    def __init__(
        self, config: OffshoreBasinConfig, geometry: Geometry, next_layer: Layer
    ):
        super().__init__(config, geometry, next_layer)
        self.model = OffshoreModel.build(config.basin_depth, config.model)

    def __call__(
        self,
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        """Apply the offshore taper to *grid* and return the result.

        Parameters
        ----------
        grid :
            Grid chunk to evaluate.  Must have the ``coastline`` coordinate
            set by :class:`~nzcvm.layers.coastline.CoastlineLayer`.
        model_range :
            Priority range for velocity-model queries.
        """
        if model_range == ModelRange.TOMOGRAPHY:
            return self.next_layer(grid, model_range=model_range)

        is_above_model_bottom_depth = grid.depth < self.model.absolute_bottom
        if not is_above_model_bottom_depth.any():
            logger.debug("Chunk below maximum basin depth, skipping calculation.")
            return self.next_layer(grid, model_range=model_range)

        logger.debug("Calculating offshore distances")
        offshore_distance = grid[Coordinate.COASTLINE]
        logger.debug("Offshore distances calculated")
        is_offshore = offshore_distance > 0

        basin_depth = self.model.depth(offshore_distance)

        if not is_offshore.any():
            logger.debug("Chunk entirely within onshore region, skipping calculation.")
            return self.next_layer(grid, model_range=model_range)

        is_above_basin = grid.depth < basin_depth

        if not is_above_basin.any():
            logger.debug("Chunk below basin surface, skipping calculation.")
            return self.next_layer(grid, model_range=model_range)

        surface_layer = grid_like_at_depth(grid, 0.0)
        footprint = (
            self.next_layer(surface_layer, model_range=ModelRange.BASINS)
            .squeeze()
            .alpha
        )

        if np.allclose(footprint, 1.0):
            logger.debug(
                "Chunk inside basin at surface, skipping offshore calculation."
            )
            return self.next_layer(grid, model_range=model_range)

        basins = self.next_layer(grid, model_range=ModelRange.BASINS)

        if np.allclose(basins.alpha, 1.0):
            logger.debug("Chunk inside modelled basin, skipping offshore calculation.")
            return basins

        background = self.next_layer(grid, model_range=model_range)
        offshore_qualities = self.model.qualities(grid.depth)

        # How much basin was present at the surface but has faded out at this
        # depth: 1 below a basin's base inside its footprint, 0 above the base
        # and outside the footprint, and a ramp across the smoothing boundary.
        # Used to cross-fade the basin with the offshore basin.
        faded = (footprint - basins.alpha).clip(0.0, 1.0)
        offshore_qualities[Component.ALPHA] = offshore_qualities.alpha * (1.0 - faded)

        # Composite offshore over the background, then basins over the lot, only
        # where the offshore body exists. The background already renders basins
        # onto tomography, but wherever the basin is present the offshore is
        # either occluded by it or opaque itself, so re-compositing the basin on
        # top will not double count.
        mask = (is_above_basin & is_offshore).values
        qualities.blend(offshore_qualities, background, out=background, where=mask)
        qualities.blend(basins, background, out=background, where=mask)

        return background
