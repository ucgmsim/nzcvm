"""Pipeline layer for applying the Ely et al. (2010) GTL taper."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from shapely import Geometry

from nzcvm import qualities
from nzcvm.config.layers.ely import ElyLayerConfig
from nzcvm.coordinates import Coordinate
from nzcvm.ely_taper import ely_vs_profile
from nzcvm.grids.grid import grid_like_at_depth
from nzcvm.layers.core import Layer
from nzcvm.models.surface import Surface
from nzcvm.query import ModelRange

if TYPE_CHECKING:
    from nzcvm.grids.grid import Grid
    from nzcvm.qualities import Qualities

logger = logging.getLogger(__name__)


class ElyLayer(Layer[ElyLayerConfig], config_cls=ElyLayerConfig):
    def __init__(
        self, config: ElyLayerConfig, geometry: Geometry, next_layer: Layer
    ) -> None:
        super().__init__(config, geometry, next_layer)
        self.interpolator = Surface.load(config.vs30)

    def __call__(
        self,
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        """Apply the Ely GTL taper to *grid* and return the result.

        Parameters
        ----------
        grid :
            Grid chunk to evaluate.
        model_range :
            Priority range for velocity-model queries.
        """
        logger.debug("Beginning Ely Taper with model_range=%s", model_range)

        # Fast path: basins-only queries skip Ely entirely.
        if model_range == ModelRange.BASINS:
            return self.next_layer(grid, model_range=model_range)

        depth_t = self.config.depth_t
        is_in_taper = (grid.depth < depth_t) & (grid[Coordinate.COASTLINE] <= 1e-6)
        # If the whole chunk is below the taper, skip Ely entirely.
        if not is_in_taper.any():
            logger.debug("Chunk outside taper, skipping Ely taper calculation.")
            return self.next_layer(grid, model_range=model_range)

        basins = None

        if model_range == ModelRange.TOMOGRAPHY:
            in_basin = xr.full_like(is_in_taper, False)
        else:
            surface_layer = grid_like_at_depth(grid, 0.0)
            basin_at_surface = self.next_layer(
                surface_layer, model_range=ModelRange.BASINS
            ).squeeze()
            in_basin = xr.apply_ufunc(np.isclose, basin_at_surface.alpha, 1.0)
            if in_basin.all():
                # Below the basin everywhere in the velocity model, skip the
                # Ely taper and defer every quality to the layer underneath,
                # preserving the impedance contrast between the basin and
                # tomography layers.
                logger.debug("Chunk inside basin, skipping Ely taper calculation.")
                return self.next_layer(grid, model_range=model_range)
            else:
                basins = self.next_layer(grid, model_range=ModelRange.BASINS)

        safe_depth = grid.depth.clip(max=depth_t)

        x_top = grid.x.isel({Coordinate.K: 0}).drop_vars(Coordinate.K.value)
        y_top = grid.y.isel({Coordinate.K: 0}).drop_vars(Coordinate.K.value)

        vs30 = xr.apply_ufunc(
            self.interpolator.transform,
            x_top,
            y_top,
            input_core_dims=[[], []],
            output_core_dims=[[]],
            output_dtypes=[np.float32],
        )
        non_nan_vs30 = ~np.isnan(vs30)
        is_in_taper = non_nan_vs30 & ~in_basin & is_in_taper

        # Select a z-layer of the block.
        # The array [0] as the selection is important because it preserves the k
        # axis for downstream layers.
        reference_rock_layer = grid_like_at_depth(grid, depth_t)

        # Calculate bounding taper qualities using *only* the tomography.
        # Calling squeeze here drops the phony K dimension kept around to
        # calculate the qualities at the surface layer.
        logger.debug("Calculating taper qualities")
        taper_qualities = self.next_layer(
            reference_rock_layer, model_range=ModelRange.TOMOGRAPHY
        ).squeeze()

        ely_qualities = ely_vs_profile(
            safe_depth,
            vs30,
            taper_qualities.vp,
            taper_qualities.vs,
            depth_t=depth_t,
        )

        # Get background for all points in this chunk (becomes the out buffer).
        background = self.next_layer(grid, model_range=ModelRange.TOMOGRAPHY)

        if basins is not None:
            qualities.blend(basins, background, out=background)
            qualities.blend(
                ely_qualities, background, out=background, where=is_in_taper.values
            )
        else:
            qualities.blend(
                ely_qualities, background, out=background, where=is_in_taper.values
            )

        return background
