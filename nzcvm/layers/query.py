"""Pipeline layer that queries a :class:`~nzcvm.models.model.ModelTree`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import shapely
import xarray as xr
from shapely import Geometry

from nzcvm.config.layers.query import QueryLayerConfig
from nzcvm.layers.core import Layer
from nzcvm.models.model import ModelTree
from nzcvm.query import ModelRange

if TYPE_CHECKING:
    from nzcvm.grids.grid import Grid
    from nzcvm.qualities import Qualities

logger = logging.getLogger(__name__)


def _model_intersects_geometry(model_path: Path, geometry: Geometry) -> bool:
    with xr.open_dataset(model_path) as model:
        model_geometry = model.attrs.get("geometry")
        if model_geometry is None:
            # No geometry recorded (say, a whole-domain tomography model) means
            # the model has no bounds to prune against, so always load it.
            return True
        return shapely.intersects(shapely.from_wkb(model_geometry), geometry)


class QueryLayer(Layer[QueryLayerConfig], config_cls=QueryLayerConfig):
    def __init__(
        self, config: QueryLayerConfig, geometry: Geometry, next_layer: Layer
    ) -> None:
        super().__init__(config, geometry, next_layer)
        models = sorted(
            {
                p
                for glob in config.model_globs
                for p in config.model_path.rglob(glob)
                if _model_intersects_geometry(p, geometry)
            }
        )
        logger.info(
            f"Loading {len(models)} models ({', '.join(str(m) for m in sorted(models))}) for querying"
        )
        self.model = ModelTree.load_models(models)

    def __call__(
        self,
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        """Query the velocity model at every grid point and return the results.

        Parameters
        ----------
        grid :
            Grid chunk with spatial variables ``x``, ``y``, ``z``.
        model_range :
            Priority range used for the query.
        """

        logger.debug("Beginning query layer query with model_range=%s", model_range)
        qualities = self.model.query_many(
            grid.x, grid.y, grid.z, model_range=model_range
        )
        logger.debug("Query complete")
        return qualities
