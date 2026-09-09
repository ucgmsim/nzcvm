from __future__ import annotations

import dataclasses
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import xarray as xr
from shapely import Geometry

from nzcvm.config.layers.core import LayerConfig
from nzcvm.grids.grid import Grid
from nzcvm.layers.core import Layer, layer_from_config
from nzcvm.qualities import QualitiesSchema, template_like
from nzcvm.query import ModelRange
from nzcvm.velocity_model import VelocityModel

if TYPE_CHECKING:
    from nzcvm.qualities import Qualities


class PipelineError(Exception):
    pass


@dataclass
class _SentinelConfig(LayerConfig):
    type: Literal["_sentinel"] = "_sentinel"


class _SentinelLayer(Layer[_SentinelConfig]):
    """Terminal sentinel that raises when the grid falls outside all layers."""

    def __init__(self) -> None:
        super().__init__(_SentinelConfig(), None, None)  # ty: ignore[invalid-argument-type]

    def __call__(
        self, grid: Grid, model_range: ModelRange = ModelRange.ALL
    ) -> Qualities:
        e = ValueError("Grid out of bounds of any layer")
        e.add_note(str(grid))
        raise e


def build_pipeline(geometry: Geometry, configs: list[LayerConfig]) -> Layer:
    if not configs:
        raise ValueError("Pipeline configuration list cannot be empty.")

    pipeline: Layer = _SentinelLayer()

    for config in reversed(configs):
        layer_type = layer_from_config(config)
        pipeline = layer_type(config, geometry, pipeline)

    return pipeline


def execute_model_pipeline(
    velocity_model: VelocityModel, pipeline: Callable[[Grid], Qualities]
) -> VelocityModel:
    """Apply *pipeline* to every grid in *velocity_model* via one
    ``map_blocks`` per grid.

    Hoisting the chunked dispatch here means layers never need to call
    ``map_blocks`` or ``apply_ufunc(..., dask="parallelized")`` internally;
    each layer always receives a fully computed concrete chunk and can use
    plain NumPy operations without creating extra Dask tasks.
    """

    def _run_pipeline(chunk: xr.Dataset) -> xr.Dataset:
        grid = typing.cast(Grid, chunk)
        qualities = pipeline(grid)
        return qualities

    qualities = {
        name: QualitiesSchema.from_dataset(
            grid.map_blocks(
                _run_pipeline,
                template=template_like(grid.x),
            )
        )
        for name, grid in velocity_model.grids.items()
    }
    return dataclasses.replace(velocity_model, qualities=qualities)
