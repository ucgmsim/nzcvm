"""Abstract base class and registry for pipeline layers."""

# Lets Layer[Any] appear without the string quotes, which look ugly.
from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from shapely import Geometry

from nzcvm.config.layers import LayerConfig
from nzcvm.query import ModelRange

if TYPE_CHECKING:
    from nzcvm.grids.grid import Grid
    from nzcvm.qualities import Qualities


class Layer[C: LayerConfig](ABC):
    """Abstract base class for one stage in the model-query pipeline.

    A layer receives a :class:`~nzcvm.grids.grid.Grid` chunk, computes or
    transforms :class:`~nzcvm.qualities.Qualities`, and returns the result.
    Each concrete layer holds a reference to the next layer in the chain and
    calls it to obtain background or downstream results.

    See Also
    --------
    nzcvm.layers.query.QueryLayer : Queries a velocity model.
    nzcvm.layers.functional.functional_layer : Derive a layer from a plain function.
    """

    registry: ClassVar[dict[type[LayerConfig], type[Layer]]] = {}
    config: C

    def __init__(self, config: C, geometry: Geometry, next_layer: Layer[Any]) -> None:
        self.config = config
        self.geometry = geometry
        self.next_layer = next_layer

    def __init_subclass__(cls, config_cls: type[C] | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if config_cls:
            value = typing.cast(type[Layer], cls)
            Layer.registry[config_cls] = value

    @abstractmethod
    def __call__(
        self,
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        """Apply this layer to *grid* and return the result.

        Parameters
        ----------
        grid :
            Grid chunk to evaluate.
        model_range :
            Priority range for velocity-model queries
            (:class:`~nzcvm.query.ModelRange`).
        """
        ...


def layer_from_config(config: LayerConfig) -> type[Layer]:
    """Look up the :class:`Layer` subclass registered for *config*'s type.

    Raises
    ------
    KeyError
        If the registry doesn't have a layer for the given config type.
    """
    config_type = type(config)
    if config_type not in Layer.registry:
        raise KeyError(f"Could not find matching layer for {type(config)!r}")

    return Layer.registry[config_type]
