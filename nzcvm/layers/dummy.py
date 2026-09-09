"""Reusable utility layers for testing and scientific isolation.

The :func:`~nzcvm.layers.functional.functional_layer` decorator defines
:class:`constant`, which suits both test code and scientific pipelines that
want a spatially uniform background model, such as when isolating source
effects from path effects.

:class:`CountingLayer` and :class:`RecordingLayer` are stateful wrappers that
track how many times they're called and what arguments they received.  Test
code is the place for them.

Example
-------
::

    from nzcvm.layers.dummy import constant, CountingLayer
    from nzcvm.layers.clamp import ClampLayer
    from nzcvm.config.layers.clamp import ClampLayerConfig, Bound

    terminal = constant(vs=3500.0)
    counter  = CountingLayer(geometry, terminal)
    clamp    = ClampLayer(ClampLayerConfig(clamps={"vs": Bound(min=4000.0)}), geometry, counter)

    qualities = clamp(grid)
    assert counter.call_count == 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from shapely import Geometry

from nzcvm.config.layers.core import LayerConfig
from nzcvm.grids.grid import Grid
from nzcvm.layers.core import Layer
from nzcvm.layers.functional import functional_layer
from nzcvm.qualities import Qualities, QualitiesSchema
from nzcvm.query import ModelRange

# ---------------------------------------------------------------------------
# Functional layers
# ---------------------------------------------------------------------------


@functional_layer
def constant(
    grid: Grid,
    model_range: ModelRange = ModelRange.ALL,
    *,
    next_layer: Layer | None = None,
    rho: float = 2700.0,
    vp: float = 6000.0,
    vs: float = 3500.0,
    qp: float = 200.0,
    qs: float = 100.0,
    alpha: float = 1.0,
) -> Qualities:
    """Return spatially uniform constant qualities regardless of the grid.

    Useful as a terminal layer in test pipelines or as a flat background model
    in scientific experiments where one wants to isolate source effects.

    Parameters
    ----------
    rho, vp, vs, qp, qs, alpha :
        Uniform component values broadcast to the full grid shape.
    next_layer :
        Ignored, because a terminal layer never delegates downstream.
    """
    shape = grid.x.shape
    ones = np.ones(shape, dtype=np.float32)
    return QualitiesSchema.new(
        rho=ones * rho,
        vp=ones * vp,
        vs=ones * vs,
        qp=ones * qp,
        qs=ones * qs,
        alpha=ones * alpha,
    )


# ---------------------------------------------------------------------------
# Stateful wrappers (inherently class-based due to mutable state)
# ---------------------------------------------------------------------------


@dataclass
class _NullConfig(LayerConfig):
    type: Literal["_null"] = "_null"  # type: ignore[assignment]


class CountingLayer(Layer[_NullConfig]):
    """Transparent wrapper that counts calls into the layer.

    Parameters
    ----------
    next_layer :
        The downstream layer to delegate to.
    """

    def __init__(self, geometry: Geometry, next_layer: Layer) -> None:
        super().__init__(_NullConfig(), geometry, next_layer)
        self.call_count = 0

    def __call__(
        self, grid: Grid, model_range: ModelRange = ModelRange.ALL
    ) -> Qualities:
        self.call_count += 1
        return self.next_layer(grid, model_range=model_range)


class RecordingLayer(Layer[_NullConfig]):
    """Transparent wrapper that records every ``(grid, model_range)`` call.

    Parameters
    ----------
    next_layer :
        The downstream layer to delegate to.
    """

    def __init__(self, geometry: Geometry, next_layer: Layer) -> None:
        super().__init__(_NullConfig(), geometry, next_layer)
        self.calls: list[tuple[Grid, ModelRange]] = []

    def __call__(
        self, grid: Grid, model_range: ModelRange = ModelRange.ALL
    ) -> Qualities:
        self.calls.append((grid, model_range))
        return self.next_layer(grid, model_range=model_range)


# Keep a class alias so older code naming ConstantLayer still works.
ConstantLayer = constant
