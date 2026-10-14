"""Pipeline layer for enforcing physically consistent bounds on Components.

Vs acts as the *master* property.  Whenever a clamp moves Vs, the layer
regenerates Vp and density from the same Brocher (2005) / Nafe-Drake empirical
relations the rest of the pipeline uses (see :mod:`nzcvm.ely_taper`), keeping
the ``(vs, vp, rho)`` triple on a physically realisable manifold rather than
clipping each component independently into an inconsistent state.
"""

from __future__ import annotations

import graphlib
import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from shapely import Geometry

from nzcvm.components import Component
from nzcvm.config.layers.clamp import Bound, ClampLayerConfig
from nzcvm.ely_taper import DENSITY_RELATION, VP_FROM_VS_RELATION
from nzcvm.interpolation import step_interpolator
from nzcvm.layers.core import Layer
from nzcvm.query import ModelRange

if TYPE_CHECKING:
    from nzcvm.grids.grid import Grid
    from nzcvm.qualities import Qualities

logger = logging.getLogger(__name__)


def _raise_by_floors(
    floors: list[tuple[float, float]],
    minimum: float | xr.DataArray | None,
    key: xr.DataArray,
) -> float | xr.DataArray | None:
    """Raise *minimum* by the ``(threshold, min)`` *floors* over *key*.

    *key* grows downward and each floor holds at and beyond its threshold.
    A constant *minimum* seeds the step table, and the running maximum over
    floors sorted by threshold gives the tightest minimum at each step.
    :func:`step_interpolator` then picks the step for each point in one pass,
    however many floors there are.  A per-point *minimum* (a relative bound,
    or the floors of another key) can't seed the table and costs one extra
    pass to clip in.  The grid's coordinates stay off the result, so they
    don't leak onto the clamped component.
    """
    thresholds, minimums = np.array(sorted(floors), dtype=key.dtype).T
    per_point = isinstance(minimum, xr.DataArray)
    absolute_floor = -np.inf if minimum is None or per_point else minimum
    floor = xr.apply_ufunc(
        step_interpolator,
        key.reset_coords(drop=True),
        kwargs={
            "xp": np.r_[-np.inf, thresholds],
            "fp": np.maximum.accumulate(np.r_[absolute_floor, minimums]).astype(
                key.dtype
            ),
        },
        dask="parallelized",
        output_dtypes=[key.dtype],
    )
    return floor.clip(min=minimum) if per_point else floor


def _clip(bound: Bound, values: xr.DataArray, qualities: Qualities, grid: Grid):
    """Clip *values* to *bound*, with its minimum raised by its ``floors``.

    A depth floor compares against ``grid.depth`` and an elevation floor
    against ``grid.z``.  Both are positive down.
    """
    minimum = bound.resolve("min", qualities)
    by_depth = [(d, f.min) for f in bound.floors if (d := f.depth) is not None]
    by_z = [(e, f.min) for f in bound.floors if (e := f.elevation) is not None]
    for key, floors in ((grid.depth, by_depth), (grid.z, by_z)):
        if floors:
            minimum = _raise_by_floors(floors, minimum, key)
    return values.clip(min=minimum, max=bound.resolve("max", qualities))


class ClampLayer(Layer[ClampLayerConfig], config_cls=ClampLayerConfig):
    """Clamp seismic material properties, keeping Vs/Vp/density coherent."""

    def __init__(self, config: ClampLayerConfig, geometry: Geometry, next_layer: Layer):
        super().__init__(config, geometry, next_layer)
        self.config = config

    def __call__(
        self,
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        qualities = self.next_layer(grid, model_range=model_range)

        # Vs gets special handling. Where the clamp moves it, the layer updates other
        # properties match empirical relations based on the clamped Vs value.
        vs_bound = self.config.clamps.get(Component.VS)
        if vs_bound is not None:
            vs = qualities.vs
            clamped_vs = _clip(vs_bound, vs, qualities, grid)
            moved = clamped_vs != vs
            vp = xr.where(moved, VP_FROM_VS_RELATION(clamped_vs), qualities.vp)
            rho = xr.where(moved, DENSITY_RELATION(vp), qualities.rho)
            qualities[Component.VS] = clamped_vs
            qualities[Component.VP] = vp
            qualities[Component.RHO] = rho

        min_ratio = self.config.min_vp_vs_ratio
        max_ratio = self.config.max_vp_vs_ratio
        if min_ratio or max_ratio:
            vs = qualities.vs
            vp = qualities.vp
            guarded_vp = vp.clip(
                min=min_ratio * vs if min_ratio else None,
                max=max_ratio * vs if max_ratio else None,
            )
            qualities[Component.VP] = guarded_vp
            qualities[Component.RHO] = xr.where(
                guarded_vp != vp, DENSITY_RELATION(guarded_vp), qualities.rho
            )

        # Topologically sorting the clamps keeps every bound consistently set
        # at the point of update.
        for c, bound in self._topologically_sorted_clamps():
            if c == Component.VS:
                continue
            qualities[c] = _clip(bound, qualities[c], qualities, grid)

        return qualities

    def _topologically_sorted_clamps(self) -> list[tuple[Component, Bound]]:
        graph = {
            c: {ref for ref in [clamp.min_ref, clamp.max_ref] if ref}
            for c, clamp in self.config.clamps.items()
        }
        ts = graphlib.TopologicalSorter(graph)
        return [
            (c, self.config.clamps[c])
            for c in ts.static_order()
            if c in self.config.clamps
        ]
