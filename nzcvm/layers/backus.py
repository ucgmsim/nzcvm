"""Pipeline layer for Backus averaging the next layer over each node's span."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from shapely import Geometry

from nzcvm.config.layers.backus import BackusAveragedLayerConfig
from nzcvm.layers.core import Layer
from nzcvm.qualities import QualitiesSchema
from nzcvm.query import ModelRange

if TYPE_CHECKING:
    from nzcvm.grids.grid import Grid
    from nzcvm.qualities import Qualities

logger = logging.getLogger(__name__)


class BackusAveragedLayer(
    Layer[BackusAveragedLayerConfig], config_cls=BackusAveragedLayerConfig
):
    """Backus average the next layer over the vertical span of each node.

    Each node's span runs half-way to its neighbours in ``z``. On the top
    grid the free surface truncates the first span; on a lower grid the
    interface mirrors it. The layer samples the span ``samples`` times and
    combines the samples, weighted by their ``alpha``: ``rho`` is the
    arithmetic mean, ``vp`` and ``vs`` the Backus average
    ``sqrt(1 / (<1 / (rho v^2)> <rho>))``, ``qp`` and ``qs`` the
    ``1 / v``-weighted harmonic mean ``<1 / v> / <1 / (v Q)>``, and ``alpha``
    the fraction of covered samples.
    """

    def __init__(
        self, config: BackusAveragedLayerConfig, geometry: Geometry, next_layer: Layer
    ) -> None:
        super().__init__(config, geometry, next_layer)

    def __call__(
        self,
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        """Apply the Backus averaging layer to *grid* and return the result.

        Parameters
        ----------
        grid :
            Grid chunk to evaluate.
        model_range :
            Priority range for velocity-model queries.
        """
        z = grid.z
        free_surface_z = (grid.z - grid.depth).isel(k=0)

        upper = 0.5 * (z.shift(k=1) + z)
        lower = 0.5 * (z + z.shift(k=-1))
        lower = lower.where(z.k != z.k[-1], 2 * z - upper)
        is_top_grid = bool(z.k[0] == 0)
        if is_top_grid:
            upper = upper.where(z.k != z.k[0], z.isel(k=0))  # truncate at free surface
        else:
            upper = upper.where(z.k != z.k[0], 2 * z - lower)  # mirror at interface
        span = lower - upper
        n = self.config.samples

        # alpha-weighted accumulators
        w = xr.zeros_like(z)
        a_rho = xr.zeros_like(z)
        a_invm = xr.zeros_like(z)
        a_invmu = xr.zeros_like(z)
        a_svp = xr.zeros_like(z)
        a_svpq = xr.zeros_like(z)
        a_svs = xr.zeros_like(z)
        a_svsq = xr.zeros_like(z)

        for m in range(n):
            sample_z = upper + ((m + 0.5) / n) * span
            sample_depth = sample_z - free_surface_z
            grid_sample = grid.assign(z=sample_z, depth=sample_depth)
            q = self.next_layer(grid_sample, model_range)
            a = q.alpha
            w += a
            a_rho += a * q.rho
            a_invm += a * np.reciprocal(q.rho * np.square(q.vp))
            a_invmu += a * np.reciprocal(q.rho * np.square(q.vs))
            inv_vp = np.reciprocal(q.vp)
            a_svp += a * inv_vp
            a_svpq += a * inv_vp * np.reciprocal(q.qp)
            inv_vs = np.reciprocal(q.vs)
            a_svs += a * inv_vs
            a_svsq += a * inv_vs * np.reciprocal(q.qs)

        covered = w > 0
        w_safe = w.where(covered, 1.0)

        alpha = w / n
        rho = xr.where(covered, a_rho / w_safe, 0.0)
        vp = xr.where(covered, w / np.sqrt((a_invm * a_rho).where(covered, 1.0)), 0.0)
        vs = xr.where(covered, w / np.sqrt((a_invmu * a_rho).where(covered, 1.0)), 0.0)
        qp = xr.where(covered, a_svp / a_svpq.where(covered, 1.0), 0.0)
        qs = xr.where(covered, a_svs / a_svsq.where(covered, 1.0), 0.0)

        return QualitiesSchema.new(rho=rho, vp=vp, vs=vs, qp=qp, qs=qs, alpha=alpha)
