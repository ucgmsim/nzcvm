"""Pipeline layers for the NZCVM query pipeline.

Each layer receives a :class:`~nzcvm.grids.grid.Grid` chunk, applies some
transformation or query, and delegates to the next layer in the chain.
:func:`~nzcvm.layers.pipeline.build_pipeline` composes layers into a pipeline.

Available layers
----------------
:class:`~nzcvm.layers.query.QueryLayer`
    Queries a :class:`~nzcvm.models.model.ModelTree` and fills each grid point
    with seismic material properties.
:class:`~nzcvm.layers.ely.ElyLayer`
    Applies the Ely et al. (2010) near-surface velocity taper.
:class:`~nzcvm.layers.clamp.ClampLayer`
    Clamps seismic component values to per-component min/max bounds.
:class:`~nzcvm.layers.offshore.OffshoreBasinLayer`
    Fills the near-surface velocity model in offshore and coastal regions.
:class:`~nzcvm.layers.coastline.CoastlineLayer`
    Computes signed distance to the coastline and attaches it as a grid coordinate.
"""

from nzcvm.layers import backus, clamp, coastline, ely, offshore, query

__all__ = ["backus", "clamp", "coastline", "ely", "offshore", "query"]
