"""SW4 curvilinear velocity model grid builder.

Builds the nested 2:1 refinement blocks defined by a
:class:`~nzcvm.config.grids.sw4.SW4GridConfig` on the shared hybrid
terrain-following coordinate in :mod:`nzcvm.grids.terrain`.

Without a decay SW4 meshes the topography itself and samples the sfile at
true positions, so the first refinement is a linear stretch and its columns
are evenly spaced, as SW4's sfile reader assumes.  With a decay SW4 runs
without topography on the flat nominal box, as EMOD3D does.

See Also
--------
nzcvm.config.grids.sw4.SW4GridConfig : Grid configuration.
nzcvm.grids.terrain : The shared terrain-following construction.
"""

import logging

import numpy as np
import shapely

from nzcvm.config.grids.sw4 import SW4GridConfig
from nzcvm.config.grids.terrain import Solver
from nzcvm.grids import terrain
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.grids.grid import Grid


@build_grids_from_config.register
def build_sw4(config: SW4GridConfig) -> dict[str, Grid]:
    refinements = sorted(
        config.refinements.items(), key=lambda refinement: refinement[1].bottom
    )
    _, top_refinement = refinements[0]
    finest = top_refinement.resolution

    ni = np.round(config.extent_x / finest).astype(int) + 1
    nj = np.round(config.extent_y / finest).astype(int) + 1

    frame = terrain.Frame.new(config.orientation, ni, nj, finest, config.chunks)

    geometry = frame.attrs["geometry"]
    trns = config.orientation.to_wgs84
    geometry_wgs = shapely.transform(
        geometry, lambda c: np.stack(trns.transform(*c.T), axis=1)
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Geometry: {shapely.to_geojson(geometry_wgs)}")

    surface = terrain.TerrainSurface.load(frame, config.surface)
    decay = config.terrain_decay(top_refinement.bottom)

    grids = {}
    top = 0.0
    for name, refinement in refinements:
        # SW4 runs on the flat nominal box when nominal, so the nominal spacing
        # is the resolution.  With topography, the thickest column sizes the
        # stretched first block instead.
        thickness = refinement.bottom - top
        if config.solver == Solver.PHYSICAL and top == 0.0:
            thickness = refinement.bottom - surface.minimum
        nk = int(np.round(thickness / refinement.resolution)) + 1
        grids[name] = terrain.terrain_grid(
            frame,
            surface,
            decay,
            np.linspace(top, refinement.bottom, nk),
            name=name,
            solver=config.solver,
            stride=round(refinement.resolution / finest),
        )
        top = refinement.bottom

    return grids
