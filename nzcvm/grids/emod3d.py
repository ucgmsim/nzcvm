"""EMOD3D velocity model grid builder.

Builds the one block defined by an
:class:`~nzcvm.config.grids.emod3d.EMOD3DGrid` on the shared hybrid
terrain-following coordinate in :mod:`nzcvm.grids.terrain`, with
``nz`` levels at a fixed nominal spacing.  EMOD3D runs on the flat nominal
box, so the grid is always a nominal one.

See Also
--------
nzcvm.config.grids.emod3d.EMOD3DGrid : Grid configuration.
nzcvm.grids.terrain : The shared terrain-following construction.
"""

import numpy as np

from nzcvm.config.grids.emod3d import EMOD3DGrid
from nzcvm.config.grids.terrain import Solver
from nzcvm.grids import terrain
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.grids.grid import Grid

GRID_NAME = "grid_0"


@build_grids_from_config.register
def build_emod3d(config: EMOD3DGrid) -> dict[str, Grid]:
    resolution = config.resolution
    frame = terrain.Frame.new(
        config.orientation, config.nx, config.ny, resolution, config.chunks
    )
    surface = terrain.TerrainSurface.load(frame, config.surface)

    levels = np.arange(config.nz) * resolution
    decay = config.terrain_decay(float(levels[-1]))

    grid = terrain.terrain_grid(
        frame,
        surface,
        decay,
        levels,
        name=GRID_NAME,
        solver=Solver.NOMINAL,
    )
    return {grid.name: grid}
