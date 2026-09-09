#!/usr/bin/env python3

# Run this script like
# ./near_fault.py near_fault_config.toml <MODEL_OUTPUT>

import numpy as np
import xarray as xr

from nzcvm.grids import Grid
from nzcvm.layers.core import Layer
from nzcvm.layers.functional import functional_layer
from nzcvm.query import ModelRange
from nzcvm.scripts import nzcvm_cli


# This example uses a point source for simplicity, but it could be
# any complex geometry.
@functional_layer
def fault_zone(
    grid: Grid,
    model_range: ModelRange,
    *,
    next_layer: Layer,
    fault_x: float,
    fault_y: float,
    fault_z: float,
):
    dist = np.sqrt(
        (grid.x - fault_x) ** 2 + (grid.y - fault_y) ** 2 + (grid.z - fault_z) ** 2
    )
    qualities = next_layer(grid, model_range)
    # Artificially lower Vs values in 10km radius (numbers chosen for effect).
    qualities["vs"] = xr.where(dist < 10000.0, 0.8 * qualities.vs, qualities.vs)
    return qualities


if __name__ == "__main__":
    nzcvm_cli.app()
