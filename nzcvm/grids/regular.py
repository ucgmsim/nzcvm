"""Topography-following spatially regular velocity model grid builder.

Provides :func:`build_regular` for constructing the 3D curvilinear mesh
defined by a :class:`~nzcvm.config.grids.regular.RegularGridConfig`.
The builder returns :class:`xarray.DataTree` nodes with chunked coordinates
and topography-following ``z`` / ``depth`` arrays at a strictly fixed Z resolution.
"""

from typing import Any

import dask
import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation

from nzcvm import coordinates
from nzcvm.config.grids.regular import RegularGridConfig
from nzcvm.coordinates import Coordinate
from nzcvm.grids import helpers
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.grids.grid import Grid, GridSchema
from nzcvm.models.surface import Surface


def _regular_grid(
    x_phys: xr.DataArray,
    y_phys: xr.DataArray,
    surface: xr.DataArray,
    thickness: float,
    resolution_z: float,
    **kwargs: Any,
) -> Grid:
    nk = np.round(thickness / resolution_z).astype(int) + 1
    k = np.arange(nk)

    # Depth is purely a function of k and resolution_z
    depth_values = np.linspace(0.0, thickness, num=nk, dtype=np.float32)

    # Chunking only ever applies to i/j. k always stays one chunk.
    zeta_depth = xr.DataArray(
        depth_values,
        dims=[Coordinate.K],
        coords={Coordinate.K: k},
    ).chunk({Coordinate.K: -1})

    # Elevation (z) is the surface elevation shifted downward by the fixed depths.
    # The bottom then follows the topography exactly.
    z = surface + zeta_depth

    # Same idiomatic trick as the SW4 template to ensure coordinate ordering (i, j, k)
    depth = zeta_depth

    x, y, z, depth = xr.broadcast(x_phys, y_phys, z, depth)

    x, y, z, depth = helpers.ensure_chunks(x, y, z, depth)

    return GridSchema.new(
        x,
        y,
        z,
        depth,
        **kwargs,
    )


@build_grids_from_config.register
def build_regular(config: RegularGridConfig) -> dict[str, Grid]:
    offset = 0.0

    ni = np.round(config.extent_x / config.resolution_x).astype(int) + 1
    nj = np.round(config.extent_y / config.resolution_y).astype(int) + 1
    rounded_extent_x = ni * config.resolution_x
    rounded_extent_y = nj * config.resolution_y

    # Generate unit coordinates (resolution=1.0) and scale manually to handle
    # independent X and Y resolutions with the existing helper.
    ox, oy = helpers.raw_coordinates(
        ni,
        nj,
        1.0,
        offset,
        config.chunks,
    )
    ox = ox * config.resolution_x
    oy = oy * config.resolution_y

    min_x, min_y = dask.compute(ox.isel(i=0, j=0), oy.isel(i=0, j=0))
    min_x = min_x.item()
    min_y = min_y.item()

    orientation = config.orientation
    transform = (
        coordinates.translate(orientation.grid_origin_x, orientation.grid_origin_y)
        # Consistent with the rotation specified in the z-axis down convention
        @ Rotation.from_rotvec(
            np.array([0.0, 0.0, -orientation.grid_azimuth]), degrees=True
        )
        .as_matrix()
        .astype(np.float32)
    )

    geometry = helpers.outline(transform, rounded_extent_x, rounded_extent_y)

    x_phys, y_phys = coordinates.apply_affine_transform(transform, ox, oy)
    min_x, min_y = coordinates.apply_affine_transform(transform, min_x, min_y)
    min_lon, min_lat = orientation.to_wgs84.transform(min_x, min_y)

    topographic_surface = Surface.load(config.surface)
    z_surface = helpers.compute_surface_elevation(
        topographic_surface,
        x_phys,
        y_phys,
    )

    grid = _regular_grid(
        x_phys,
        y_phys,
        z_surface,
        name="grid",
        thickness=config.thickness,
        origin_lat=orientation.origin_lat,
        origin_lon=orientation.origin_lon,
        azimuth=orientation.azimuth,
        grid_azimuth=orientation.grid_azimuth,
        bottom_left_lon=min_lon,
        bottom_left_lat=min_lat,
        resolution_z=config.resolution_z,
        resolution=min((config.resolution_x, config.resolution_y, config.resolution_z)),
        geometry=geometry,
    )

    return {grid.name: grid}
