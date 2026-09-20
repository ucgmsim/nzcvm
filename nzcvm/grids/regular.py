"""Topography-following spatially regular velocity model grid builder.

Provides :func:`build_regular` for constructing the 3D curvilinear mesh
defined by a :class:`~nzcvm.config.grids.regular.RegularGridConfig`.
The builder returns :class:`xarray.DataTree` nodes with chunked coordinates
and topography-following ``z`` / ``depth`` arrays at a strictly fixed Z resolution.
"""

import dask
import numpy as np
from scipy.spatial.transform import Rotation

from nzcvm import coordinates
from nzcvm.config.grids.regular import RegularGridConfig
from nzcvm.grids import helpers
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.grids.grid import Grid
from nzcvm.models.surface import Surface


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

    grid = helpers.topography_following_grid(
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
