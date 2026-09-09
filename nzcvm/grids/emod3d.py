from collections.abc import Callable

import dask
import numpy as np
import xarray as xr
from scipy.spatial.transform import Rotation

from nzcvm import coordinates
from nzcvm.config.grids.emod3d import EMOD3DGrid, TopographyType
from nzcvm.coordinates import Coordinate
from nzcvm.grids import helpers
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.grids.grid import Grid, GridSchema
from nzcvm.models.surface import Surface

LAYER_DIM = "layer"
GRID_NAME = "grid_0"


def squashed_interpolator(
    z_surface: xr.DataArray, depth: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    # Rely on implicit mathematical broadcasting.
    # Returning depth as 1D here is fine, it gets explicitly broadcasted to 3D at the end.
    return z_surface + depth, depth


def squashed_tapered_interpolator(
    z_surface: xr.DataArray, depth: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    elevation_asl = -z_surface
    taper = 1.0 - depth / elevation_asl.where(elevation_asl > 0, np.inf)
    taper = taper.clip(min=0)
    z = depth - elevation_asl * taper
    return z, z - z_surface


def _topography_type_interpolator(
    topo_type: TopographyType,
) -> Callable[..., tuple[xr.DataArray, xr.DataArray]]:
    match topo_type:
        case TopographyType.SQUASHED:
            return squashed_interpolator
        case TopographyType.SQUASHED_TAPERED:
            return squashed_tapered_interpolator


def _depth_array(nk: int, resolution: float) -> xr.DataArray:
    k = np.arange(nk)
    k_da = xr.DataArray(
        np.float32(k) * np.float32(resolution),
        dims=[Coordinate.K],
        coords={Coordinate.K: k},
    )
    # Chunking only ever applies to i/j. k always stays one chunk.
    return k_da.chunk({Coordinate.K: -1})


@build_grids_from_config.register
def build_emod3d(config: EMOD3DGrid) -> dict[str, Grid]:
    resolution = config.resolution

    ox, oy = helpers.raw_coordinates(
        config.nx,
        config.ny,
        config.resolution,
        0.0,
        config.chunks,
    )
    min_x, min_y = dask.compute(ox.sel(i=0, j=0), oy.sel(i=0, j=0))
    min_x = min_x.item()
    min_y = min_y.item()
    orientation = config.orientation

    transform = coordinates.translate(
        orientation.grid_origin_x, orientation.grid_origin_y
    ) @ Rotation.from_rotvec(
        np.array([0, 0, -orientation.grid_azimuth]), degrees=True
    ).as_matrix().astype(np.float32)

    geometry = helpers.outline(
        transform, config.nx * config.resolution, config.ny * config.resolution
    )

    x_phys, y_phys = coordinates.apply_affine_transform(transform, ox, oy)
    min_x, min_y = coordinates.apply_affine_transform(transform, min_x, min_y)
    min_lon, min_lat = orientation.to_wgs84.transform(min_x, min_y)

    topographic_surface = Surface.load(config.surface)
    z_surface = helpers.compute_surface_elevation(topographic_surface, x_phys, y_phys)

    interpolator = _topography_type_interpolator(config.topo_type)

    depth_1d = _depth_array(config.nz, config.resolution)

    z_phys, depth_out = interpolator(z_surface, depth_1d)

    x, y, z, depth = xr.broadcast(x_phys, y_phys, z_phys, depth_out)

    depth, x, y, z = helpers.ensure_chunks(depth, x, y, z)

    grid = GridSchema.new(
        x,
        y,
        z,
        depth,
        name=GRID_NAME,
        resolution=resolution,
        origin_lon=orientation.origin_lon,
        origin_lat=orientation.origin_lat,
        azimuth=orientation.azimuth,
        grid_azimuth=orientation.grid_azimuth,
        bottom_left_lon=min_lon,
        bottom_left_lat=min_lat,
        geometry=geometry,
    )

    return {grid.name: grid}
