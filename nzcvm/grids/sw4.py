"""SW4 curvilinear velocity model grid builder.

Provides :func:`skeleton_velocity_model` and :func:`fill_grid` for
constructing the 3D curvilinear mesh defined by a
:class:`~nzcvm.config.grids.sw4.SW4GridConfig`.  Both return
:class:`xarray.DataTree` nodes with chunked coordinates and topography-following
``z`` / ``depth`` arrays.

See Also
--------
nzcvm.config.grids.sw4.SW4GridConfig : Grid configuration.
nzcvm.grids.helpers : Shared surface-elevation helpers.
"""

import logging
from typing import Any

import dask
import numpy as np
import pyproj
import shapely
import xarray as xr
from scipy.spatial.transform import Rotation

from nzcvm import coordinates
from nzcvm.config.grids.sw4 import SW4GridConfig
from nzcvm.coordinates import Coordinate
from nzcvm.grids import helpers
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.grids.grid import Grid, GridSchema
from nzcvm.models.surface import Surface


def _logical_k_indices(nk: int, dtype: np.dtype, k_offset: int = 0) -> xr.DataArray:
    k_indices = np.arange(nk) + k_offset
    k_coord = np.linspace(0.0, 1.0, num=nk, dtype=dtype)

    return xr.DataArray(
        k_coord,
        dims=Coordinate.K,
        coords={Coordinate.K: k_indices},
    )


def _curvilinear_grid(
    x_phys: xr.DataArray,
    y_phys: xr.DataArray,
    surface: xr.DataArray,
    top: float | xr.DataArray,
    bottom: float,
    resolution: float,
    **kwargs: Any,
) -> Grid:
    z_min = top

    if isinstance(top, xr.DataArray):
        z_min = top.min().compute().item()

    thickness = bottom - z_min

    nk = np.round(thickness / resolution).astype(int) + 1
    k = np.arange(nk)
    # Chunking only ever applies to i/j. k always stays one chunk.
    zeta = xr.DataArray(
        np.linspace(0, 1, num=nk, dtype=np.float32),
        dims=[Coordinate.K],
        coords={Coordinate.K: k},
    ).chunk({Coordinate.K: -1})

    z = top * (np.float32(1.0) - zeta) + bottom * zeta

    # HACK: If wrote this the idiomatic way like so:
    # depth = z - surface
    # Then if z is a one-dimensional variable of shape (k,), which happens when top and bottom are both floats, the array has shape
    # (k, i, j)
    # But the GridSchema enforces
    # (i, j, k)
    # So ordering it as
    # depth = -surface + z
    # puts the k coordinate at the end, because this adds z to the surface
    # rather than subtracting the surface from z.
    depth = -surface + z

    x, y, z, depth = xr.broadcast(x_phys, y_phys, z, depth)

    x, y, z, depth = helpers.ensure_chunks(x, y, z, depth)

    return GridSchema.new(
        x,
        y,
        z,
        depth,
        resolution=resolution,
        **kwargs,
    )


def _resample_refinement(
    x: xr.DataArray,
    y: xr.DataArray,
    z: xr.DataArray,
    resolution: float,
    refinement: float,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    refinement_ratio = int(refinement / resolution)
    ni = len(x.coords[Coordinate.I])
    nj = len(x.coords[Coordinate.J])
    x_sample = x.isel(
        {
            Coordinate.I: range(0, ni, refinement_ratio),
            Coordinate.J: range(0, nj, refinement_ratio),
        }
    )
    y_sample = y.isel(
        {
            Coordinate.I: range(0, ni, refinement_ratio),
            Coordinate.J: range(0, nj, refinement_ratio),
        }
    )
    z_sample = z.isel(
        {
            Coordinate.I: range(0, ni, refinement_ratio),
            Coordinate.J: range(0, nj, refinement_ratio),
        }
    )
    return x_sample, y_sample, z_sample


@build_grids_from_config.register
def build_sw4(config: SW4GridConfig) -> dict[str, Grid]:
    refinements = sorted(
        config.refinements.items(), key=lambda refinement: refinement[1].bottom
    )
    top_name, top_refinement = refinements[0]

    offset = 0.0

    ni = np.round(config.extent_x / top_refinement.resolution).astype(int) + 1
    nj = np.round(config.extent_y / top_refinement.resolution).astype(int) + 1
    rounded_extent_x = ni * top_refinement.resolution
    rounded_extent_y = nj * top_refinement.resolution

    orientation = config.orientation
    transform = (
        coordinates.translate(orientation.grid_origin_x, orientation.grid_origin_y)
        # This is consistent with the rotation specified in the z-axis down
        # convention.
        @ Rotation.from_rotvec(
            np.array([0.0, 0.0, -orientation.grid_azimuth]), degrees=True
        )
        .as_matrix()
        .astype(np.float32)
    )

    geometry = helpers.outline(transform, rounded_extent_x, rounded_extent_y)
    trns = pyproj.Transformer.from_crs(orientation.crs, 4326, always_xy=True)
    geometry_wgs = shapely.transform(
        geometry, lambda c: np.stack(trns.transform(*c.T), axis=1)
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Geometry: {shapely.to_geojson(geometry_wgs)}")

    ox, oy = helpers.raw_coordinates(
        ni,
        nj,
        top_refinement.resolution,
        offset,
        config.chunks,
    )
    min_x, min_y = dask.compute(ox.isel(i=0, j=0), oy.isel(i=0, j=0))
    min_x = min_x.item()
    min_y = min_y.item()

    x_phys, y_phys = coordinates.apply_affine_transform(transform, ox, oy)
    min_x, min_y = coordinates.apply_affine_transform(transform, min_x, min_y)
    min_lon, min_lat = orientation.to_wgs84.transform(min_x, min_y)

    topographic_surface = Surface.load(config.surface)
    z_surface = helpers.compute_surface_elevation(
        topographic_surface,
        x_phys,
        y_phys,
    )

    grids = []

    # First layer: curvilinear mesh to account for topography.
    grids.append(
        _curvilinear_grid(
            x_phys,
            y_phys,
            z_surface,
            z_surface,
            top_refinement.bottom,
            top_refinement.resolution,
            name=top_name,
            origin_lat=orientation.origin_lat,
            origin_lon=orientation.origin_lon,
            azimuth=orientation.azimuth,
            grid_azimuth=orientation.grid_azimuth,
            bottom_left_lon=min_lon,
            bottom_left_lat=min_lat,
            geometry=geometry,
        )
    )

    # Next n - 1 layers: Cartesian grids filled between the bottom of the
    # previous layer and the new bottom.
    top = top_refinement.bottom
    for name, refinement in refinements[1:]:
        x_refinement, y_refinement, z_refinement = _resample_refinement(
            x_phys, y_phys, z_surface, top_refinement.resolution, refinement.resolution
        )
        grids.append(
            _curvilinear_grid(
                x_refinement,
                y_refinement,
                z_refinement,
                top,
                refinement.bottom,
                refinement.resolution,
                name=name,
                origin_lat=orientation.origin_lat,
                origin_lon=orientation.origin_lon,
                azimuth=orientation.azimuth,
                grid_azimuth=orientation.grid_azimuth,
                bottom_left_lon=min_lon,
                bottom_left_lat=min_lat,
                geometry=geometry,
            )
        )
        top = refinement.bottom

    return {grid.name: grid for grid in grids}
