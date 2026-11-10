"""Merge a coarse bathymetry under a fine land DEM, as one topography surface.

The land DEM (Copernicus GLO-30, built by ``just dem``) holds exactly 0 m
over the sea. Nodes above or below 0 are land, and keep their value
exactly. Sea nodes take the bathymetry instead, interpolated onto the
DEM's lattice and clipped to at most sea level.

A coarse bathymetry doesn't meet the coastline at 0 m: its cells straddle
the shore, so the first sea node can sit tens of metres deep and leave a
cliff at the coast. So the merge multiplies the bathymetry by a weight
that rises smoothly (a smoothstep) from 0 at the coast to 1 a *taper*
distance offshore. The seafloor leaves the shore at sea level and joins the
bathymetry with a continuous slope.

The result is on the DEM's own lattice. The command reads only the window
of the bathymetry under the DEM, straight from a zip if need be, and
samples it in its own projection, so it never warps or stores a copy.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import dask.array as da
import numpy as np
import typer
import xarray as xr
from scipy import ndimage

from nzcvm.models.mesh import (
    DEFAULT_STRUCTURED_ENCODING_SETTINGS,
    SURFACE_CHUNK_SIZE,
    StructuredMesh,
    surface_mesh,
)
from nzcvm.scripts._extras import missing_extra

app = typer.Typer()

#: EPSG code of NZTM 2000, the frame every nzcvm surface's coordinates are in.
NZTM_EPSG = 2193

#: Distance (m) past the DEM's bounds to read the bathymetry, so the spline
#: has support at the edges.
WINDOW_MARGIN = 5000.0


def coast_weight(sea: np.ndarray, spacing: float, taper: float) -> np.ndarray:
    """Weight of the bathymetry at each node: 0 at the coast, 1 offshore.

    Parameters
    ----------
    sea :
        2-D mask, True at sea nodes.
    spacing :
        Node spacing (m), the same along both axes.
    taper :
        Distance (m) from the nearest land node over which the weight rises
        from 0 to 1.

    Returns
    -------
    np.ndarray
        The smoothstep ``3 s² - 2 s³`` of ``s = min(d / taper, 1)``, where
        ``d`` is each node's distance to the nearest land node. Land nodes
        get 0, and a block without land is all 1.
    """
    if sea.all():
        return np.ones(sea.shape, dtype=np.float32)
    if taper <= 0:
        return sea.astype(np.float32)
    distance = ndimage.distance_transform_edt(sea, sampling=spacing)
    s = np.clip(distance / taper, 0.0, 1.0)
    return (s * s * (3.0 - 2.0 * s)).astype(np.float32)


def round_mantissa(values: np.ndarray, keep_bits: int) -> np.ndarray:
    """Round float32 *values* to *keep_bits* bits of mantissa.

    The dropped bits become zeros, which compress far better. The relative
    error is at most ``2 ** -(keep_bits + 1)``.

    Parameters
    ----------
    values :
        float32 array.
    keep_bits :
        Mantissa bits to keep, 0 to 23. 23 keeps every bit.

    Returns
    -------
    np.ndarray
        The rounded values, as float32.
    """
    values = np.ascontiguousarray(values, dtype=np.float32)
    drop = 23 - keep_bits
    if drop <= 0:
        return values
    bits = values.view(np.uint32)
    half = np.uint32(1 << (drop - 1))
    mask = np.uint32((0xFFFFFFFF << drop) & 0xFFFFFFFF)
    return ((bits + half) & mask).view(np.float32)


#: Spacing, in nodes along each axis of a block, of the nodes projected
#: exactly into the bathymetry's projection. Nodes between them take
#: interpolated coordinates. Projections bend over hundreds of kilometres,
#: so this is exact to well under a millimetre.
PROJECT_STRIDE = 16


def _project_block(
    x: np.ndarray, y: np.ndarray, crs: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """The ``(len(y), len(x))`` lattice of NZTM nodes, in the projection *crs*.

    *crs* None leaves the nodes in NZTM.
    """
    xx, yy = np.meshgrid(x, y)
    if crs is None:
        return xx, yy
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator

    # A Transformer isn't thread-safe, so each block makes its own.
    transformer = Transformer.from_crs(f"EPSG:{NZTM_EPSG}", crs, always_xy=True)
    rows = np.unique(np.r_[np.arange(0, len(y), PROJECT_STRIDE), len(y) - 1])
    cols = np.unique(np.r_[np.arange(0, len(x), PROJECT_STRIDE), len(x) - 1])
    if len(rows) < 2 or len(cols) < 2:
        return transformer.transform(xx, yy)
    cx, cy = np.meshgrid(x[cols], y[rows])
    u, v = transformer.transform(cx, cy)
    ii, jj = np.meshgrid(np.arange(len(y)), np.arange(len(x)), indexing="ij")
    at = np.c_[ii.ravel(), jj.ravel()]
    shape = (len(y), len(x))
    return (
        RegularGridInterpolator((rows, cols), u)(at).reshape(shape),
        RegularGridInterpolator((rows, cols), v)(at).reshape(shape),
    )


def merge_bathymetry(
    dem: xr.DataArray,
    bathymetry: xr.DataArray,
    crs: str | None = None,
    taper: float = 1000.0,
    order: int = 3,
    keep_bits: int = 23,
) -> xr.DataArray:
    """Fill the sea in *dem* with *bathymetry*, tapered to sea level at the coast.

    Parameters
    ----------
    dem :
        The land DEM as depth (m, +z down), on a uniform ``(y, x)`` NZTM
        lattice with the same spacing along both axes. Exactly 0 is the
        sea. May be Dask-backed, and stays lazy.
    bathymetry :
        Elevation (m, +z up, negative below sea level) on a uniform
        ``(y, x)`` lattice in the projection *crs*, NaN where it has no
        data. Loaded whole, so crop it to the DEM first.
    crs :
        The bathymetry's projection, as pyproj accepts it (WKT, ``EPSG:n``).
        None for NZTM.
    taper :
        Distance (m) offshore over which the bathymetry fades in.
    order :
        Spline order interpolating the bathymetry onto the DEM's nodes:
        1 for bilinear, 3 for cubic.
    keep_bits :
        Mantissa bits kept of each sea node's depth (see
        :func:`round_mantissa`), so the sea compresses well. Land keeps
        every bit.

    Returns
    -------
    xr.DataArray
        The merged topography as depth (m, +z down), on *dem*'s lattice and
        chunks. The sea takes ``w · min(bathymetry, 0)``, or 0 where the
        bathymetry has no data.
    """
    dem = dem.transpose("y", "x")
    spacing = float(abs(dem.x.values[1] - dem.x.values[0]))
    if not math.isclose(spacing, abs(dem.y.values[1] - dem.y.values[0]), rel_tol=1e-6):
        raise ValueError("The DEM's spacing must be the same along x and y.")

    x, y = dem.x.values, dem.y.values
    bx, by = bathymetry.x.values, bathymetry.y.values
    depth = dem.data
    if not isinstance(depth, da.Array):
        depth = da.from_array(depth, chunks=SURFACE_CHUNK_SIZE)

    # Fill gaps with sea level before the spline prefilter, which would
    # otherwise spread each NaN over the whole array. Nodes whose stencil
    # reaches a gap fall back to sea level below.
    values = bathymetry.transpose("y", "x").values.astype(np.float64)
    gaps = np.isnan(values)
    coefficients = np.where(gaps, 0.0, values)
    if order > 1:
        coefficients = ndimage.spline_filter(coefficients, order=order)
    gap_mask = np.where(gaps, np.nan, 0.0)

    def sample(grid: np.ndarray, at: list[np.ndarray], k: int) -> np.ndarray:
        return ndimage.map_coordinates(
            grid, at, order=k, mode="constant", cval=np.nan, prefilter=False
        )

    sea = depth == 0
    # One node more than the taper, so the distance is exact out to it.
    halo = math.ceil(taper / spacing) + 1
    weight = da.map_overlap(
        coast_weight,
        sea,
        depth=halo,
        boundary=True,  # Past the edge is sea, so it never pulls in land.
        dtype=np.float32,
        spacing=spacing,
        taper=taper,
    )

    def fill(
        depth: np.ndarray, weight: np.ndarray, block_info: dict | None = None
    ) -> np.ndarray:
        assert block_info is not None
        (r0, r1), (c0, c1) = block_info[0]["array-location"]
        u, v = _project_block(x[c0:c1], y[r0:r1], crs)
        # Fractional indices into the bathymetry's lattice.
        at = [(v - by[0]) / (by[1] - by[0]), (u - bx[0]) / (bx[1] - bx[0])]
        # NaN where the stencil leaves the bathymetry or touches a gap.
        elevation = sample(coefficients, at, order) + sample(gap_mask, at, 1)
        seafloor = np.nan_to_num(np.minimum(elevation, 0.0), nan=0.0)
        sea_depth = round_mantissa(-weight * seafloor, keep_bits)
        return np.where(depth == 0, sea_depth, depth).astype(np.float32)

    # A halo wider than a chunk makes map_overlap rechunk, so line the weight
    # back up with the depth before combining them block by block.
    weight = weight.rechunk(depth.chunks)
    merged = da.map_blocks(fill, depth, weight, dtype=np.float32)
    return xr.DataArray(merged, dims=("y", "x"), coords={"y": y, "x": x})


@app.command()
def main(
    dem_path: Annotated[
        Path,
        typer.Argument(
            help="Land DEM surface (Zarr, depth +z down, exactly 0 over the "
            "sea), as `nzcvm convert-tiff --flip` builds from Copernicus.",
            exists=True,
        ),
    ],
    bathymetry_path: Annotated[
        str,
        typer.Argument(
            help="Bathymetry raster (elevation, negative below sea level) in any "
            "projection, as GDAL opens it: a file, or a path inside a zip such as "
            "/vsizip/nzbathy2016.zip/nzbathy_2016.tif. Only the window under "
            "the DEM is read.",
        ),
    ],
    output_path: Annotated[Path, typer.Argument(help="Output Zarr surface.")],
    band: int = 1,
    taper: Annotated[
        float,
        typer.Option(
            help="Distance (m) offshore over which the bathymetry fades in from "
            "sea level at the coast.",
            min=0.0,
        ),
    ] = 1000.0,
    order: Annotated[
        int,
        typer.Option(
            help="Spline order interpolating the bathymetry: 1 bilinear, 3 cubic.",
            min=1,
            max=5,
        ),
    ] = 3,
    keep_bits: Annotated[
        int,
        typer.Option(
            help="Mantissa bits kept of the sea's depths, so they compress well. "
            "10 bits rounds to within 0.05% (1 m at 2 km depth); 23 keeps all.",
            min=0,
            max=23,
        ),
    ] = 10,
) -> None:
    """Fill the sea in a land DEM with a coarser bathymetry."""
    try:
        import rioxarray
    except ImportError as err:
        raise missing_extra("rioxarray", "raster", "merge-bathymetry") from err

    raster = rioxarray.open_rasterio(bathymetry_path, masked=True)
    if not isinstance(raster, xr.DataArray):
        raise typer.BadParameter("The bathymetry must be a single raster.")
    if raster.rio.crs is None:
        raise typer.BadParameter("The bathymetry has no projection.")

    dem = xr.open_zarr(dem_path)
    x, y = dem.x.values, dem.y.values
    # Only the window under the DEM, a little past it so the spline has
    # support at the edges.
    window = raster.rio.clip_box(
        float(x.min()) - WINDOW_MARGIN,
        float(y.min()) - WINDOW_MARGIN,
        float(x.max()) + WINDOW_MARGIN,
        float(y.max()) + WINDOW_MARGIN,
        crs=f"EPSG:{NZTM_EPSG}",
    )
    bathymetry = window.sel(band=band).drop_vars(["band", "spatial_ref"]).load()
    merged = merge_bathymetry(
        dem.z,
        bathymetry,
        crs=raster.rio.crs.to_wkt(),
        taper=taper,
        order=order,
        keep_bits=keep_bits,
    )
    surface: StructuredMesh = surface_mesh(
        output_path.stem,
        merged.data,
        merged.x.values,
        merged.y.values,
        frame=tuple(dem.attrs["frame"]),
    )
    surface.to_zarr(output_path, encoding=DEFAULT_STRUCTURED_ENCODING_SETTINGS)


if __name__ == "__main__":
    app()
