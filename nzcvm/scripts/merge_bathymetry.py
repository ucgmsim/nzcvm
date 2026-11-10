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

The result keeps the DEM's node spacing and alignment, extended to cover
the bathymetry too. Beyond the DEM is sea.
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


def _extend(axis: np.ndarray, low: float, high: float) -> tuple[np.ndarray, int, int]:
    """Extend the uniform *axis* with nodes on its step until it spans [low, high].

    Returns the new axis and the number of nodes added before and after.
    """
    step = axis[1] - axis[0]
    first, last = (axis[0], axis[-1]) if step > 0 else (axis[-1], axis[0])
    before = max(0, math.floor((first - low) / abs(step)))
    after = max(0, math.floor((high - last) / abs(step)))
    if step < 0:
        before, after = after, before
    n = len(axis) + before + after
    return axis[0] - before * step + step * np.arange(n), before, after


def _fractional_index(axis: np.ndarray, at: np.ndarray) -> np.ndarray:
    return (at - axis[0]) / (axis[1] - axis[0])


def merge_bathymetry(
    dem: xr.DataArray,
    bathymetry: xr.DataArray,
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
        ``(y, x)`` NZTM lattice, NaN where it has no data. Loaded whole.
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
        The merged topography as depth (m, +z down). Its lattice is *dem*'s,
        extended with the same spacing and alignment over *bathymetry*'s
        extent. The sea takes ``w · min(bathymetry, 0)``, or 0 where the
        bathymetry has no data.
    """
    dem = dem.transpose("y", "x")
    spacing = float(abs(dem.x.values[1] - dem.x.values[0]))
    if not math.isclose(spacing, abs(dem.y.values[1] - dem.y.values[0]), rel_tol=1e-6):
        raise ValueError("The DEM's spacing must be the same along x and y.")

    bx, by = bathymetry.x.values, bathymetry.y.values
    x, left, right = _extend(dem.x.values, bx.min(), bx.max())
    y, top, bottom = _extend(dem.y.values, by.min(), by.max())

    depth = dem.data
    if not isinstance(depth, da.Array):
        depth = da.from_array(depth, chunks=SURFACE_CHUNK_SIZE)
    # Beyond the DEM is sea.
    depth = da.pad(depth, ((top, bottom), (left, right)), constant_values=0.0)
    depth = depth.rechunk(SURFACE_CHUNK_SIZE)

    # Fill gaps with sea level before the spline prefilter, which would
    # otherwise spread each NaN over the whole array. Nodes whose stencil
    # reaches a gap fall back to sea level below.
    values = bathymetry.transpose("y", "x").values.astype(np.float64)
    gaps = np.isnan(values)
    coefficients = np.where(gaps, 0.0, values)
    if order > 1:
        coefficients = ndimage.spline_filter(coefficients, order=order)
    gap_mask = np.where(gaps, np.nan, 0.0)
    rows, cols = _fractional_index(by, y), _fractional_index(bx, x)

    def sample(grid: np.ndarray, r: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
        rr, cc = np.meshgrid(r, c, indexing="ij")
        return ndimage.map_coordinates(
            grid, [rr, cc], order=k, mode="constant", cval=np.nan, prefilter=False
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
        r, c = rows[r0:r1], cols[c0:c1]
        # NaN where the stencil leaves the bathymetry or touches a gap.
        elevation = sample(coefficients, r, c, order) + sample(gap_mask, r, c, 1)
        seafloor = np.nan_to_num(np.minimum(elevation, 0.0), nan=0.0)
        sea_depth = round_mantissa(-weight * seafloor, keep_bits)
        return np.where(depth == 0, sea_depth, depth).astype(np.float32)

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
        Path,
        typer.Argument(
            help="Bathymetry raster in NZTM (elevation, negative below sea "
            "level), cropped to the region the surface should cover. "
            "`just bathymetry` builds it.",
            exists=True,
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
    if raster.rio.crs is None or raster.rio.crs.to_epsg() != NZTM_EPSG:
        raise typer.BadParameter(
            f"The bathymetry must be in NZTM (EPSG:{NZTM_EPSG}), not "
            f"{raster.rio.crs}. Warp it with gdalwarp -t_srs EPSG:{NZTM_EPSG}."
        )
    bathymetry = raster.sel(band=band).drop_vars(["band", "spatial_ref"])

    dem = xr.open_zarr(dem_path)
    merged = merge_bathymetry(
        dem.z, bathymetry, taper=taper, order=order, keep_bits=keep_bits
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
