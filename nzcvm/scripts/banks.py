#!/usr/bin/env python3
"""Assign Banks Peninsula basin qualities using weathering from Ely."""

import os
from pathlib import Path
from typing import Annotated

import h5py
import typer
import xarray as xr

from nzcvm import ely_taper
from nzcvm.models.surface import median_spacing, read_surface_file, read_surface_window
from nzcvm.scripts import construct_mesh

app = typer.Typer(
    help="Apply the Ely GTL near-surface taper to the Banks Peninsula Volcanics mesh."
)

DEFAULT_BASEMENT = (
    Path(os.environ.get("NZCVM_DATA_ROOT", "."))
    / "regional/BanksPeninsulaVolcanics/BanksPeninsulaVolcanics_basement_WGS84.h5"
)

ELY_TAPER_DEPTH = 350.0
VS30_TAPER_DEPTH = 1000.0
VS0 = 0.700
VS_DEPTH = 1.500
VS_FULL = 2281.8
VP_FULL = 4000.0
RHO_FULL = 2393.0


@app.command()
def main(
    mesh: Annotated[
        Path,
        typer.Argument(
            help="Banks Peninsula Volcanics mesh (zarr) to apply the taper to.",
            exists=True,
        ),
    ] = Path("models/BanksPeninsula.zarr"),
    dem: Annotated[
        Path,
        typer.Argument(
            help="DEM surface (zarr) that the BPV basement depth is measured from.",
            exists=True,
        ),
    ] = Path("resources/dem.zarr"),
    basement: Annotated[
        Path,
        typer.Argument(
            help="BPV basement surface (HDF5, WGS84 lat/lon + elevation).",
            exists=True,
        ),
    ] = DEFAULT_BASEMENT,
    output: Annotated[
        Path, typer.Argument(help="Output path for the tapered mesh (zarr).")
    ] = Path("models/BanksPeninsula_GTL.zarr"),
    ely_taper_depth: Annotated[
        float,
        typer.Option(
            help="Depth (m) below the BPV top at which the taper reaches the reference velocities.",
            min=0.0,
        ),
    ] = ELY_TAPER_DEPTH,
    vs30_taper_depth: Annotated[
        float,
        typer.Option(
            help="Depth (m) below the DEM at which the Vs30 taper saturates.", min=0.0
        ),
    ] = VS30_TAPER_DEPTH,
    vs0: Annotated[
        float, typer.Option(help="Vs30 endpoint (km/s) at the DEM surface.", min=0.0)
    ] = VS0,
    vs_depth: Annotated[
        float,
        typer.Option(help="Vs30 endpoint (km/s) at vs30_taper_depth.", min=0.0),
    ] = VS_DEPTH,
    vs_full: Annotated[
        float,
        typer.Option(
            help="Reference Vs (m/s) the taper blends towards at ely_taper_depth, "
            "and the fill value outside the GTL mask.",
            min=0.0,
        ),
    ] = VS_FULL,
    vp_full: Annotated[
        float,
        typer.Option(
            help="Reference Vp (m/s) the taper blends towards at ely_taper_depth, "
            "and the fill value outside the GTL mask.",
            min=0.0,
        ),
    ] = VP_FULL,
    rho_full: Annotated[
        float,
        typer.Option(help="Density (kg/m^3) fill value outside the GTL mask.", min=0.0),
    ] = RHO_FULL,
) -> None:
    """Apply the Ely GTL near-surface taper to the Banks Peninsula Volcanics mesh."""
    dset = xr.open_dataset(mesh)
    # The mesh is depth-queried: its z is depth below the topography.
    dem_depth = dset.z.clip(min=0)

    # The basement is an elevation grid on latitude/longitude nodes. As
    # construct_mesh does, read it bilinearly and convert it to depth below
    # the topography at its own nodes, against the DEM averaged to its spacing.
    bpv_grid = read_surface_file(basement)
    with h5py.File(basement) as f:
        latitude, longitude = f["latitude"][:], f["longitude"][:]
    bbox = (latitude.min(), latitude.max(), longitude.min(), longitude.max())
    dem_grid = read_surface_window(dem, bbox, median_spacing(bpv_grid))
    bpv_top = construct_mesh.depth_below(bpv_grid, dem_grid)
    bpv_top_depth = xr.apply_ufunc(
        lambda x, y: bpv_top(x, y, clamp=False)[0], dset.x, dset.y
    )
    bpv_depth = (dset.z - bpv_top_depth).clip(min=0)

    # Apparent surface Vs30 (m/s) at each point, the taper's near-surface endpoint.
    vs_bpv_top = (vs0 + (vs_depth - vs0) * (dem_depth / vs30_taper_depth)) * 1000.0

    gtl_qualities = ely_taper._ely_vs_profile(
        depth=bpv_depth,
        vs30=vs_bpv_top,
        vp_at_z_t=xr.full_like(dset.vp, vp_full),
        vs_at_z_t=xr.full_like(dset.vs, vs_full),
        depth_t=ely_taper_depth,
    )

    # True within 1000m of the DEM and within 350m of the BPV basement.
    gtl_mask = (dem_depth < vs30_taper_depth) & (bpv_depth < ely_taper_depth)

    dset["vs"] = xr.where(gtl_mask, gtl_qualities.vs, vs_full)
    dset["vp"] = xr.where(gtl_mask, gtl_qualities.vp, vp_full)
    dset["rho"] = xr.where(gtl_mask, gtl_qualities.rho, rho_full)
    if "qp" in dset.data_vars:
        dset["qp"] = xr.where(gtl_mask, gtl_qualities.qp, dset["qp"])
        dset["qs"] = xr.where(gtl_mask, gtl_qualities.qs, dset["qs"])

    dset.to_zarr(output, mode="w")
    print(f"Saved tapered Banks Peninsula Volcanics model to {output}")


if __name__ == "__main__":
    app()
