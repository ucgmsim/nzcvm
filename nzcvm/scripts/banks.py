#!/usr/bin/env python3
"""Assign Banks Peninsula basin qualities using weathering from Ely."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
import xarray as xr

from nzcvm import ely_taper

app = typer.Typer(
    help="Apply the Ely GTL near-surface taper to the Banks Peninsula Volcanics mesh."
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
    """Apply the Ely GTL near-surface taper to the Banks Peninsula Volcanics mesh.

    The BPV top of each column is its shallowest vertex, so build the mesh
    with ``--no-pad-top``. A padded top would sit ``PAD_DEPTH`` above it.
    """
    dset = xr.open_dataset(mesh)
    # The mesh is depth-queried: its z is depth below the topography.
    dem_depth = dset.z.clip(min=0)

    # construct_mesh layers this mesh without padding. The shallowest vertex
    # of each (x, y) column is the BPV top it meshed.
    column_top = (
        pd.Series(dset.z.values)
        .groupby([dset.x.values, dset.y.values])
        .transform("min")
        .to_numpy()
    )
    bpv_depth = (dset.z - column_top).clip(min=0)

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

    # Keep the mesh's float32 qualities.
    def masked(inside: xr.DataArray, outside) -> xr.DataArray:
        return xr.where(gtl_mask, inside, outside).astype(np.float32)

    dset["vs"] = masked(gtl_qualities.vs, vs_full)
    dset["vp"] = masked(gtl_qualities.vp, vp_full)
    dset["rho"] = masked(gtl_qualities.rho, rho_full)
    if "qp" in dset.data_vars:
        dset["qp"] = masked(gtl_qualities.qp, dset["qp"])
        dset["qs"] = masked(gtl_qualities.qs, dset["qs"])

    dset.to_zarr(output, mode="w")
    print(f"Saved tapered Banks Peninsula Volcanics model to {output}")


if __name__ == "__main__":
    app()
