"""EMOD3D binary velocity-model writer.

Writes ``rho3dfile.d``, ``vp3dfile.p``, and ``vs3dfile.s`` binary files into
a directory using memory-mapped I/O via :mod:`numpy.memmap`.
"""

import os
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.qualities import Qualities
from nzcvm.velocity_model import VelocityModel

RHOFILE = "rho3dfile.d"
VPFILE = "vp3dfile.p"
VSFILE = "vs3dfile.s"
DTYPE = np.float32

KM_PER_S = 1 / 1000.0


def _prepare_component(qualities: Qualities, component: Component) -> da.Array:
    contiguous_chunking = {0: "auto", 1: "auto", 2: -1}
    return qualities[component].data.rechunk(contiguous_chunking) * KM_PER_S


def to_emod3d(velocity_model: VelocityModel, directory: Path):
    """Write a single-block velocity model to an EMOD3D binary directory.

    Parameters
    ----------
    directory :
        Output directory, created if it doesn't exist.

    Raises
    ------
    ValueError
        If *dtree* contains more or fewer than one block.
    """

    # The EMOD3D format expects the grid to have the form z, y, x (with y points
    # *south*). This code takes i, j, k to mean east, north,
    # down. To correct for the difference it transposes the outputs and reverses
    # the y-direction.
    velocity_model = velocity_model.orient(
        Coordinate.J, Coordinate.K, Coordinate.I
    ).flip(Coordinate.J)

    resolutions = [grid.resolution for grid in velocity_model.grids.values()]

    if not np.allclose(resolutions, resolutions[0]):
        raise ValueError("EMOD3D format requires exactly one horizontal resolution")

    qualities = xr.concat(velocity_model.qualities.values(), Coordinate.K, join="outer")

    # Each array has the same size, may as well be the rho values
    file_size = qualities.rho.nbytes

    directory.mkdir(parents=True, exist_ok=True)
    with (
        open(directory / RHOFILE, "wb") as rho_file,
        open(directory / VPFILE, "wb") as vp_file,
        open(directory / VSFILE, "wb") as vs_file,
    ):
        os.posix_fallocate(rho_file.fileno(), 0, file_size)
        os.posix_fallocate(vp_file.fileno(), 0, file_size)
        os.posix_fallocate(vs_file.fileno(), 0, file_size)

    output_shape = (
        len(qualities.coords[Coordinate.J]),
        len(qualities.coords[Coordinate.K]),
        len(qualities.coords[Coordinate.I]),
    )

    rho_target = np.memmap(
        directory / RHOFILE, shape=output_shape, mode="r+", dtype=np.float32
    )
    vp_target = np.memmap(
        directory / VPFILE, shape=output_shape, mode="r+", dtype=np.float32
    )
    vs_target = np.memmap(
        directory / VSFILE, shape=output_shape, mode="r+", dtype=np.float32
    )

    rho_source = _prepare_component(qualities, Component.RHO)
    vp_source = _prepare_component(qualities, Component.VP)
    vs_source = _prepare_component(qualities, Component.VS)
    sources = [rho_source, vp_source, vs_source]
    targets = [rho_target, vp_target, vs_target]

    da.store(sources, targets, lock=True)  # ty: ignore[invalid-argument-type]
