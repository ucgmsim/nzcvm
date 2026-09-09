"""Convert an HDF5 topography file to a VTK UnstructuredGrid (VTKHDF compatible)."""

from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pyproj
import typer

from nzcvm.models.mesh import DEFAULT_STRUCTURED_ENCODING_SETTINGS, StructuredMeshSchema

TRANSFORMER = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)

app = typer.Typer(help="Convert an HDF5 topography surface to a VTK unstructured grid.")


def read_surface_file(
    surface_path: Path, scalar_key: str, flip: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(surface_path, "r") as f:
        latitude = np.array(f["latitude"])
        longitude = np.array(f["longitude"])
        scalars = np.array(f[scalar_key])

    if flip:
        # Swap convention if necessary (+z pointing up from sea level)
        scalars *= -1

    x_lon, x_lat = np.meshgrid(longitude, latitude)
    x, y = TRANSFORMER.transform(x_lon, x_lat)

    return x, y, scalars


@app.command()
def convert(
    surface: Annotated[
        Path,
        typer.Argument(
            help="Input HDF5 surface file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path, typer.Argument(help="Output VTK surface mesh path (e.g. .vtkhdf).")
    ],
    scalar_key: str = "elevation",
    flip: bool = True,
) -> None:
    """Entry point for the conversion."""
    x, y, scalars = read_surface_file(surface, scalar_key, flip)
    ni, nj = x.shape
    surface_mesh = StructuredMeshSchema.new(
        x=x, y=y, z=scalars, i=np.arange(ni), j=np.arange(nj), name=surface.stem
    )
    surface_mesh.to_zarr(
        output, encoding=DEFAULT_STRUCTURED_ENCODING_SETTINGS, mode="w"
    )


if __name__ == "__main__":
    app()
