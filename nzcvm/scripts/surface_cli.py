"""Convert an HDF5 topography file to a VTK UnstructuredGrid (VTKHDF compatible)."""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from nzcvm.models.mesh import DEFAULT_STRUCTURED_ENCODING_SETTINGS, StructuredMeshSchema
from nzcvm.models.surface import read_surface_file

app = typer.Typer(help="Convert an HDF5 topography surface to a VTK unstructured grid.")


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
    grid = read_surface_file(surface, scalar_key, flip)
    ni, nj = grid.x.shape
    surface_mesh = StructuredMeshSchema.new(
        x=grid.x,
        y=grid.y,
        z=grid.values,
        i=np.arange(ni),
        j=np.arange(nj),
        name=surface.stem,
    )
    surface_mesh.to_zarr(
        output, encoding=DEFAULT_STRUCTURED_ENCODING_SETTINGS, mode="w"
    )


if __name__ == "__main__":
    app()
