"""Convert an HDF5 surface file into a surface on an upright NZTM lattice."""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from nzcvm.models.mesh import (
    DEFAULT_STRUCTURED_ENCODING_SETTINGS,
    SURFACE_CHUNK_SIZE,
    StructuredMesh,
    surface_mesh,
)
from nzcvm.models.surface import SurfaceGrid, median_spacing, read_surface_file

app = typer.Typer(help="Convert an HDF5 surface into a surface on an upright lattice.")


def resample_upright(
    name: str, grid: SurfaceGrid, spacing: float | None = None
) -> StructuredMesh:
    """Sample *grid* on an upright lattice covering its footprint.

    Projecting a latitude/longitude grid into NZTM curves it, so it can't
    sit on a lattice as it stands. This lays an axis-aligned lattice over the
    grid's bounding box and interpolates the grid bilinearly at each node,
    leaving NaN at nodes outside it.

    Parameters
    ----------
    name :
        Name of the surface.
    grid :
        Values on the projected source grid.
    spacing :
        Node spacing in metres. Defaults to the median spacing of the
        source nodes.

    Returns
    -------
    StructuredMesh
        The resampled surface, indexed ``(y, x)`` by NZTM coordinates.
    """
    x = np.asarray(grid.x, dtype=np.float64)
    y = np.asarray(grid.y, dtype=np.float64)
    if spacing is None:
        spacing = median_spacing(grid)
    origin = (x.min(), y.min())
    node_x = origin[0] + spacing * np.arange(
        int(np.ceil((x.max() - origin[0]) / spacing)) + 1
    )
    node_y = origin[1] + spacing * np.arange(
        int(np.ceil((y.max() - origin[1]) / spacing)) + 1
    )
    grid_x, grid_y = np.meshgrid(node_x, node_y)
    values, _ = grid(grid_x.ravel(), grid_y.ravel(), clamp=False)
    return surface_mesh(
        name, values.reshape(grid_x.shape).astype(np.float32), node_x, node_y
    )


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
    output: Annotated[Path, typer.Argument(help="Output surface path (.zarr).")],
    scalar_key: str = "elevation",
    flip: bool = True,
    spacing: Annotated[
        float | None,
        typer.Option(
            help="Lattice spacing in metres. Defaults to the source's median spacing.",
            min=0.0,
        ),
    ] = None,
) -> None:
    """Entry point for the conversion."""
    grid = read_surface_file(surface, scalar_key, flip)
    surface_mesh = resample_upright(surface.stem, grid, spacing).chunk(
        {"y": SURFACE_CHUNK_SIZE, "x": SURFACE_CHUNK_SIZE}
    )
    surface_mesh.to_zarr(
        output, encoding=DEFAULT_STRUCTURED_ENCODING_SETTINGS, mode="w"
    )


if __name__ == "__main__":
    app()
