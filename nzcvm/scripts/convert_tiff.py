from pathlib import Path
from typing import Annotated

import dask
import numpy as np
import typer
import xarray as xr

from nzcvm.models.mesh import (
    DEFAULT_STRUCTURED_ENCODING_SETTINGS,
    SURFACE_CHUNK_SIZE,
    StructuredMesh,
    regular_mesh,
)
from nzcvm.scripts._extras import missing_extra

app = typer.Typer()

#: Raster block read at once, in pixels along each axis.
READ_CHUNK_SIZE = 4096


def _valid_span(valid: xr.DataArray) -> slice:
    """The span from the first to the last True entry of 1-D *valid*."""
    index = np.flatnonzero(valid.values)
    return slice(index[0], index[-1] + 1)


def convert_tiff(
    name: str, tiff_model: xr.DataArray, downsample: int, flip: bool = False
) -> StructuredMesh:
    """Turn a north-up raster into a surface on an upright lattice.

    The result stays as lazy as *tiff_model*, so a chunked raster streams
    through to the output without loading whole.

    Parameters
    ----------
    name :
        Name of the surface.
    tiff_model :
        Raster with 1-D ``x`` and ``y`` pixel-centre coordinates.
    downsample :
        Average blocks of this many pixels along each axis.
    flip :
        Negate the values, turning elevation (up) into depth (down).

    Returns
    -------
    StructuredMesh
        The surface, with ``i`` along ``y`` and ``j`` along ``x``.
    """
    # Crop to the bounding box of the data. Dropping every all-NaN row or
    # column instead would leave gaps in the coordinates wherever no-data
    # rows sit inside the raster, and the grid would no longer be uniform.
    valid = tiff_model.notnull()
    rows, columns = dask.compute(valid.any("x"), valid.any("y"))
    clipped = tiff_model.isel(x=_valid_span(columns), y=_valid_span(rows))
    step_x = (clipped.x[1] - clipped.x[0]).item()
    step_y = (clipped.y[1] - clipped.y[0]).item()
    if downsample > 1:
        # The first block is always full, so its mean coordinate is the
        # centre of a full block, and the rest follow at the coarse step.
        # The padded last block averages fewer coordinates, which would put
        # its node off the lattice.
        origin = (
            clipped.x[:downsample].mean().item(),
            clipped.y[:downsample].mean().item(),
        )
        clipped = clipped.coarsen(x=downsample, y=downsample, boundary="pad").mean()
        step_x *= downsample
        step_y *= downsample
    else:
        origin = (clipped.x[0].item(), clipped.y[0].item())

    z = clipped.transpose("y", "x").astype(np.float32).data
    return regular_mesh(
        name, -z if flip else z, origin, step_i=(0.0, step_y), step_j=(step_x, 0.0)
    )


@app.command()
def main(
    tiff_path: Path,
    band: int,
    output_path: Path,
    downsample: int = 1,
    flip: Annotated[
        bool, typer.Option(help="Negate the values, as for elevation to depth.")
    ] = False,
) -> None:
    try:
        # Imported for its side effect: it registers xarray's "rasterio" engine.
        import rioxarray  # noqa: F401
    except ImportError as err:
        raise missing_extra("rioxarray", "raster", "convert-tiff") from err

    chunk = READ_CHUNK_SIZE * downsample
    dset = xr.open_dataset(
        tiff_path, engine="rasterio", chunks={"x": chunk, "y": chunk}
    )
    surface = convert_tiff(
        output_path.stem, dset["band_data"].sel(band=band), downsample, flip
    )
    surface = surface.chunk({"i": SURFACE_CHUNK_SIZE, "j": SURFACE_CHUNK_SIZE})
    surface.to_zarr(output_path, encoding=DEFAULT_STRUCTURED_ENCODING_SETTINGS)


if __name__ == "__main__":
    app()
