from pathlib import Path

import numpy as np
import typer
import xarray as xr

from nzcvm.models.mesh import (
    DEFAULT_STRUCTURED_ENCODING_SETTINGS,
    StructuredMesh,
    StructuredMeshSchema,
)
from nzcvm.scripts._extras import missing_extra

app = typer.Typer()


def _valid_span(valid: xr.DataArray) -> slice:
    """The span from the first to the last True entry of 1-D *valid*."""
    index = np.flatnonzero(valid.values)
    return slice(index[0], index[-1] + 1)


def convert_tiff(
    name: str, tiff_model: xr.DataArray, downsample: int
) -> StructuredMesh:
    # Crop to the bounding box of the data. Dropping every all-NaN row or
    # column instead would leave gaps in the coordinates wherever no-data
    # rows sit inside the raster, and the grid would no longer be uniform.
    valid = tiff_model.notnull()
    clipped = tiff_model.isel(
        x=_valid_span(valid.any("y")), y=_valid_span(valid.any("x"))
    )
    if downsample > 1:
        coarse = clipped.coarsen(x=downsample, y=downsample, boundary="pad").mean()
        # The padded last block averages fewer coordinates, which would put
        # its node off the grid. Lay every coarse node out at the uniform
        # step from the first, which is always a full block.
        clipped = coarse.assign_coords(
            {
                dim: coarse[dim][0].item()
                + downsample
                * (clipped[dim][1] - clipped[dim][0]).item()
                * np.arange(coarse.sizes[dim])
                for dim in ("x", "y")
            }
        )

    xi = clipped.x.astype(np.float32).values
    yi = clipped.y.astype(np.float32).values
    z = clipped.astype(np.float32).values

    x, y = np.meshgrid(xi, yi)
    ni, nj = x.shape
    return StructuredMeshSchema.new(
        x=x, y=y, z=z, i=np.arange(ni), j=np.arange(nj), name=name
    )


@app.command()
def main(tiff_path: Path, band: int, output_path: Path, downsample: int = 1) -> None:
    try:
        # Imported for its side effect: it registers xarray's "rasterio" engine.
        import rioxarray  # noqa: F401
    except ImportError as err:
        raise missing_extra("rioxarray", "raster", "convert-tiff") from err

    dset = xr.open_dataset(tiff_path, engine="rasterio")
    surface = convert_tiff(
        output_path.stem, dset["band_data"].sel(band=band), downsample
    )
    surface.to_zarr(output_path, encoding=DEFAULT_STRUCTURED_ENCODING_SETTINGS)


if __name__ == "__main__":
    app()
