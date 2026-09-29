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


def convert_tiff(
    name: str, tiff_model: xr.DataArray, downsample: int
) -> StructuredMesh:
    clipped = tiff_model.dropna("x", how="all").dropna("y", how="all")
    if downsample > 1:
        clipped = clipped.coarsen(  # ty: ignore[unresolved-attribute]
            x=downsample, y=downsample, boundary="pad"
        ).mean()

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
