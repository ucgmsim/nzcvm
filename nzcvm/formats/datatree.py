from enum import Enum
from pathlib import Path
from typing import Any

import hdf5plugin
import shapely
import xarray as xr
from numcodecs import ZFPY, Blosc

from nzcvm.formats import quantise
from nzcvm.formats.core import register_format
from nzcvm.velocity_model import VelocityModel
from nzcvm.xarray import encode


# TODO: Eliminate in favour of encode architecture
def _coerce_attribute_value_to_netcdf(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.value
    elif isinstance(v, shapely.Geometry):
        return shapely.to_wkt(v)
    return v


def _translate_compressor_to_hdf5(dset: xr.Dataset) -> xr.Dataset:
    dset = dset.copy(deep=False)

    for var_name in dset.data_vars:
        encoding = dset[var_name].encoding
        compressor = encoding.get("compressors")

        if compressor is not None:
            compressor = compressor[0]

        if isinstance(compressor, ZFPY):
            kwargs = {}
            if compressor.tolerance > 0:
                kwargs["accuracy"] = compressor.tolerance
            elif compressor.rate > 0:
                kwargs["rate"] = compressor.rate
            elif compressor.precision > 0:
                kwargs["precision"] = compressor.precision

            encoding.update(hdf5plugin.Zfp(**kwargs))
            encoding.pop("compressor", None)

        elif isinstance(compressor, Blosc):
            encoding.update(
                hdf5plugin.Blosc(
                    cname=compressor.cname,
                    clevel=compressor.clevel,
                    shuffle=compressor.shuffle,
                )
            )
            encoding.pop("compressor", None)

    return dset


def _normalise_dataset_attributes(dset: xr.Dataset) -> xr.Dataset:
    dset = dset.copy(deep=False)
    attributes = {
        k: _coerce_attribute_value_to_netcdf(v)
        for k, v in dset.attrs.items()
        if (v != 0 and v)
    }
    if "refinements" in attributes:
        attributes.pop("refinements")
    dset.attrs = attributes
    return dset


@register_format("netcdf", extensions=(".h5",), supports_quantisation=True)
def to_netcdf(
    velocity_model: VelocityModel, path: Path, quantise_arrays: bool = True
) -> None:
    hdf5plugin.register(("zfp", "blosc"))
    dtree = velocity_model.to_datatree()
    dtree = dtree.map_over_datasets(_normalise_dataset_attributes)
    if quantise_arrays:
        dtree = quantise.apply_compression(dtree, settings=quantise.DEFAULT_PRECISION)
        dtree = dtree.map_over_datasets(_translate_compressor_to_hdf5)
    dtree.to_netcdf(path, engine="h5netcdf", mode="w")


@register_format("zarr", extensions=(".zarr",))
def to_zarr(velocity_model: VelocityModel, path: Path) -> None:
    dtree = velocity_model.to_datatree()
    dtree = encode(dtree, attr_hook=_coerce_attribute_value_to_netcdf)
    dtree.to_zarr(path, mode="w")
