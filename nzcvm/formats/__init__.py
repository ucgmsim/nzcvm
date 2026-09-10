"""Output format selection and velocity-model serialisation.

:class:`Format` enumerates supported output formats.  :func:`from_path`
infers a format from a path, and :func:`write_velocity_model` dispatches
to the appropriate writer.
"""

from enum import StrEnum, auto
from pathlib import Path

from nzcvm.velocity_model import VelocityModel

from . import csv, datatree, emod3d, sfile


class Format(StrEnum):
    """Supported output format for a velocity model.

    ``INFERRED`` defers the choice to :func:`from_path`.

    Examples
    --------
    >>> from nzcvm.formats import Format
    >>> Format.EMOD3D == "emod3d"
    True
    """

    INFERRED = auto()
    EMOD3D = auto()
    SFILE = auto()
    NETCDF = auto()
    ZARR = auto()
    CSV = auto()


def from_path(path: Path) -> Format:
    """Infer the output :class:`Format` from a path extension.

    Directories (and paths with no suffix) default to ``EMOD3D``.

    Parameters
    ----------
    path :
        Output path whose suffix determines the format.

    Returns
    -------
    Format
        The inferred format for the path suffix.

    Raises
    ------
    ValueError
        If the extension isn't recognised.

    Examples
    --------
    >>> from pathlib import Path
    >>> from nzcvm.formats import from_path, Format
    >>> from_path(Path("model.h5"))
    <Format.NETCDF: 'netcdf'>
    """
    format_map = {
        ".sfile": Format.SFILE,
        ".h5": Format.NETCDF,
        ".zarr": Format.ZARR,
        ".csv": Format.CSV,
    }
    ext = path.suffix

    if ext in format_map:
        return format_map[ext]
    elif path.is_dir() or not path.suffix:
        return Format.EMOD3D
    else:
        raise ValueError(f"Could not infer a format for {path=}")


def write_velocity_model(
    velocity_model: VelocityModel,
    path: Path,
    format: Format,
    quantise_arrays: bool = True,
) -> None:
    """Write *velocity_model* to *path* in the given *format*.

    Parameters
    ----------
    velocity_model : DataTree
        Populated :class:`xarray.DataTree` produced by the query pipeline.
    path : Path
        Destination file or directory path.
    format : Format
        Output format. Use :func:`from_path` to infer one from the extension.
    quantise : bool
        If True, quantise the velocity model output for formats that support it
    """
    if format == Format.INFERRED:
        format = from_path(path)

    if quantise_arrays and format != Format.NETCDF:
        raise ValueError(
            "Lossy array quantisation is only supported with the NetCDF format."
        )

    match format:
        case Format.EMOD3D:
            emod3d.to_emod3d(velocity_model, path)
        case Format.SFILE:
            sfile.to_sfile(velocity_model, path)
        case Format.NETCDF:
            datatree.to_netcdf(velocity_model, path, quantise_arrays)
        case Format.ZARR:
            datatree.to_zarr(velocity_model, path)
        case Format.CSV:
            csv.to_csv(velocity_model, path)
