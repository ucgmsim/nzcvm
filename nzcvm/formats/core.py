"""Registry of velocity-model output formats.

Each output format registers its writer with :func:`register_format`. Built-in
formats and third-party plugins (entry-point group ``nzcvm.formats``) register
the same way.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nzcvm.velocity_model import VelocityModel

type Writer = Callable[..., None]


@dataclass(frozen=True)
class OutputFormat:
    """A registered output format.

    Attributes
    ----------
    name :
        Name that selects the format, as in ``--format``.
    write :
        Called as ``write(velocity_model, path)``, plus
        ``quantise_arrays=...`` when *supports_quantisation* is true.
    extensions :
        Path suffixes (with the leading dot) that infer this format.
    supports_quantisation :
        Whether the writer accepts lossy array quantisation.
    """

    name: str
    write: Writer
    extensions: tuple[str, ...] = ()
    supports_quantisation: bool = False


#: Registered output formats, keyed by name.
FORMATS: dict[str, OutputFormat] = {}

#: Format for directories and paths without a suffix.
DIRECTORY_FORMAT = "emod3d"


def register_format(
    name: str,
    extensions: tuple[str, ...] = (),
    supports_quantisation: bool = False,
) -> Callable[[Writer], Writer]:
    """Register the decorated function as the writer for format *name*.

    Parameters
    ----------
    name :
        Name used to select the format.
    extensions :
        Path suffixes (with the leading dot) that infer this format.
    supports_quantisation :
        Whether the writer accepts a ``quantise_arrays`` keyword.

    Returns
    -------
    Callable
        Decorator returning the writer unchanged.

    Raises
    ------
    ValueError
        If *name* or one of *extensions* is already registered.

    Examples
    --------
    >>> from nzcvm.formats import FORMATS, register_format
    >>> @register_format("example", extensions=(".example",))
    ... def to_example(velocity_model, path):
    ...     ...
    >>> FORMATS.pop("example").extensions
    ('.example',)
    """

    def decorator(write: Writer) -> Writer:
        if name in FORMATS:
            raise ValueError(f"Output format {name!r} is already registered.")
        for registered in FORMATS.values():
            if clash := set(extensions) & set(registered.extensions):
                raise ValueError(
                    f"Extensions {sorted(clash)} already infer format {registered.name!r}."
                )
        FORMATS[name] = OutputFormat(name, write, extensions, supports_quantisation)
        return write

    return decorator


def from_path(path: Path) -> str:
    """Infer the output format name from a path extension.

    Directories (and paths with no suffix) default to ``emod3d``.

    Parameters
    ----------
    path :
        Output path whose suffix determines the format.

    Returns
    -------
    str
        The name of the inferred format.

    Raises
    ------
    ValueError
        If the extension isn't recognised.

    Examples
    --------
    >>> from pathlib import Path
    >>> from nzcvm.formats import from_path
    >>> from_path(Path("model.h5"))
    'netcdf'
    """
    for output_format in FORMATS.values():
        if path.suffix in output_format.extensions:
            return output_format.name
    if path.is_dir() or not path.suffix:
        return DIRECTORY_FORMAT
    raise ValueError(f"Could not infer a format for {path=}")


def write_velocity_model(
    velocity_model: "VelocityModel",
    path: Path,
    format: str | None = None,
    quantise_arrays: bool = True,
) -> None:
    """Write *velocity_model* to *path* in the given *format*.

    Parameters
    ----------
    velocity_model :
        Velocity model populated by the query pipeline.
    path :
        Destination file or directory path.
    format :
        Name of a registered output format, or ``None`` to infer one from
        *path* with :func:`from_path`.
    quantise_arrays :
        If True, quantise the velocity model output for formats that support it.

    Raises
    ------
    ValueError
        If *format* isn't registered, or doesn't support quantisation when
        *quantise_arrays* is true.
    """
    name = format or from_path(path)
    if name not in FORMATS:
        raise ValueError(
            f"Unknown output format {name!r}, expected one of {sorted(FORMATS)}."
        )
    output_format = FORMATS[name]

    if output_format.supports_quantisation:
        output_format.write(velocity_model, path, quantise_arrays=quantise_arrays)
    elif quantise_arrays:
        raise ValueError(
            f"Lossy array quantisation isn't supported by the {name!r} format."
        )
    else:
        output_format.write(velocity_model, path)
