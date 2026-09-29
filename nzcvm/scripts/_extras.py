"""Helpful errors for subcommands whose dependencies live in an optional extra.

The base package doesn't install the heavy, single-purpose dependencies
(gmsh, rioxarray/GDAL, geopandas...), so each subcommand that needs one
imports it lazily and raises :func:`missing_extra` if it's absent. The base
``nzcvm`` command-line tool then still starts, and ``--help`` works, without
them.
"""


def missing_extra(package: str, extra: str, command: str) -> ImportError:
    """Build the error raised when an optional dependency isn't installed.

    Parameters
    ----------
    package : str
        The missing package's import name.
    extra : str
        The ``nzcvm`` extra that installs it.
    command : str
        The ``nzcvm`` subcommand that needs it.

    Returns
    -------
    ImportError
        An error that specifies the extra to install. Raise it ``from`` the
        original ``ImportError``.
    """
    return ImportError(
        f"{package} is required for `nzcvm {command}` but is not installed. "
        f"Install it with: pip install 'nzcvm[{extra}]' "
        f"(or `uv sync --extra {extra}` in a checkout)."
    )
