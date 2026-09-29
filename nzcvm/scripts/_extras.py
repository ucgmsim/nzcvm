"""Helpful errors for subcommands whose dependencies live in an optional extra.

The heavy, single-purpose dependencies (gmsh, rioxarray/GDAL, geopandas...)
are not installed with the base package, so each subcommand that needs one
imports it lazily and raises :func:`missing_extra` if it is absent. The base
``nzcvm`` CLI then still starts, and ``--help`` works, without them.
"""


def missing_extra(package: str, extra: str, command: str) -> ImportError:
    """Build the error raised when an optional dependency is not installed.

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
        An error naming the extra to install, to be raised ``from`` the
        original ``ImportError``.
    """
    return ImportError(
        f"{package} is required for `nzcvm {command}` but is not installed. "
        f"Install it with: pip install 'nzcvm[{extra}]' "
        f"(or `uv sync --extra {extra}` in a checkout)."
    )
