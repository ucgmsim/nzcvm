"""Output format selection and velocity-model serialisation.

:data:`FORMATS` maps each registered output format's name to its writer.  :func:`from_path`
infers a format from a path, and :func:`write_velocity_model` dispatches
to the appropriate writer.  :func:`register_format` adds a new format; third
parties can do so from the ``nzcvm.formats`` entry-point group.
"""

from nzcvm import plugins

from . import datatree, emod3d, sfile, table
from .core import (
    FORMATS,
    OutputFormat,
    from_path,
    register_format,
    write_velocity_model,
)

plugins.load_plugins(plugins.FORMATS)

__all__ = [
    "FORMATS",
    "OutputFormat",
    "datatree",
    "emod3d",
    "from_path",
    "register_format",
    "sfile",
    "table",
    "write_velocity_model",
]
