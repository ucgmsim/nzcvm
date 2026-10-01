from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .core import LayerConfig


@dataclass
class QueryLayerConfig(LayerConfig):
    """Configuration DTO for a :class:`~nzcvm.layers.query.QueryLayer`.

    Specifies where to find the velocity-model mesh files.  *model_path*
    and *model_globs* together identify the set of ``*.zarr`` mesh stores to load.

    Attributes
    ----------
    model_path :
        Directory containing the mesh files.
    model_globs :
        List of glob patterns used to find mesh files under *model_path*
        (default ``["*.zarr"]``).  The layer loads every file matching any
        of the patterns.

    Examples
    --------
    TOML::

        [[layers]]
        type = "query"
        model_path = "path/to/models"
        model_globs = ["*.zarr"]
    """

    model_path: Path
    model_globs: list[str] = field(default_factory=lambda: ["*.zarr"])
    type: str = "query"
