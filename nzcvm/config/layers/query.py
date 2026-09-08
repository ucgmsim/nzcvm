from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .core import LayerConfig


@dataclass
class QueryLayerConfig(LayerConfig):
    """Configuration DTO for a :class:`~nzcvm.layers.query.QueryLayer`.

    Specifies where to find the velocity-model mesh files.  *model_path*
    and *model_globs* together identify the set of ``*.vtkhdf`` files to load.

    Attributes
    ----------
    model_path :
        Directory containing the mesh files.
    model_globs :
        List of glob patterns used to find mesh files under *model_path*
        (default ``["*.vtkhdf"]``).  Files matched by any of the patterns
        are loaded.

    Examples
    --------
    TOML::

        [[layers]]
        type = "query"
        model_path = "path/to/models"
        model_globs = ["*.vtkhdf"]
    """

    model_path: Path
    model_globs: list[str] = field(default_factory=lambda: ["*.vtkhdf"])
    type: str = "query"
