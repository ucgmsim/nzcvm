from dataclasses import dataclass, field
from pathlib import Path

from nzcvm.config.validation import PositiveFloat

from .core import DERIVED, LayerConfig


@dataclass
class ElyLayerConfig(LayerConfig):
    """Configuration DTO for an :class:`~nzcvm.layers.ely.ElyLayer`.

    Attributes
    ----------
    vs30 :
        Path to the Vs30 surface file.
    depth_t :
        Taper depth in metres (default ``450.0``).

    Examples
    --------
    TOML::

        [[layers]]
        type = "ely"
        vs30 = "path/to/vs30.h5"
        depth_t = 450.0
    """

    vs30: Path
    depth_t: PositiveFloat = 450.0
    type: str = "ely"
    requires: list[str] = field(
        default_factory=lambda: ["coastline"], init=False, metadata=DERIVED
    )
