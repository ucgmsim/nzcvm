from dataclasses import dataclass, field
from pathlib import Path

from nzcvm.config.layers import LayerConfig
from nzcvm.config.layers.core import DERIVED


@dataclass
class CoastlineConfig(LayerConfig):
    coastline: Path
    provides: list[str] = field(
        default_factory=lambda: ["coastline"], init=False, metadata=DERIVED
    )
    type: str = "coastline"
