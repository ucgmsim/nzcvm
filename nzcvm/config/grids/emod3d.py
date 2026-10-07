from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from nzcvm.config.grids import GridConfig
from nzcvm.config.grids.model import Model
from nzcvm.config.grids.terrain import Decay
from nzcvm.config.validation import (
    PositiveFloat,
    PositiveInt,
)
from nzcvm.coordinates import Coordinate

DEFAULT_CHUNK_SIZES = {Coordinate.I: 128, Coordinate.J: 128}


@dataclass
class EMOD3DGrid(GridConfig):
    """A single-block EMOD3D grid with ``nz`` levels at a fixed nominal spacing.

    *decay* is any :class:`~nzcvm.config.grids.terrain.Decay`.  A decay with
    an unset length decays over the whole grid.  EMOD3D's ``squashed`` and
    ``squashed_tapered`` topography types are ``squashed`` and ``tapered``
    with a ratio of 1.
    """

    # Topographic surface path.
    surface: Path

    nx: PositiveInt
    ny: PositiveInt
    nz: PositiveInt

    resolution: PositiveFloat

    # Coordinate metadata
    orientation: Model

    decay: Decay

    chunks: dict[Coordinate, int] = field(default_factory=lambda: DEFAULT_CHUNK_SIZES)

    type: Literal["emod3d"] = "emod3d"

    def terrain_decay(self, default_length: float) -> Decay:
        """The decay profile this grid uses, resolved with *default_length*."""
        return self.decay.resolve(default_length)
