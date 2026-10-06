from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Literal

from nzcvm.config.grids import GridConfig
from nzcvm.config.grids.model import Model
from nzcvm.config.grids.terrain import Decay, SquashedDecay, TaperedDecay
from nzcvm.config.validation import (
    PositiveFloat,
    PositiveInt,
)
from nzcvm.coordinates import Coordinate

DEFAULT_CHUNK_SIZES = {Coordinate.I: 128, Coordinate.J: 128}


class TopographyType(StrEnum):
    """EMOD3D's named topography types, shorthands for a :class:`Decay`."""

    SQUASHED = auto()
    SQUASHED_TAPERED = auto()

    def decay(self) -> Decay:
        match self:
            case TopographyType.SQUASHED:
                return SquashedDecay()
            case TopographyType.SQUASHED_TAPERED:
                return TaperedDecay(ratio=1.0)


@dataclass
class EMOD3DGrid(GridConfig):
    """A single-block EMOD3D grid with ``nz`` levels at a fixed nominal spacing.

    Set exactly one of *topo_type* (the EMOD3D names) or *decay* (any
    :class:`~nzcvm.config.grids.terrain.Decay`).  A decay with an unset
    length decays over the whole grid.
    """

    # Topographic surface path.
    surface: Path

    nx: PositiveInt
    ny: PositiveInt
    nz: PositiveInt

    resolution: PositiveFloat

    # Coordinate metadata
    orientation: Model

    topo_type: TopographyType | None = None
    decay: Decay | None = None

    chunks: dict[Coordinate, int] = field(default_factory=lambda: DEFAULT_CHUNK_SIZES)

    type: Literal["emod3d"] = "emod3d"

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.topo_type is None) == (self.decay is None):
            raise ValueError("Set exactly one of topo_type or decay.")

    def terrain_decay(self, default_length: float) -> Decay:
        """The decay profile this grid uses, resolved with *default_length*."""
        if self.decay is not None:
            return self.decay.resolve(default_length)
        assert self.topo_type is not None
        return self.topo_type.decay().resolve(default_length)
