from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from nzcvm.config.grids.model import Model
from nzcvm.config.validation import PositiveFloat
from nzcvm.coordinates import Coordinate

from .core import GridConfig

DEFAULT_CHUNK_SIZES = {Coordinate.I: 128, Coordinate.J: 128}


@dataclass
class RegularGridConfig(GridConfig):
    """Horizontal and vertical grid configuration for a topography-following model.

    Unlike SW4, this model uses spatially variable resolutions for X, Y, and Z,
    and maintains a fixed Z resolution across the entire domain. The bottom of
    the model perfectly mirrors the topographic surface at a constant depth.

    Attributes
    ----------
    surface :
        Path to the topographic surface mesh file. Used to translate depth
        to elevation.
    extent_x, extent_y :
        Physical extent of the grid in metres along each horizontal axis.
    thickness :
        Total depth/thickness of the model in metres, at least *resolution_z*.
        The bottom surface perfectly follows the top surface at this depth,
        rounded up to the next multiple of *resolution_z*.
    resolution_x, resolution_y, resolution_z :
        Spatially independent grid resolutions in metres.
    orientation :
        Origin, azimuth and projected CRS that place the grid on the earth.
    chunks :
        Dask chunk sizes along the horizontal axes.
    """

    # Topographic surface path.
    surface: Path

    # Extents and total thickness
    extent_x: PositiveFloat
    extent_y: PositiveFloat
    thickness: PositiveFloat

    # Independent spatial resolutions
    resolution_x: PositiveFloat
    resolution_y: PositiveFloat
    resolution_z: PositiveFloat

    orientation: Model

    chunks: dict[Coordinate, int] = field(default_factory=lambda: DEFAULT_CHUNK_SIZES)

    type: Literal["regular"] = "regular"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.thickness < self.resolution_z:
            raise ValueError(
                f"thickness ({self.thickness}) must be at least resolution_z"
                f" ({self.resolution_z})."
            )
