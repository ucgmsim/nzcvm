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
        Total depth/thickness of the model in metres. The invariant maintained
        is that the bottom surface perfectly follows the top surface at this depth.
    resolution_x, resolution_y, resolution_z :
        Spatially independent grid resolutions in metres.
    azimuth :
        Clockwise rotation of the grid from north, in degrees.
    target_crs :
        Target projected CRS integer code (e.g. ``2193`` for NZTM2000).
    origin_lon, origin_lat :
        Geographic origin of the local grid in *origin_crs* (longitude, latitude).
    transpose :
        If ``True``, swap the I and J axes after applying the affine transform.
    origin_crs :
        CRS of the *origin_lon* / *origin_lat* values (default: ``4326`` for WGS-84).
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

    transpose: bool = False

    chunks: dict[Coordinate, int] = field(default_factory=lambda: DEFAULT_CHUNK_SIZES)

    type: Literal["regular"] = "regular"
