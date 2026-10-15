import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from nzcvm.config.core import ConfigObject
from nzcvm.config.grids.model import Model
from nzcvm.config.grids.terrain import Decay, LinearDecay, Solver
from nzcvm.config.validation import PositiveFloat
from nzcvm.coordinates import Coordinate

from .core import GridConfig


@dataclass
class MeshRefinement(ConfigObject):
    """Vertical mesh refinement for one depth layer.

    A refinement's name is its key in :attr:`SW4GridConfig.refinements`.

    Attributes
    ----------
    resolution :
        Horizontal and nominal vertical resolution in metres.
    bottom :
        Nominal depth of the base of this block, in metres (positive down).
        The block spans from the previous refinement's *bottom* (or the
        surface, for the first) down to this depth.
        :attr:`SW4GridConfig.decay` sets how the top block follows the
        topography.
    """

    # Horizontal and (nominal) vertical resolution.
    resolution: float
    bottom: float


DEFAULT_CHUNK_SIZES = {Coordinate.I: 128, Coordinate.J: 128}


@dataclass
class SW4GridConfig(GridConfig):
    """Horizontal and vertical grid configuration for the velocity model.

    Attributes
    ----------
    surface :
        Path to the topographic surface mesh file.  Used to translate depth
        to elevation.
    extent_x, extent_y :
        Physical extent of the grid in metres along each horizontal axis.
    azimuth :
        Clockwise rotation of the grid from north, in degrees.
    target_crs :
        Target projected CRS integer code, such as ``2193`` for NZTM2000.
    origin_lon, origin_lat :
        Geographic origin of the local grid in *origin_crs* (longitude,
        latitude).
    refinements :
        Ordered list of :class:`MeshRefinement` objects.  Must contain at
        least one entry. The *bottom* of the last entry sets the model bottom.
    decay :
        Unset (the default) for SW4 with topography: the first refinement
        stretches linearly between the surface and its flat bottom, and the
        sfile stores true positions.  Any decay instead builds an
        EMOD3D-compatible model for SW4 without topography: the decay
        distorts the earth into a flat nominal box, and the sfile stores
        nominal depths.  See :attr:`solver`.
    transpose :
        If ``True``, swap the I and J axes after applying the affine transform.
    origin_crs :
        CRS of the *origin_lon* / *origin_lat* values (default: ``4326``
        for WGS-84).
    origin_x, origin_y :
        Optional additional translation offset applied after the CRS
        transform (metres).
    """

    # Topographic surface path.
    surface: Path
    # Extents in x and y.
    extent_x: PositiveFloat
    extent_y: PositiveFloat

    orientation: Model

    # Mesh refinements.
    refinements: dict[str, MeshRefinement]

    decay: Decay | None = None

    transpose: bool = False

    chunks: dict[Coordinate, int] = field(default_factory=lambda: DEFAULT_CHUNK_SIZES)

    type: Literal["sw4"] = "sw4"

    def __post_init__(self):
        super().__post_init__()
        if not self.refinements:
            raise ValueError("SW4 grid needs at least one refinement.")
        # Sort by depth, as build_sw4 does: each block must sit below the one
        # above it and be exactly twice as coarse.
        refinements = sorted(
            self.refinements.items(), key=lambda refinement: refinement[1].bottom
        )
        for (above_name, above), (below_name, below) in itertools.pairwise(refinements):
            if below.bottom == above.bottom:
                raise ValueError(
                    f"Refinements {above_name!r} and {below_name!r} share a bottom"
                    f" ({above.bottom})."
                )
            if not np.isclose(below.resolution / above.resolution, 2.0):
                raise ValueError(
                    f"Refinement {below_name!r} (resolution {below.resolution}) must be"
                    f" twice as coarse as {above_name!r} above it"
                    f" (resolution {above.resolution})."
                )

        coarsest_grid_resolution = refinements[-1][1].resolution
        # Adjust extent x and y so it's divisible by the coarsest mesh refinement.
        self.extent_x = (
            np.round(self.extent_x / coarsest_grid_resolution)
            * coarsest_grid_resolution
        )
        self.extent_y = (
            np.round(self.extent_y / coarsest_grid_resolution)
            * coarsest_grid_resolution
        )

    @property
    def solver(self) -> Solver:
        """Physical without a decay, nominal with one."""
        return Solver.PHYSICAL if self.decay is None else Solver.NOMINAL

    def terrain_decay(self, default_length: float) -> Decay:
        """The decay profile this grid uses, resolved with *default_length*.

        Without a decay, SW4 with topography stretches the first refinement
        linearly between the surface and its flat bottom.
        """
        decay = LinearDecay() if self.decay is None else self.decay
        return decay.resolve(default_length)
