from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mashumaro import field_options
from pyproj import CRS

from nzcvm.config.core import ConfigObject
from nzcvm.config.grids.model import Projection
from nzcvm.config.validation import (
    CRSStrategy,
    GeographicCRS,
    Latitude,
    Longitude,
    NonEmptyStr,
    PositiveFloat,
)
from nzcvm.coordinates import WGS84_EPSG, Coordinate

from .core import GridConfig

DEFAULT_CHUNK_SIZES = {Coordinate.I: 64}


@dataclass
class Site(ConfigObject):
    """One borehole location, in the global CRS.

    Attributes
    ----------
    name :
        Label for the site.  The builder keeps it on the ``site`` coordinate
        of the grid, so the output reads back per station.
    longitude, latitude :
        Position in :attr:`BoreholeGridConfig.sites_crs`, which defaults to
        WGS84.  The grid builder projects it into the grid CRS.
    """

    name: NonEmptyStr
    longitude: Longitude
    latitude: Latitude


@dataclass
class BoreholeGridConfig(GridConfig):
    """Vertical profiles at a set of sites, at a fixed vertical resolution.

    A borehole grid holds independent columns, so an extent, an azimuth and a
    model origin have nothing to describe.  It takes a bare
    :class:`~nzcvm.config.grids.model.Projection` in place of the
    :class:`~nzcvm.config.grids.model.Model` orientation block that the
    extent-based grids use.

    The built grid has shape ``(len(sites), 1, nk)``: one column per site,
    a singleton ``j`` axis, and ``nk`` samples spaced *resolution_z* apart
    from the topography down to *depth*.

    Attributes
    ----------
    surface :
        Path to the topographic surface mesh file.  Used to translate depth
        to elevation, so each column starts at the ground.
    sites :
        Either an inline list of :class:`Site` objects, or a path to a CSV or
        Parquet file with ``name``, ``longitude`` and ``latitude`` columns.
    depth :
        Depth of the bottom of every column, in metres below the topography.
    resolution_z :
        Vertical sample spacing in metres.
    projection :
        Projected CRS to extract the profiles in.
    sites_crs :
        Geographic CRS of the site coordinates (default WGS84).  The builder
        maps each site from here into *projection* before querying.  It has to
        be geographic, since a site is a longitude and a latitude.

    Examples
    --------
    TOML, with the sites inline::

        [grid]
        type = "borehole"
        surface = "./synthetic/dem.zarr"
        depth = 500.0
        resolution_z = 25.0

        [grid.projection]
        crs = 'EPSG:2193'

        [[grid.sites]]
        name = "CACS"
        longitude = 172.62
        latitude = -43.53

    or read from a file::

        sites = "examples/sites.csv"
    """

    surface: Path

    sites: list[Site] | Path

    depth: PositiveFloat
    resolution_z: PositiveFloat

    projection: Projection

    sites_crs: GeographicCRS = field(
        default_factory=lambda: CRS.from_epsg(WGS84_EPSG),
        metadata=field_options(serialization_strategy=CRSStrategy()),
    )

    chunks: dict[Coordinate, int] = field(default_factory=lambda: DEFAULT_CHUNK_SIZES)

    type: Literal["borehole"] = "borehole"

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.sites, list) and not self.sites:
            raise ValueError("A borehole grid needs at least one site.")
