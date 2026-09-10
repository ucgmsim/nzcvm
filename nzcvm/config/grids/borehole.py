from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mashumaro import field_options
from pyproj import CRS

from nzcvm.config.core import ConfigObject
from nzcvm.config.grids.model import Projection
from nzcvm.config.validation import (
    CRSStrategy,
    GeographicCRS,
    Latitude,
    Longitude,
    PositiveFloat,
)
from nzcvm.coordinates import WGS84_EPSG, Coordinate

from .core import GridConfig

DEFAULT_CHUNK_SIZES = {Coordinate.I: 64}


#: The keys that place a site. Everything else given for a site is a label.
SPATIAL_KEYS = ("longitude", "latitude")


@dataclass
class Site(ConfigObject):
    """One borehole location, in the global CRS, plus whatever labels it has.

    Longitude and latitude place the site, and the config doesn't reserve any
    other key, so a site takes as much or as little description as the caller
    has to give it. Each label becomes a coordinate on the grid's ``i`` axis,
    which the writers keep.

    Attributes
    ----------
    longitude, latitude :
        Position in :attr:`BoreholeGridConfig.sites_crs`, which defaults to
        WGS84.  The grid builder projects it into the grid CRS.
    labels :
        Everything else given for the site.  Decoding a config folds every
        key except :data:`SPATIAL_KEYS` in here, so a config file needn't
        spell the mapping out.

    Examples
    --------
    >>> Site.from_dict(
    ...     {"longitude": 172.15, "latitude": -43.7, "site": "GULL", "network": "NZ"}
    ... )
    Site(longitude=172.15, latitude=-43.7, labels={'site': 'GULL', 'network': 'NZ'})
    """

    longitude: Longitude
    latitude: Latitude
    labels: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def __pre_deserialize__(cls, d: dict[str, Any]) -> dict[str, Any]:
        """Fold every key that doesn't place the site into :attr:`labels`.

        Named *d* to match the mashumaro hook this overrides.
        """
        return {
            **{k: v for k, v in d.items() if k in SPATIAL_KEYS},
            "labels": {k: v for k, v in d.items() if k not in SPATIAL_KEYS},
        }


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
        Parquet file with ``longitude`` and ``latitude`` columns.  Any other
        key or column labels the site.
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
    keep_extra_columns :
        Whether the builder puts the site labels on the grid, and so in the
        output (default ``True``).  Set it to ``False`` to keep only the
        spatial coordinates and drop the rest.

    Examples
    --------
    TOML, with the sites inline.  Neither ``site`` nor ``network`` is a
    keyword here, and both end up in the output because nothing reserves
    them::

        [grid]
        type = "borehole"
        surface = "./synthetic/dem.zarr"
        depth = 500.0
        resolution_z = 25.0

        [grid.projection]
        crs = 'EPSG:2193'

        [[grid.sites]]
        longitude = 172.62
        latitude = -43.53
        site = "CACS"
        network = "NZ"

    or read from a file, where the extra columns do the same job::

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

    keep_extra_columns: bool = True

    chunks: dict[Coordinate, int] = field(default_factory=lambda: DEFAULT_CHUNK_SIZES)

    type: Literal["borehole"] = "borehole"

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.sites, list) and not self.sites:
            raise ValueError("A borehole grid needs at least one site.")
