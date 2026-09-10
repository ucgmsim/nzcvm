import functools
import warnings
from dataclasses import dataclass, field

from mashumaro import field_options
from pyproj import CRS, Proj, Transformer

from nzcvm.config.core import ConfigObject
from nzcvm.config.validation import CRSStrategy, Latitude, Longitude
from nzcvm.coordinates import NZGD2000_EPSG, WGS84_EPSG


@dataclass(frozen=True)
class Projection(ConfigObject):
    """The projected CRS a grid's coordinates live in.

    Just enough to convert between geographic and projected coordinates.  A
    grid with no origin to rotate about, such as a set of boreholes, takes one
    of these on its own.  A grid placed at a model origin takes a
    :class:`Model` instead.

    Attributes
    ----------
    crs :
        Target projected CRS, such as ``EPSG:2193`` for NZTM2000.
    """

    crs: CRS = field(metadata=field_options(serialization_strategy=CRSStrategy()))

    @functools.cached_property
    def from_wgs84(self) -> Transformer:
        return Transformer.from_crs(WGS84_EPSG, self.crs, always_xy=True)

    @functools.cached_property
    def to_wgs84(self) -> Transformer:
        return Transformer.from_crs(self.crs, WGS84_EPSG, always_xy=True)

    def transformer_from(self, crs: CRS) -> Transformer:
        """Return a transformer from *crs* into this projection."""
        return Transformer.from_crs(crs, self.crs, always_xy=True)


@dataclass(frozen=True)
class Model(Projection):
    """A projection plus the origin and azimuth that place a grid.

    Attributes
    ----------
    origin_lon, origin_lat :
        Geographic origin of the local grid, in WGS84 degrees.
    azimuth :
        Clockwise rotation of the grid from true north, in degrees.
    """

    origin_lon: Longitude
    origin_lat: Latitude
    azimuth: float

    @functools.cached_property
    def origin(self) -> tuple[float, float]:
        return tuple(self.from_wgs84.transform(self.origin_lon, self.origin_lat))

    @property
    def grid_origin_x(self) -> float:
        return self.origin[0]

    @property
    def grid_origin_y(self) -> float:
        return self.origin[1]

    @functools.cached_property
    def grid_azimuth(self) -> float:
        projection = Proj(self.crs)

        geodetic_crs = self.crs.geodetic_crs

        datum_shift_implied = geodetic_crs is None or geodetic_crs.to_epsg() not in (
            # NZGD2000 is technically different, but the alignment is extremely
            # close and the warning would be annoying to see on every run
            # the code
            NZGD2000_EPSG,
            WGS84_EPSG,  # WGS84 itself
        )

        if datum_shift_implied:
            msg = (
                "Grid azimuth calculations assume a WGS84-compatible geodetic CRS. "
                "Because the target CRS utilises a different underlying geodetic datum, "
                "the calculated grid azimuth may experience slight skew due to orientation "
                "and ellipsoidal geometry differences between WGS84 and the target datum."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        wgs84_to_geodetic = Transformer.from_crs(
            WGS84_EPSG, geodetic_crs, always_xy=True
        )
        native_lon, native_lat = wgs84_to_geodetic.transform(
            self.origin_lon, self.origin_lat
        )

        meridian_convergence = projection.get_factors(
            native_lon, native_lat
        ).meridian_convergence

        return (self.azimuth - meridian_convergence) % 360
