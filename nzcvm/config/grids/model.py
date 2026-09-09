import functools
import warnings
from dataclasses import dataclass, field

from mashumaro import field_options
from pyproj import CRS, Proj, Transformer

from nzcvm.config.core import ConfigObject
from nzcvm.config.validation import CRSStrategy, Latitude, Longitude
from nzcvm.coordinates import NZGD2000_EPSG, WGS84_EPSG


@dataclass(frozen=True)
class Model(ConfigObject):
    origin_lon: Longitude
    origin_lat: Latitude
    azimuth: float
    crs: CRS = field(metadata=field_options(serialization_strategy=CRSStrategy()))

    @functools.cached_property
    def from_wgs84(self) -> Transformer:
        return Transformer.from_crs(WGS84_EPSG, self.crs, always_xy=True)

    @functools.cached_property
    def to_wgs84(self) -> Transformer:
        return Transformer.from_crs(self.crs, WGS84_EPSG, always_xy=True)

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
