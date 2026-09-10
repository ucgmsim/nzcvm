"""Analytic stand-ins for the data NZCVM normally reads off a data root.

Every function here is a closed-form expression evaluated over a small
rectangular patch of Canterbury coastline (:data:`DOMAIN`).  Nothing is
random and nothing comes off the network, so a synthetic data root rebuilds
byte-identically on any machine, and a test that asserts on a velocity
profile can work out the expected answer by hand.

The synthetic world is deliberately simple:

* Elevation falls linearly west to east, crossing sea level at
  :data:`COASTLINE_U`, with a sinusoidal ridge riding on top.  Land lies in
  the west, open water in the east.
* Vs30 tracks elevation, so the coastal plain is soft and the hills are stiff.
* The basins in :data:`BASINS` sit inland, each a paraboloid bowl that closes
  to zero thickness on its own outline.
* A tomography block fills the rest of the volume with a velocity that
  increases with depth.

:mod:`nzcvm.scripts.synthetic` wraps these in a command-line tool that writes
the file formats the rest of the toolkit consumes, and ``just synthetic``
drives it end to end.

Examples
--------
>>> float(elevation(172.0, -43.6).round(1))   # inland, west of the coast
550.0
>>> float(elevation(172.5, -43.6).round(1))   # offshore, east of the coast
-283.3
"""

from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
from pyproj import Transformer

from nzcvm.coordinates import Affine
from nzcvm.scripts.convert_tomography import CRS_NZTM, CRS_WGS, MODEL_COLUMNS, ModelType

#: Metres of elevation lost across the full width of the domain.
SLOPE = 1000.0
#: Fractional distance across the domain at which elevation crosses sea level.
COASTLINE_U = 0.75
#: Height of the ridges superimposed on the coastal slope, in metres.
RIDGE_AMPLITUDE = 200.0
#: Number of ridge half-periods spanning the domain from south to north.
RIDGE_PERIODS = 3.0

#: Vs30 on the shoreline and on the highest ground, in m/s.
VS30_MIN = 200.0
VS30_MAX = 700.0
#: Elevation, in metres, at which Vs30 saturates at :data:`VS30_MAX`.
VS30_PLATEAU = 800.0

#: Vp at sea level and its gradient with depth, in km/s and km/s per km.
TOMOGRAPHY_VP = 2.5
TOMOGRAPHY_VP_GRADIENT = 0.35
#: Relative amplitude of the lateral Vp variation, and its wavelength in km.
TOMOGRAPHY_UNDULATION = 0.05
TOMOGRAPHY_WAVELENGTH = 40.0
VP_VS_RATIO = 1.8

#: Top of the tomography block, in km over sea level. Chosen to clear the
#: highest topography.
TOMOGRAPHY_HEADROOM = 1.5


@dataclass(frozen=True)
class Domain:
    """A rectangle of WGS84 longitude and latitude.

    The synthetic fields are all written against the normalised coordinates
    ``(u, v)``, which run from 0 to 1 across the rectangle, so moving or
    resizing the domain stretches the whole world with it.
    """

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def normalise(
        self, lon: npt.ArrayLike, lat: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map longitude and latitude onto the unit square.

        Parameters
        ----------
        lon, lat :
            Geographic coordinates in degrees.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(u, v)``, zero at the south-west corner and one at the
            north-east corner.  Points outside the domain fall outside
            ``[0, 1]`` rather than clipping to it.

        Examples
        --------
        >>> u, v = DOMAIN.normalise(172.3, -43.6)
        >>> float(u.round(3)), float(v.round(3))
        (0.5, 0.5)
        """
        u = (np.asarray(lon, dtype=np.float64) - self.lon_min) / (
            self.lon_max - self.lon_min
        )
        v = (np.asarray(lat, dtype=np.float64) - self.lat_min) / (
            self.lat_max - self.lat_min
        )
        return u, v

    def denormalise(
        self, u: npt.ArrayLike, v: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invert :meth:`normalise`, mapping the unit square back to degrees."""
        lon = self.lon_min + np.asarray(u, dtype=np.float64) * (
            self.lon_max - self.lon_min
        )
        lat = self.lat_min + np.asarray(v, dtype=np.float64) * (
            self.lat_max - self.lat_min
        )
        return lon, lat

    def sample(self, n_lon: int, n_lat: int) -> tuple[np.ndarray, np.ndarray]:
        """Return 1-D longitude and latitude axes spanning the domain.

        Parameters
        ----------
        n_lon, n_lat :
            Number of samples along each axis.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Ascending ``(longitude, latitude)`` axes, endpoints included.
        """
        return (
            np.linspace(self.lon_min, self.lon_max, n_lon),
            np.linspace(self.lat_min, self.lat_max, n_lat),
        )

    @property
    def polygon(self) -> shapely.Polygon:
        """The domain rectangle, in WGS84."""
        return shapely.box(self.lon_min, self.lat_min, self.lon_max, self.lat_max)


#: A patch of the Canterbury coast roughly 48 km east-west by 44 km north-south.
DOMAIN = Domain(lon_min=172.0, lon_max=172.6, lat_min=-43.8, lat_max=-43.4)


def elevation(
    lon: npt.ArrayLike, lat: npt.ArrayLike, domain: Domain = DOMAIN
) -> np.ndarray:
    """Synthetic topography, in metres of elevation, positive up.

    A plain sloping into the sea, with a north-south corrugation on top.

    Parameters
    ----------
    lon, lat :
        Geographic coordinates in degrees.
    domain :
        Rectangle holding the field.

    Returns
    -------
    numpy.ndarray
        Elevation in metres, positive up.  Negative offshore.
    """
    u, v = domain.normalise(lon, lat)
    ridges = RIDGE_AMPLITUDE * np.sin(RIDGE_PERIODS * np.pi * v)
    return SLOPE * (COASTLINE_U - u) + ridges


def vs30(lon: npt.ArrayLike, lat: npt.ArrayLike, domain: Domain = DOMAIN) -> np.ndarray:
    """Synthetic Vs30, in m/s, ramping from soft coast to stiff hills.

    Parameters
    ----------
    lon, lat :
        Geographic coordinates in degrees.
    domain :
        Rectangle holding the field.

    Returns
    -------
    numpy.ndarray
        Vs30 between :data:`VS30_MIN` and :data:`VS30_MAX`.

    Examples
    --------
    >>> float(vs30(172.6, -43.6))   # offshore, so the soft-sediment floor
    200.0
    """
    height = elevation(lon, lat, domain)
    ramp = np.clip(height / VS30_PLATEAU, 0.0, 1.0)
    return VS30_MIN + (VS30_MAX - VS30_MIN) * ramp


def coastline_u(v: npt.ArrayLike) -> np.ndarray:
    """Normalised easting of the shoreline at each normalised northing.

    The shoreline is the zero contour of :func:`elevation`, which the
    linear slope makes solvable in closed form.

    Parameters
    ----------
    v :
        Normalised northing, zero at the south edge of the domain.

    Returns
    -------
    numpy.ndarray
        Normalised easting ``u`` where elevation crosses zero.
    """
    v = np.asarray(v, dtype=np.float64)
    return COASTLINE_U + (RIDGE_AMPLITUDE / SLOPE) * np.sin(RIDGE_PERIODS * np.pi * v)


def land(domain: Domain = DOMAIN, pad: float = 1.0, samples: int = 256) -> np.ndarray:
    """Vertices of the land polygon, in WGS84 degrees.

    Only the eastern edge follows the shoreline.  The other three run *pad*
    domain widths beyond the domain, far enough that a grid inside the domain
    never mistakes the edge of the polygon for a coast.

    Parameters
    ----------
    domain :
        Rectangle holding the field.
    pad :
        How far to extend the polygon past the domain, in domain widths.
    samples :
        Number of points used to trace the shoreline.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` array of longitude, latitude vertices, closed and wound
        anticlockwise.
    """
    v = np.linspace(-pad, 1.0 + pad, samples)
    shore_u = coastline_u(v)
    # South-west corner, north up the shoreline, then back around the west.
    u = np.concatenate(([-pad], shore_u, [-pad]))
    v = np.concatenate(([v[0]], v, [v[-1]]))
    lon, lat = domain.denormalise(u, v)
    return np.column_stack((lon, lat))


@dataclass(frozen=True)
class Basin:
    """A paraboloid bowl of sediment inside an elliptical outline.

    Thickness peaks at the centre and closes to zero on the outline, so the
    basement surface meets the topography exactly at the basin edge.
    """

    centre_lon: float
    centre_lat: float
    #: Semi-axes of the elliptical outline, in degrees.
    radius_lon: float
    radius_lat: float
    #: Sediment thickness at the centre, in metres.
    thickness: float

    def radius(self, lon: npt.ArrayLike, lat: npt.ArrayLike) -> np.ndarray:
        """Elliptical radius, one on the outline and zero at the centre."""
        dx = (np.asarray(lon, dtype=np.float64) - self.centre_lon) / self.radius_lon
        dy = (np.asarray(lat, dtype=np.float64) - self.centre_lat) / self.radius_lat
        return np.hypot(dx, dy)

    def sediment(self, lon: npt.ArrayLike, lat: npt.ArrayLike) -> np.ndarray:
        """Sediment thickness in metres, zero outside the outline."""
        squared = self.radius(lon, lat) ** 2
        return self.thickness * np.clip(1.0 - squared, 0.0, None)

    def basement(
        self, lon: npt.ArrayLike, lat: npt.ArrayLike, domain: Domain = DOMAIN
    ) -> np.ndarray:
        """Basement elevation in metres, positive up from sea level."""
        return elevation(lon, lat, domain) - self.sediment(lon, lat)

    def outline(self, samples: int = 128) -> np.ndarray:
        """Vertices of the elliptical outline, in WGS84 degrees.

        Parameters
        ----------
        samples :
            Number of points around the ellipse.

        Returns
        -------
        numpy.ndarray
            ``(N, 2)`` array of longitude, latitude vertices, closed.
        """
        theta = np.linspace(0.0, 2.0 * np.pi, samples)
        lon = self.centre_lon + self.radius_lon * np.cos(theta)
        lat = self.centre_lat + self.radius_lat * np.sin(theta)
        return np.column_stack((lon, lat))


class BasinName(StrEnum):
    """The basins in the synthetic dataset."""

    GULLY = auto()
    TERRACE = auto()


#: A deep narrow basin up in the hills, plus a shallow broad one by the coast.
BASINS: dict[BasinName, Basin] = {
    BasinName.GULLY: Basin(
        centre_lon=172.15,
        centre_lat=-43.70,
        radius_lon=0.08,
        radius_lat=0.06,
        thickness=900.0,
    ),
    BasinName.TERRACE: Basin(
        centre_lon=172.30,
        centre_lat=-43.53,
        radius_lon=0.09,
        radius_lat=0.07,
        thickness=500.0,
    ),
}

#: A layered 1-D profile in ``fd_modfile`` column order, in km and km/s.
PROFILE = pd.DataFrame(
    {
        "vp": [1.80, 2.10, 2.50, 2.85, 3.20],
        "vs": [0.50, 0.68, 0.98, 1.28, 1.60],
        "rho": [1.81, 1.95, 2.09, 2.19, 2.30],
        "qp": [100.0, 120.0, 150.0, 180.0, 200.0],
        "qs": [50.0, 60.0, 75.0, 90.0, 100.0],
        "bottom_depth": [0.10, 0.30, 0.80, 2.00, 5.00],
    }
)


def _model_frame() -> Affine:
    """The EP2020 affine mapping NZTM metres onto tomography model kilometres."""
    return MODEL_COLUMNS[ModelType.EP2020].affine_inverse


def _apply(transform: Affine, points: np.ndarray) -> np.ndarray:
    """Apply a 4×4 homogeneous affine to an ``(N, 3)`` array of points."""
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ np.asarray(transform, dtype=np.float64).T)[:, :3]


def tomography(
    domain: Domain = DOMAIN,
    n_horizontal: int = 12,
    n_depth: int = 10,
    bottom: float = 30.0,
    margin: float = 20.0,
) -> pd.DataFrame:
    """A tomography block covering *domain*, in EP2020 model coordinates.

    :func:`nzcvm.scripts.convert_tomography.data_frame_to_mesh` needs a
    rectilinear grid in the model's own frame, so this lays the grid out there
    and derives the geographic columns from it, rather than the other way
    round.

    Parameters
    ----------
    domain :
        Rectangle the block has to cover.
    n_horizontal :
        Samples along each horizontal model axis.
    n_depth :
        Samples through the depth axis.
    bottom :
        Depth of the deepest sample, in km below sea level.
    margin :
        Horizontal padding around the domain, in km, so that queries near the
        domain edge stay inside the block.

    Returns
    -------
    pandas.DataFrame
        One row per grid point, with the columns
        :data:`~nzcvm.scripts.convert_tomography.MODEL_COLUMNS` expects.
    """
    to_nztm = Transformer.from_crs(CRS_WGS, CRS_NZTM, always_xy=True)
    to_wgs = Transformer.from_crs(CRS_NZTM, CRS_WGS, always_xy=True)

    inverse = _model_frame()
    forward = np.linalg.inv(inverse)

    corners = np.asarray(domain.polygon.exterior.coords)
    corner_x, corner_y = to_nztm.transform(corners[:, 0], corners[:, 1])
    corner_model = _apply(
        inverse, np.column_stack((corner_x, corner_y, np.zeros_like(corner_x)))
    )

    x = np.linspace(
        corner_model[:, 0].min() - margin,
        corner_model[:, 0].max() + margin,
        n_horizontal,
    )
    y = np.linspace(
        corner_model[:, 1].min() - margin,
        corner_model[:, 1].max() + margin,
        n_horizontal,
    )
    z = np.linspace(-TOMOGRAPHY_HEADROOM, bottom, n_depth)

    grid_x, grid_y, grid_z = (axis.ravel() for axis in np.meshgrid(x, y, z))

    nztm = _apply(forward, np.column_stack((grid_x, grid_y, grid_z)))
    lon, lat = to_wgs.transform(nztm[:, 0], nztm[:, 1])

    undulation = TOMOGRAPHY_UNDULATION * (
        np.sin(2 * np.pi * grid_x / TOMOGRAPHY_WAVELENGTH)
        * np.cos(2 * np.pi * grid_y / TOMOGRAPHY_WAVELENGTH)
    )
    vp = (TOMOGRAPHY_VP + TOMOGRAPHY_VP_GRADIENT * grid_z) * (1.0 + undulation)

    return pd.DataFrame(
        {
            "Latitude": lat,
            "Longitude": lon,
            "x(km)": grid_x,
            "y(km)": grid_y,
            "Depth(km_BSL)": grid_z,
            # Gardner's relation, in g/cm3, which the converter scales to SI.
            "Density": 1.74 * vp**0.25,
            "Vp": vp,
            "Vs": vp / VP_VS_RATIO,
        }
    )
