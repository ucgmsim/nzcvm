"""Grid of scattered points, for running a layer pipeline away from a volume.

:func:`points_grid` lays arbitrary points out as a :class:`~nzcvm.grids.grid.Grid`
so that a pipeline from :func:`~nzcvm.layers.pipeline.build_pipeline` can read
the model at them, such as on a fault plane.  Every point becomes its own
column of one sample: the ``i`` axis indexes the points, and ``j`` and ``k``
each have length one.

A one-sample column has no vertical neighbours, so a layer that reads them
can't do its job here.  The Backus layer spans zero depth at each point and
returns the model at the point itself, which isn't an average.  Leave it out of
the pipeline.
Layers that read only the top of each column, like the Ely taper and the
coastline, work as they do on any other grid.
"""

import numpy as np
import numpy.typing as npt
import shapely
import xarray as xr

from nzcvm.config.grids.model import Projection
from nzcvm.config.grids.terrain import Solver
from nzcvm.coordinates import Coordinate
from nzcvm.grids.grid import Grid, GridSchema
from nzcvm.models.surface import Surface

#: Default name of a points grid.
GRID_NAME = "points"


def points_grid(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    z: npt.ArrayLike,
    surface: Surface,
    projection: Projection,
    name: str = GRID_NAME,
) -> Grid:
    """Lay scattered points out as a grid of one-sample columns.

    Parameters
    ----------
    x, y :
        Projected horizontal coordinates of the points, in metres in
        *projection*'s CRS, which is also *surface*'s.
    z :
        Elevation of the points in metres, positive down.  *x*, *y* and *z*
        may take any shapes that broadcast together.  The grid flattens them
        in C order.
    surface :
        Topography, positive down, that gives each point its depth.  A point
        beyond its edge takes the nearest edge value.
    projection :
        The CRS of *x* and *y*, which places the WGS84 origin attributes.
    name :
        Name of the grid.

    Returns
    -------
    Grid
        A grid with dimensions ``(i, j, k)`` of shape ``(n, 1, 1)`` for *n*
        points, holding ``x``, ``y``, ``z`` and ``depth`` in float32.  A point
        above the topography moves down onto it, so it reads the ground
        rather than the air: ``z`` is the larger of the given elevation and
        the surface's, and ``depth`` is ``z`` less the surface's elevation,
        never negative.  The grid uses the ``physical`` solver.  Its
        ``nominal_depth`` and ``resolution`` are NaN because the points have
        no spacing, and its origin and bottom-left attributes are the mean
        and minimum of the points in WGS84.

    Raises
    ------
    ValueError
        If there are no points.

    Examples
    --------
    Reading a pipeline at the points and putting the result back in the
    shape of the input::

        grid = points_grid(x, y, z, Surface.load(dem), Projection(crs=nztm))
        qualities = pipeline(grid)
        vs = qualities.vs.values.reshape(np.broadcast_shapes(*map(np.shape, (x, y, z))))
    """
    x, y, z = (
        np.ravel(values).astype(np.float32) for values in np.broadcast_arrays(x, y, z)
    )
    if x.size == 0:
        raise ValueError("A points grid needs at least one point.")

    surface_z = np.asarray(surface.transform(x, y, extrapolate=True), np.float32)
    z = np.maximum(z, surface_z)
    depth = z - surface_z

    # A points grid has no origin of its own, so stand one up from the points
    # for the benefit of the Grid attributes every writer expects.
    (origin_lon, min_lon), (origin_lat, min_lat) = projection.to_wgs84.transform(
        [x.mean(), x.min()], [y.mean(), y.min()]
    )

    def column(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(
            values.reshape(-1, 1, 1), dims=(Coordinate.I, Coordinate.J, Coordinate.K)
        )

    grid = GridSchema.new(
        column(x),
        column(y),
        column(z),
        column(depth),
        nominal_depth=np.full(1, np.nan, dtype=np.float32),
        name=name,
        resolution=np.nan,
        solver=Solver.PHYSICAL.value,
        geometry=shapely.MultiPoint(np.column_stack((x, y))),
        origin_lon=np.float32(origin_lon),
        origin_lat=np.float32(origin_lat),
        azimuth=np.float32(0.0),
        grid_azimuth=np.float32(0.0),
        bottom_left_lon=np.float32(min_lon),
        bottom_left_lat=np.float32(min_lat),
    )
    # Layers that read the top of each column, like the Ely taper, select it
    # by `k`, and each point is the top of its own column.
    return grid.assign_coords(
        {
            Coordinate.I: np.arange(x.size),
            Coordinate.J: [0],
            Coordinate.K: [0],
        }
    )
