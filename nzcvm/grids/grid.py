import dataclasses
from dataclasses import dataclass
from typing import Literal

import numpy as np
import shapely
import xarray as xr
from xarray_dataclasses import AsDataset, Attr, Coord, Data, DataOptions

from nzcvm.components import Component
from nzcvm.config.grids.terrain import Solver
from nzcvm.coordinates import Coordinate


class Grid(xr.Dataset):
    __slots__ = ()


i = Literal["i"]
j = Literal["j"]
k = Literal["k"]


@dataclass
class GridSchema(AsDataset):
    __dataoptions__ = DataOptions(Grid)

    x: Data[tuple[i, j, k], np.float32]
    y: Data[tuple[i, j, k], np.float32]
    z: Data[tuple[i, j, k], np.float32]
    depth: Data[tuple[i, j, k], np.float32]
    #: Depth of each level in a flat box, before the decay bends it to follow
    #: the topography.  See :mod:`nzcvm.grids.terrain`.
    nominal_depth: Coord[k, np.float32]
    name: Attr[str]
    resolution: Attr[float]
    #: A :class:`~nzcvm.config.grids.terrain.Solver`: whether a solver places
    #: samples at ``z`` or at ``nominal_depth``.
    solver: Attr[str]

    geometry: Attr[shapely.Geometry]
    origin_lon: Attr[np.float32]
    origin_lat: Attr[np.float32]

    azimuth: Attr[np.float32]
    grid_azimuth: Attr[np.float32]

    bottom_left_lon: Attr[np.float32]
    bottom_left_lat: Attr[np.float32]

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> Grid:
        """Parses, validates, and builds a Grid from a standard xr.Dataset.

        Files written before grids carried solver coordinates load as
        physical grids with an unknown (NaN) nominal depth.
        """
        attrs = {"solver": Solver.PHYSICAL.value, **dataset.attrs}
        if Coordinate.NOMINAL_DEPTH in dataset.coords:
            nominal_depth = dataset[Coordinate.NOMINAL_DEPTH]
        else:
            nominal_depth = np.full(dataset.sizes[Coordinate.K], np.nan, np.float32)
        dset = cls.new(
            **dataset.data_vars,  # ty: ignore[invalid-argument-type]
            nominal_depth=nominal_depth,
            **attrs,
        )
        return dset


#: Names a grid builder may not give an extra coordinate.
#:
#: A coordinate shadows a variable or an attribute of the same name, so a
#: coordinate called ``name`` would turn ``grid.name`` from the grid's name
#: into a :class:`~xarray.DataArray`. The set covers everything
#: :class:`GridSchema` declares, the logical index, the coordinate the
#: coastline layer adds, and the components that share that index with the
#: grid.
RESERVED_COORDINATES: frozenset[str] = frozenset(
    [field.name for field in dataclasses.fields(GridSchema)]
    + [Coordinate.I, Coordinate.J, Coordinate.K, Coordinate.COASTLINE]
    + list(Component)
)


def grid_like_at_depth(grid: Grid, depth: float) -> Grid:
    # Select a z-layer of the block.
    # The array [0] as the selection is important because it preserves the k
    # axis for downstream layers.
    layer = grid.isel({Coordinate.K: [0]})

    # This hack sets the reference elevation to an equivalent to depth below topography
    layer[Coordinate.Z] -= layer.depth - depth
    layer[Coordinate.DEPTH] = xr.full_like(layer[Coordinate.DEPTH], depth)
    return layer
