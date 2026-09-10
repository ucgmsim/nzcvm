"""Flat CSV velocity-model writer.

Writes one row per grid point: the logical index, the position, and every
component, under a header naming each column.  A grid may carry coordinates
beyond the ``(i, j, k)`` index, and each of those becomes a leading label
column, so a borehole grid's ``site`` labels reach the table and a reader can
tell one profile from another without counting rows.

The whole table goes through :mod:`pandas`, so a run holds every point in
memory at once.  That suits the outputs a person reads: boreholes, transects,
a handful of profiles. Volumetric grids belong in Zarr or NetCDF.
"""

from pathlib import Path

import pandas as pd

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.grids.grid import Grid
from nzcvm.qualities import Qualities
from nzcvm.velocity_model import VelocityModel

#: Every grid and quality array is float32, and nine significant digits
#: round-trip a float32 exactly. Left to pandas, a seven-digit easting comes
#: out as ``1.5315095e+06`` instead.
FLOAT_FORMAT = "%.9g"

#: Column naming the grid a row came from.  An SW4 domain writes one grid per
#: refinement level, so the name is what separates them in a single table.
GRID_COLUMN = "grid"

#: Columns every grid has, in the order they appear after the label columns.
FIXED_COLUMNS: tuple[str, ...] = (
    Coordinate.I,
    Coordinate.J,
    Coordinate.K,
    Coordinate.X,
    Coordinate.Y,
    Coordinate.Z,
    Coordinate.DEPTH,
    *Component,
)


def _table(name: str, grid: Grid, qualities: Qualities) -> pd.DataFrame:
    """Flatten one grid and its qualities into a row-per-point table.

    Parameters
    ----------
    name :
        Name of the grid, written into the :data:`GRID_COLUMN` column.
    grid :
        Grid holding the ``x``, ``y``, ``z`` and ``depth`` positions.
    qualities :
        Components sampled on *grid*.

    Returns
    -------
    pandas.DataFrame
        One row per grid point, columns ordered
        :data:`GRID_COLUMN`, labels, then :data:`FIXED_COLUMNS`.
    """
    table = grid.assign(qualities).to_dataframe().reset_index()
    # Whatever the grid carries past the logical index labels the rows: `site`
    # on a borehole grid, and anything a custom grid builder adds.
    labels = [column for column in table.columns if column not in FIXED_COLUMNS]
    table.insert(0, GRID_COLUMN, name)
    return table[[GRID_COLUMN, *labels, *FIXED_COLUMNS]]


def to_csv(velocity_model: VelocityModel, path: Path) -> None:
    """Write *velocity_model* to *path* as a single CSV table.

    Parameters
    ----------
    velocity_model :
        Model whose grids the query pipeline has already populated.
    path :
        Destination file.
    """
    tables = [
        _table(name, grid, qualities)
        for name, (grid, qualities) in velocity_model.pairwise.items()
    ]
    pd.concat(tables, ignore_index=True).to_csv(
        path, index=False, float_format=FLOAT_FORMAT
    )
