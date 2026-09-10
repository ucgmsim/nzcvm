"""Flat table velocity-model writers, for CSV and Parquet.

Both write one row per grid point. Each row gives the logical index, the
position, and then the components.  A grid may hold coordinates beyond the
``(i, j, k)`` index, and each of those becomes a leading label column.  That
puts a borehole grid's ``site`` labels in the table, so a reader can tell one
profile from another without counting rows.

:func:`flatten` builds the table and the two writers encode it, so the column
set is the same either way.  Parquet keeps the float32 arrays typed and exact
and compresses a large table well, while CSV renders every number as text,
which is what :data:`FLOAT_FORMAT` is for. Per-column metadata makes a small
Parquet file the larger of the two, so the choice is about the reader rather
than about size.

The whole table goes through :mod:`pandas`, which keeps every point in memory
at once.  That suits the outputs a person reads: boreholes, transects, a few
profiles. Volumetric grids belong in Zarr or NetCDF.
"""

from pathlib import Path

import pandas as pd

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.grids.grid import Grid
from nzcvm.qualities import Qualities
from nzcvm.velocity_model import VelocityModel

#: CSV precision. Every grid and quality array is float32, and nine digits
#: round-trip a float32 exactly. Left to pandas, a seven-digit easting comes
#: out as ``1.5315095e+06`` instead. Parquet keeps the float32 typed and
#: needs none of this.
FLOAT_FORMAT = "%.9g"

#: Column for the name of the grid a row came from.  An SW4 domain writes one
#: grid per refinement level, so the name is what separates them in one table.
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
    # Any coordinate past the logical index labels the rows: `site` on a
    # borehole grid, and anything a custom grid builder adds.
    labels = [column for column in table.columns if column not in FIXED_COLUMNS]
    table.insert(0, GRID_COLUMN, name)
    return table[[GRID_COLUMN, *labels, *FIXED_COLUMNS]]


def flatten(velocity_model: VelocityModel) -> pd.DataFrame:
    """Flatten every grid in *velocity_model* into one row-per-point table.

    Parameters
    ----------
    velocity_model :
        Model whose grids the query pipeline has already populated.

    Returns
    -------
    pandas.DataFrame
        The grids concatenated in order, one row per point.
    """
    tables = [
        _table(name, grid, qualities)
        for name, (grid, qualities) in velocity_model.pairwise.items()
    ]
    return pd.concat(tables, ignore_index=True)


def to_csv(velocity_model: VelocityModel, path: Path) -> None:
    """Write *velocity_model* to *path* as one CSV table.

    Parameters
    ----------
    velocity_model :
        Model whose grids the query pipeline has already populated.
    path :
        Destination file.
    """
    flatten(velocity_model).to_csv(path, index=False, float_format=FLOAT_FORMAT)


def to_parquet(velocity_model: VelocityModel, path: Path) -> None:
    """Write *velocity_model* to *path* as one Parquet table.

    The same columns as :func:`to_csv`, with the float32 arrays kept typed
    rather than rendered as text.

    Parameters
    ----------
    velocity_model :
        Model whose grids the query pipeline has already populated.
    path :
        Destination file.
    """
    flatten(velocity_model).to_parquet(path, index=False)
