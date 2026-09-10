"""Topography-following borehole grid builder.

Provides :func:`build_borehole` for constructing the set of vertical columns
described by a :class:`~nzcvm.config.grids.borehole.BoreholeGridConfig`.
Each site becomes one column of query points, sampled at a strictly fixed Z
resolution from the topography down to a fixed depth, so a run extracts a
directly comparable profile per site rather than a filled volume.

Columns are independent, so none of the extent, azimuth or origin metadata
the volumetric grids record means anything here.  The builder still fills in
the :class:`~nzcvm.grids.grid.Grid` attributes that name an origin, taking the
centroid of the sites as the origin and the south-west corner of their
bounding box as the bottom-left corner.

Longitude and latitude place a site.  The config doesn't reserve any other
key, so every other key or column becomes a coordinate on the ``i`` axis under
the name the caller gave it, which is how a station code or a network ends up
in the layer chain and the output.
:data:`~nzcvm.grids.grid.RESERVED_COORDINATES` lists the names a label may not
take, and ``keep_extra_columns = false`` drops the labels altogether.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pandas as pd
import shapely
import xarray as xr

from nzcvm.config.grids.borehole import SPATIAL_KEYS, BoreholeGridConfig, Site
from nzcvm.coordinates import Coordinate
from nzcvm.grids import helpers
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.grids.grid import RESERVED_COORDINATES, Grid, GridSchema
from nzcvm.models.surface import Surface

#: Name of the one grid a borehole config builds.
GRID_NAME = "boreholes"

#: Readers for the supported site file formats, keyed by suffix.
SITE_READERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".pq": pd.read_parquet,
}


def read_sites(path: Path) -> list[Site]:
    """Read borehole sites from a CSV or Parquet file.

    Parameters
    ----------
    path :
        File with ``longitude`` and ``latitude`` columns.  Every other column
        becomes a label on the site it belongs to.

    Returns
    -------
    list[Site]
        One :class:`~nzcvm.config.grids.borehole.Site` per row, in file order.

    Raises
    ------
    ValueError
        If the suffix isn't a supported format, or a spatial column is
        missing.
    """
    reader = SITE_READERS.get(path.suffix.lower())
    if reader is None:
        supported = ", ".join(sorted(SITE_READERS))
        raise ValueError(
            f"Cannot read sites from '{path}': expected one of {supported}"
        )

    frame = reader(path)
    missing = [column for column in SPATIAL_KEYS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Site file '{path}' is missing the {', '.join(missing)} column(s). "
            f"Expected {', '.join(SPATIAL_KEYS)}."
        )

    return [
        Site(
            longitude=float(record.pop("longitude")),
            latitude=float(record.pop("latitude")),
            labels=record,
        )
        for record in frame.to_dict("records")
    ]


def site_labels(sites: list[Site]) -> dict[str, np.ndarray]:
    """Collect the site labels into one array per label name.

    Parameters
    ----------
    sites :
        Sites to read the labels off.

    Returns
    -------
    dict[str, numpy.ndarray]
        One array per label, ordered as *sites* are and typed by whatever
        NumPy infers from the values.

    Raises
    ------
    ValueError
        If a label collides with
        :data:`~nzcvm.grids.grid.RESERVED_COORDINATES`, or if the sites do
        not agree on which labels they carry.  Disagreement is nearly always
        a typo, and the alternative is a column of nulls.
    """
    names = list(sites[0].labels)

    reserved = sorted(RESERVED_COORDINATES.intersection(names))
    if reserved:
        raise ValueError(
            f"Site label(s) {', '.join(reserved)} would shadow a grid variable "
            f"or attribute of the same name. Rename them, or set "
            f"keep_extra_columns = false."
        )

    for position, site in enumerate(sites):
        if set(site.labels) != set(names):
            raise ValueError(
                f"Site {position} carries labels "
                f"{sorted(site.labels) or 'none'}, but site 0 carries "
                f"{sorted(names)}. Every site needs the same labels."
            )

    return {name: np.asarray([site.labels[name] for site in sites]) for name in names}


def resolve_sites(config: BoreholeGridConfig) -> list[Site]:
    """Return the sites a config names, reading them from disk if needed."""
    sites = config.sites if isinstance(config.sites, list) else read_sites(config.sites)
    if not sites:
        raise ValueError(f"No sites found in '{config.sites}'.")
    return sites


def _columns(values: np.ndarray, index: np.ndarray, chunk: int) -> xr.DataArray:
    """Lay a per-site value out over the ``(i, j)`` plane as one column each."""
    chunked = da.from_array(values.astype(np.float32), chunks=chunk)
    return xr.DataArray(
        chunked[:, np.newaxis],
        dims=[Coordinate.I, Coordinate.J],
        coords={Coordinate.I: index, Coordinate.J: [0]},
    )


def _borehole_grid(
    x_phys: xr.DataArray,
    y_phys: xr.DataArray,
    surface: xr.DataArray,
    depth: float,
    resolution_z: float,
    **kwargs: Any,
) -> Grid:
    nk = np.round(depth / resolution_z).astype(int) + 1

    # Depth is purely a function of k and resolution_z, identically for every
    # column, which is what makes the profiles comparable between sites.
    # Chunking only ever applies to i/j. k always stays one chunk.
    zeta_depth = xr.DataArray(
        np.linspace(0.0, depth, num=nk, dtype=np.float32),
        dims=[Coordinate.K],
        coords={Coordinate.K: np.arange(nk)},
    ).chunk({Coordinate.K: -1})

    # Elevation (z) is the surface elevation shifted downward by the fixed
    # depths, so every column starts at the ground rather than at sea level.
    x, y, z, column_depth = xr.broadcast(
        x_phys, y_phys, surface + zeta_depth, zeta_depth
    )
    x, y, z, column_depth = helpers.ensure_chunks(x, y, z, column_depth)

    return GridSchema.new(x, y, z, column_depth, **kwargs)


@build_grids_from_config.register
def build_borehole(config: BoreholeGridConfig) -> dict[str, Grid]:
    sites = resolve_sites(config)
    index = np.arange(len(sites))
    labels = site_labels(sites) if config.keep_extra_columns else {}

    transformer = config.projection.transformer_from(config.sites_crs)
    x, y = transformer.transform(
        np.array([site.longitude for site in sites]),
        np.array([site.latitude for site in sites]),
    )

    chunk = config.chunks[Coordinate.I]
    x_phys = _columns(x, index, chunk)
    y_phys = _columns(y, index, chunk)

    # A borehole grid has no origin of its own, so stand one up from the sites
    # for the benefit of the Grid attributes every writer expects.
    to_wgs84 = config.projection.to_wgs84
    origin_lon, origin_lat = to_wgs84.transform(x.mean(), y.mean())
    min_lon, min_lat = to_wgs84.transform(x.min(), y.min())

    topographic_surface = Surface.load(config.surface)
    z_surface = helpers.compute_surface_elevation(
        topographic_surface,
        x_phys,
        y_phys,
    )

    grid = _borehole_grid(
        x_phys,
        y_phys,
        z_surface,
        name=GRID_NAME,
        depth=config.depth,
        resolution_z=config.resolution_z,
        resolution=config.resolution_z,
        geometry=shapely.MultiPoint(np.column_stack((x, y))),
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        azimuth=np.float32(0.0),
        grid_azimuth=np.float32(0.0),
        bottom_left_lon=min_lon,
        bottom_left_lat=min_lat,
    )
    # Labelling i is what makes the output readable: without it, the only
    # route back to a station is the order the config listed it in.
    grid = grid.assign_coords(
        {name: (Coordinate.I, values) for name, values in labels.items()}
    )

    return {GRID_NAME: grid}
