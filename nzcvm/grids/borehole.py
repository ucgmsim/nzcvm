"""Topography-following borehole grid builder.

Provides :func:`build_borehole` for constructing the set of vertical columns
described by a :class:`~nzcvm.config.grids.borehole.BoreholeGridConfig`: one
column of query points per site, sampled at a fixed Z resolution from the
topography down to a fixed depth, so a run extracts a directly comparable
profile per site rather than a filled volume.

See :class:`~nzcvm.config.grids.borehole.BoreholeGridConfig` for what a site
is and how the builder copies its labels onto the grid.
"""

from collections.abc import Callable
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import shapely
import xarray as xr

from nzcvm.config.grids.borehole import SPATIAL_KEYS, BoreholeGridConfig, Site
from nzcvm.coordinates import Coordinate
from nzcvm.grids import helpers
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.grids.grid import RESERVED_COORDINATES, Grid
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
        If a label shadows a grid name, or the sites disagree on which labels they
        carry.  Disagreement is nearly always a typo, and the alternative is
        a column of nulls.
    """
    names = list(sites[0].labels)
    expected = set(names)

    reserved = sorted(RESERVED_COORDINATES.intersection(names))
    if reserved:
        raise ValueError(
            f"Site label(s) {', '.join(reserved)} would shadow a grid variable "
            f"or attribute of the same name. Rename them, or set "
            f"keep_extra_columns = false."
        )

    for position, site in enumerate(sites):
        if set(site.labels) != expected:
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
    (origin_lon, min_lon), (origin_lat, min_lat) = config.projection.to_wgs84.transform(
        [x.mean(), x.min()], [y.mean(), y.min()]
    )

    topographic_surface = Surface.load(config.surface)
    z_surface = helpers.compute_surface_elevation(
        topographic_surface,
        x_phys,
        y_phys,
    )

    grid = helpers.topography_following_grid(
        x_phys,
        y_phys,
        z_surface,
        name=GRID_NAME,
        thickness=config.depth,
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
    # Site labels index i, so the output reads back per station.
    grid = grid.assign_coords(
        {name: (Coordinate.I, values) for name, values in labels.items()}
    )

    return {GRID_NAME: grid}
