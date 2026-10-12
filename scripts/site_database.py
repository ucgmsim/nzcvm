"""Build the site database: a borehole profile and site parameters per station.

Writes four files to the output directory:

``profiles.parquet``
    One row per sample of every station's profile, in the layout of the
    ``parquet`` output format. This is the reference copy.
``profiles.csv``
    The same table as CSV, converted from the Parquet file.
``summary.parquet`` / ``summary.csv``
    One row per station: Vs30, Z1.0 and Z2.5, and the depth the station's
    profile reaches.

Every profile goes down to ``--depth`` at ``--resolution`` spacing. A station
whose profile never reaches Vs = 2.5 km/s has its profile re-run twice as
deep, again and again, until it does, so its rows in ``profiles.parquet``
reach further than the rest. The build fails if any station still misses
2.5 km/s at ``--max-depth``.

The script builds the grid and query pipeline once for every station, which
is the slow part since it loads the nationwide DEM and Vs30 surfaces. It then
computes the profiles in batches of about ``--rows-per-batch`` samples, since
the table writer loads a whole batch into memory and the full database
doesn't fit in a CI runner's RAM.
"""

import dataclasses
import logging
import time
from pathlib import Path
from typing import Annotated

import dask
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import typer

from nzcvm.components import Component
from nzcvm.config import VelocityModelConfig
from nzcvm.config.grids.borehole import DEFAULT_CHUNK_SIZES, Site
from nzcvm.coordinates import Coordinate
from nzcvm.formats.table import flatten
from nzcvm.layers import pipeline
from nzcvm.layers.core import Layer
from nzcvm.velocity_model import VelocityModel

logger = logging.getLogger("site_database")

#: Column naming each station in both the input CSV and the outputs.
STATION = "station"

#: Vs that Z1.0 is the depth to, in m/s.
VS_1P0 = 1000.0
#: Vs that Z2.5 is the depth to, in m/s.
VS_2P5 = 2500.0

#: Stations per chunk of the borehole grid's ``i`` axis.
CHUNK = DEFAULT_CHUNK_SIZES[Coordinate.I]

#: Depth, in metres, that Vs30 averages over.
VS30_DEPTH = 30.0

app = typer.Typer()


def read_stations(path: Path) -> pd.DataFrame:
    """Read the station list, renaming ``lon``/``lat`` to the borehole keys.

    Parameters
    ----------
    path :
        CSV with ``lon``, ``lat`` and ``station`` columns.

    Returns
    -------
    pandas.DataFrame
        Columns ``station``, ``longitude`` and ``latitude``.

    Raises
    ------
    ValueError
        If a station name appears more than once. Names key the deepening
        passes, so they have to be unique.
    """
    stations = pd.read_csv(path).rename(
        columns={"lon": "longitude", "lat": "latitude"}
    )[[STATION, "longitude", "latitude"]]
    duplicated = stations[STATION][stations[STATION].duplicated()]
    if not duplicated.empty:
        raise ValueError(f"Duplicate station names: {sorted(duplicated)[:10]}")
    return stations


def sites_model(
    config: VelocityModelConfig,
    stations: pd.DataFrame,
    depth_m: float,
    model_pipeline: Layer | None = None,
) -> tuple[VelocityModel, Layer]:
    """Lay out a profile at each of *stations* and attach the query pipeline.

    This leaves the qualities lazy, for the caller to compute a slice of
    stations at a time.  Building the grid and the pipeline
    is the slow part, since each loads a nationwide surface (the DEM and the
    Vs30 map), so this happens once per depth rather than once per slice.

    Parameters
    ----------
    config :
        Site database config. This replaces its ``sites`` and ``depth``.
    stations :
        Stations to lay profiles out at.
    depth_m :
        Bottom of every profile, in metres below the topography.
    model_pipeline :
        Pipeline to reuse from an earlier call, built over a superset of
        *stations*.  Built afresh when not given.

    Returns
    -------
    velocity_model : VelocityModel
        The profiles, with lazy qualities.
    model_pipeline : Layer
        The query pipeline, for reuse.
    """
    sites = [
        Site(
            longitude=row.longitude,
            latitude=row.latitude,
            labels={STATION: row.station},
        )
        for row in stations.itertuples(index=False)
    ]
    grid = dataclasses.replace(config.grid, sites=sites, depth=depth_m)
    velocity_model = VelocityModel.from_config(dataclasses.replace(config, grid=grid))
    if model_pipeline is None:
        model_pipeline = pipeline.build_pipeline(velocity_model.geometry, config.layers)
    velocity_model = pipeline.execute_model_pipeline(velocity_model, model_pipeline)
    return velocity_model, model_pipeline


def query_profiles(velocity_model: VelocityModel, sites: slice) -> pd.DataFrame:
    """Compute the profiles at a slice of *velocity_model*'s stations.

    Parameters
    ----------
    velocity_model :
        Output of :func:`sites_model`.
    sites :
        Stations to compute, as a slice along the ``i`` axis.

    Returns
    -------
    pandas.DataFrame
        The profiles in the layout of the ``parquet`` output format, ordered
        by station then depth.
    """
    batch = velocity_model.map(lambda dataset: dataset.isel({Coordinate.I: sites}))
    return flatten(batch).sort_values([Coordinate.I, Coordinate.K], ignore_index=True)


def summarise(profiles: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Compute Vs30, Z1.0 and Z2.5 from one query's profiles.

    Every profile in *profiles* has the same depth samples, which lets this
    work on a ``(station, depth)`` array instead of grouping.

    Parameters
    ----------
    profiles :
        Output of :func:`query_profiles` for *stations*.
    stations :
        The stations *profiles* holds, in the same order.

    Returns
    -------
    pandas.DataFrame
        One row per station. ``z1p0_m`` and ``z2p5_m`` are the depth of the
        first sample at or above 1.0 and 2.5 km/s, and NaN where the profile
        never gets there. ``vs30_m_per_s`` is the travel-time average over
        the samples above 30 m, each standing for the interval down to the
        next sample.
    """
    n_stations = len(stations)
    vs = profiles[Component.VS].to_numpy(np.float64).reshape(n_stations, -1)
    depth_m = profiles[Coordinate.DEPTH].to_numpy(np.float64).reshape(n_stations, -1)

    thickness_m = np.diff(depth_m, axis=1)
    in_top = depth_m[:, :-1] < VS30_DEPTH
    vs30 = VS30_DEPTH / np.sum(np.where(in_top, thickness_m / vs[:, :-1], 0.0), axis=1)

    def first_depth(threshold: float) -> np.ndarray:
        reached = vs >= threshold
        first = np.argmax(reached, axis=1)
        return np.where(
            reached.any(axis=1), depth_m[np.arange(n_stations), first], np.nan
        )

    return pd.DataFrame(
        {
            STATION: stations[STATION].to_numpy(),
            "longitude": stations["longitude"].to_numpy(),
            "latitude": stations["latitude"].to_numpy(),
            "vs30_m_per_s": vs30,
            "z1p0_m": first_depth(VS_1P0),
            "z2p5_m": first_depth(VS_2P5),
            "profile_depth_m": depth_m[:, -1],
        }
    )


class ProfileWriter:
    """Append profile tables to one Parquet file under a fixed schema."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer: pq.ParquetWriter | None = None

    def write(self, profiles: pd.DataFrame) -> None:
        """Append *profiles*, taking the schema from the first call."""
        if self.writer is None:
            table = pa.Table.from_pandas(profiles, preserve_index=False)
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        else:
            table = pa.Table.from_pandas(
                profiles, schema=self.writer.schema, preserve_index=False
            )
        self.writer.write_table(table)

    def close(self) -> None:
        """Finish the file."""
        if self.writer is not None:
            self.writer.close()


def write_pass(
    velocity_model: VelocityModel,
    stations: pd.DataFrame,
    batch_size: int,
    writer: ProfileWriter,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write every station's profile that reaches Z2.5, a batch at a time.

    Parameters
    ----------
    velocity_model :
        Output of :func:`sites_model` for *stations*.
    stations :
        The stations, in grid order.
    batch_size :
        Stations to compute at once.
    writer :
        Destination for the finished profiles.

    Returns
    -------
    summary : pandas.DataFrame
        Site parameters of the stations that reach Z2.5.
    pending : pandas.DataFrame
        The stations that don't, to run again deeper.
    """
    summaries = []
    pending = []
    n_batches = -(-len(stations) // batch_size)
    start = time.monotonic()
    for number, offset in enumerate(range(0, len(stations), batch_size), 1):
        sites = slice(offset, offset + batch_size)
        batch = stations.iloc[sites]
        profiles = query_profiles(velocity_model, sites)
        missing = profiles.loc[profiles[Component.VS].isna(), STATION].unique()
        if missing.size:
            # No depth gets these to Z2.5, so stop before deepening them.
            raise RuntimeError(
                f"{missing.size} station(s) have no Vs at some depth: "
                f"{missing[:10].tolist()}"
            )
        summary = summarise(profiles, batch)
        done = summary["z2p5_m"].notna().to_numpy()
        writer.write(profiles[profiles[STATION].isin(set(batch[STATION][done]))])
        summaries.append(summary[done])
        pending.append(batch[~done])
        elapsed = time.monotonic() - start
        logger.info(
            "batch %d/%d done, %.0f s elapsed, ~%.0f s left",
            number,
            n_batches,
            elapsed,
            elapsed / number * (n_batches - number),
        )
    return pd.concat(summaries, ignore_index=True), pd.concat(
        pending, ignore_index=True
    )


def parquet_to_csv(parquet_path: Path, csv_path: Path) -> None:
    """Convert *parquet_path* to CSV one row group at a time."""
    parquet = pq.ParquetFile(parquet_path)
    with pa_csv.CSVWriter(csv_path, parquet.schema_arrow) as writer:
        for index in range(parquet.num_row_groups):
            writer.write_table(parquet.read_row_group(index))


@app.command()
def main(
    stations_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_dir: Path,
    depth_m: Annotated[float, typer.Option("--depth", min=1.0)] = 5000.0,
    resolution_m: Annotated[float, typer.Option("--resolution", min=0.1)] = 5.0,
    max_depth_m: Annotated[float, typer.Option("--max-depth", min=1.0)] = 80000.0,
    n_threads: Annotated[int, typer.Option(min=1)] = 4,
    rows_per_batch: Annotated[int, typer.Option(min=1)] = 4_000_000,
    limit: Annotated[
        int | None, typer.Option(help="Only use the first N stations (for testing).")
    ] = None,
) -> None:
    """Build profiles.parquet, profiles.csv, summary.parquet and summary.csv."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    stations = read_stations(stations_path)
    if limit is not None:
        stations = stations.head(limit)
    config = VelocityModelConfig.read_config(config_path)
    config = dataclasses.replace(
        config, grid=dataclasses.replace(config.grid, resolution_z=resolution_m)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / "profiles.parquet"
    writer = ProfileWriter(profiles_path)
    summaries = []
    model_pipeline = None
    pending = stations
    try:
        with dask.config.set(scheduler="threads", num_workers=n_threads):
            while not pending.empty:
                if depth_m > max_depth_m:
                    raise RuntimeError(
                        f"{len(pending)} station(s) never reach Vs = {VS_2P5} m/s "
                        f"above {max_depth_m} m: {pending[STATION].tolist()[:10]}"
                    )
                logger.info("%d station(s) to %.0f m", len(pending), depth_m)
                velocity_model, model_pipeline = sites_model(
                    config, pending, depth_m, model_pipeline
                )
                samples_per_profile = int(depth_m // resolution_m) + 1
                # Whole chunks of the borehole grid's station axis per batch.
                batch_size = max(
                    CHUNK, rows_per_batch // samples_per_profile // CHUNK * CHUNK
                )
                summary, pending = write_pass(
                    velocity_model, pending, batch_size, writer
                )
                summaries.append(summary)
                depth_m *= 2
    finally:
        writer.close()

    summary = (
        pd.concat(summaries).set_index(STATION).loc[stations[STATION]].reset_index()
    )
    summary.to_parquet(output_dir / "summary.parquet", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    parquet_to_csv(profiles_path, output_dir / "profiles.csv")
    logger.info(
        "%d stations, %d deepened past the first pass",
        len(summary),
        int((summary["profile_depth_m"] > summary["profile_depth_m"].min()).sum()),
    )


if __name__ == "__main__":
    app()
