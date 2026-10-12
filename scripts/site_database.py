"""Build the site database: a borehole profile and site parameters per station.

Writes four files to the output directory:

``profiles.parquet``
    One row per sample of every station's profile, in the layout of the
    ``parquet`` output format. This is the reference copy.
``profiles.csv``
    The same table as CSV.
``summary.parquet`` / ``summary.csv``
    One row per station: Vs30, Z1.0 and Z2.5, and the depth the station's
    profile reaches.

The config gives the borehole grid: the station list (``sites``), the depth
every profile starts at and the sample spacing. The layers come from a
second config, the whole-country model, without its clamp layer, so the
profiles carry the Ely taper unclamped.

The stations are the observational ones in the station list: those with a
three- or four-character name, as opposed to the seven-character names of
the virtual stations. ``--all-stations`` takes every station instead.

A station whose profile never reaches Vs = 2.5 km/s has its profile re-run
twice as deep, again and again, until it does. Its rows in
``profiles.parquet`` reach further than the rest. The build fails if any
station still misses 2.5 km/s at ``--max-depth``.

The script builds the query pipeline once for every station, which is the
slow part since it loads the nationwide DEM and Vs30 surfaces. It then
computes the profiles in batches of about ``--rows-per-batch`` samples, since
the table writer loads a whole batch into memory and the full database
doesn't fit in a CI runner's RAM.
"""

import dataclasses
import logging
import time
from pathlib import Path
from typing import Annotated, Self

import dask
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from nzcvm.components import Component
from nzcvm.config import VelocityModelConfig
from nzcvm.config.grids.borehole import DEFAULT_CHUNK_SIZES, BoreholeGridConfig
from nzcvm.config.layers.clamp import ClampLayerConfig
from nzcvm.coordinates import Coordinate
from nzcvm.formats.table import FLOAT_FORMAT, flatten
from nzcvm.grids.borehole import read_site_table, sites_from_frame
from nzcvm.layers import pipeline
from nzcvm.velocity_model import VelocityModel

logger = logging.getLogger("site_database")

#: Column naming each station in both the input CSV and the outputs.
STATION = "station"

#: Name lengths of the observational stations. Virtual stations have
#: seven-character names.
OBSERVATIONAL_NAME_LENGTHS = (3, 4)

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
        Table with ``lon``, ``lat`` and ``station`` columns.

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
    stations = read_site_table(path).rename(
        columns={"lon": "longitude", "lat": "latitude"}
    )[[STATION, "longitude", "latitude"]]
    duplicated = stations[STATION][stations[STATION].duplicated()]
    if not duplicated.empty:
        raise ValueError(f"Duplicate station names: {sorted(duplicated)[:10]}")
    return stations


def sites_model(
    config: VelocityModelConfig, stations: pd.DataFrame, depth_m: float
) -> VelocityModel:
    """Lay out a profile at each of *stations*, with lazy qualities.

    Parameters
    ----------
    config :
        Site database config. This replaces its ``sites`` and ``depth``.
    stations :
        Stations to lay profiles out at.
    depth_m :
        Bottom of every profile, in metres below the topography.

    Returns
    -------
    VelocityModel
        The profiles, before the query pipeline runs.
    """
    grid = dataclasses.replace(
        config.grid, sites=sites_from_frame(stations), depth=depth_m
    )
    return VelocityModel.from_config(dataclasses.replace(config, grid=grid))


def query_profiles(velocity_model: VelocityModel, sites: slice) -> pd.DataFrame:
    """Compute the profiles at a slice of *velocity_model*'s stations.

    Parameters
    ----------
    velocity_model :
        Output of :func:`sites_model`, after the query pipeline.
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
    """Append profile tables to a Parquet file and a CSV file.

    The Parquet schema comes from the first table, and the CSV numbers use
    the ``csv`` output format's float format.
    """

    def __init__(self, parquet_path: Path, csv_path: Path) -> None:
        self.parquet_path = parquet_path
        self.csv_path = csv_path
        self.parquet: pq.ParquetWriter | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        if self.parquet is not None:
            self.parquet.close()

    def write(self, profiles: pd.DataFrame) -> None:
        """Append *profiles* to both files."""
        first = self.parquet is None
        if self.parquet is None:
            table = pa.Table.from_pandas(profiles, preserve_index=False)
            self.parquet = pq.ParquetWriter(
                self.parquet_path, table.schema, compression="zstd"
            )
        else:
            table = pa.Table.from_pandas(
                profiles, schema=self.parquet.schema, preserve_index=False
            )
        self.parquet.write_table(table)
        profiles.to_csv(
            self.csv_path,
            mode="w" if first else "a",
            header=first,
            index=False,
            float_format=FLOAT_FORMAT,
        )


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
        Output of :func:`sites_model` for *stations*, after the query
        pipeline.
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


@app.command()
def main(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    layers_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Config to take the layers from, minus any clamp layer.",
        ),
    ],
    output_dir: Path,
    max_depth_m: Annotated[float, typer.Option("--max-depth", min=1.0)] = 80000.0,
    n_threads: Annotated[int, typer.Option(min=1)] = 4,
    rows_per_batch: Annotated[int, typer.Option(min=1)] = 4_000_000,
    all_stations: Annotated[
        bool,
        typer.Option(help="Include the virtual stations, not just observational ones."),
    ] = False,
    limit: Annotated[
        int | None, typer.Option(help="Only use the first N stations (for testing).")
    ] = None,
) -> None:
    """Build profiles.parquet, profiles.csv, summary.parquet and summary.csv."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = VelocityModelConfig.read_config(config_path)
    layers = [
        layer
        for layer in VelocityModelConfig.read_config(layers_path).layers
        if not isinstance(layer, ClampLayerConfig)
    ]
    config = dataclasses.replace(config, layers=layers)
    grid = config.grid
    if not isinstance(grid, BoreholeGridConfig) or not isinstance(grid.sites, Path):
        raise typer.BadParameter(
            "The config needs a borehole grid over a station file."
        )
    stations = read_stations(grid.sites)
    if not all_stations:
        observational = stations[STATION].str.len().isin(OBSERVATIONAL_NAME_LENGTHS)
        stations = stations[observational].reset_index(drop=True)
    if limit is not None:
        stations = stations.head(limit)
    depth_m = grid.depth

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    pending = stations
    with (
        dask.config.set(scheduler="threads", num_workers=n_threads),
        ProfileWriter(
            output_dir / "profiles.parquet", output_dir / "profiles.csv"
        ) as writer,
    ):
        # Building the pipeline loads the nationwide DEM and Vs30 surfaces,
        # so it happens once, over every station, and serves the deeper
        # passes too.
        model_pipeline = pipeline.build_pipeline(
            sites_model(config, stations, depth_m).geometry, config.layers
        )
        while not pending.empty:
            if depth_m > max_depth_m:
                raise RuntimeError(
                    f"{len(pending)} station(s) never reach Vs = {VS_2P5} m/s "
                    f"above {max_depth_m} m: {pending[STATION].tolist()[:10]}"
                )
            logger.info("%d station(s) to %.0f m", len(pending), depth_m)
            velocity_model = pipeline.execute_model_pipeline(
                sites_model(config, pending, depth_m), model_pipeline
            )
            samples_per_profile = int(depth_m // grid.resolution_z) + 1
            # Whole chunks of the borehole grid's station axis per batch.
            batch_size = max(
                CHUNK, rows_per_batch // samples_per_profile // CHUNK * CHUNK
            )
            summary, pending = write_pass(velocity_model, pending, batch_size, writer)
            summaries.append(summary)
            depth_m *= 2

    summary = (
        pd.concat(summaries).set_index(STATION).loc[stations[STATION]].reset_index()
    )
    summary.to_parquet(output_dir / "summary.parquet", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False, float_format=FLOAT_FORMAT)
    logger.info(
        "%d stations, %d deepened past the first pass",
        len(summary),
        int((summary["profile_depth_m"] > summary["profile_depth_m"].min()).sum()),
    )


if __name__ == "__main__":
    app()
