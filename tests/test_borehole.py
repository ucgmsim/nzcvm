"""Tests for the borehole grid: config decoding, site loading and the builder.

Every test builds on the synthetic DEM (:mod:`nzcvm.synthetic`), so the
expected elevations follow from the analytic topography rather than a fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from mashumaro.exceptions import InvalidFieldValue
from pyproj import CRS, Transformer

from nzcvm import synthetic
from nzcvm.config.grids.borehole import BoreholeGridConfig, Site
from nzcvm.config.grids.model import Projection
from nzcvm.config.metadata import ModelMetadata
from nzcvm.config.velocity_model import VelocityModelConfig
from nzcvm.coordinates import Coordinate
from nzcvm.formats import Format, write_velocity_model
from nzcvm.grids.borehole import read_sites, resolve_sites
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.grids.grid import Grid
from nzcvm.layers.dummy import constant
from nzcvm.layers.pipeline import execute_model_pipeline
from nzcvm.models.mesh import StructuredMeshSchema
from nzcvm.velocity_model import VelocityModel

_NZTM = CRS.from_epsg(2193)
_NZGD2000 = CRS.from_epsg(4167)
_TO_NZTM = Transformer.from_crs(4326, _NZTM, always_xy=True)

# Sites in the hills and out to sea, spread across the domain.
SITES = [
    Site(longitude=172.15, latitude=-43.70, labels={"site": "GULL", "network": "NZ"}),
    Site(longitude=172.10, latitude=-43.45, labels={"site": "RIDG", "network": "NZ"}),
    Site(longitude=172.55, latitude=-43.60, labels={"site": "SEAB", "network": "SC"}),
]
NAMES = [site.labels["site"] for site in SITES]

DEPTH = 400.0
RESOLUTION_Z = 100.0
NK = int(DEPTH / RESOLUTION_Z) + 1

# Sites that disagree on their labels.
MISMATCHED = [
    Site(
        longitude=SITES[0].longitude, latitude=SITES[0].latitude, labels={"site": "A"}
    ),
    Site(longitude=SITES[1].longitude, latitude=SITES[1].latitude, labels={}),
]


@pytest.fixture(scope="module")
def synthetic_surface(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic topography, written as a Zarr surface mesh."""
    lon, lat = synthetic.DOMAIN.sample(32, 32)
    mesh_lon, mesh_lat = np.meshgrid(lon, lat, indexing="ij")
    x, y = _TO_NZTM.transform(mesh_lon, mesh_lat)
    # A surface file uses +z down, the same convention as the grids.
    z = -synthetic.elevation(mesh_lon, mesh_lat)

    path = tmp_path_factory.mktemp("surfaces") / "dem.zarr"
    StructuredMeshSchema.new(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        z=z.astype(np.float32),
        i=np.arange(32),
        j=np.arange(32),
        name="synthetic",
    ).to_zarr(path, mode="w")
    return path


def _config(surface: Path, **overrides: Any) -> BoreholeGridConfig:
    defaults: dict[str, Any] = {
        "surface": surface,
        "sites": SITES,
        "depth": DEPTH,
        "resolution_z": RESOLUTION_Z,
        "projection": Projection(crs=_NZTM),
    }
    return BoreholeGridConfig(**(defaults | overrides))


def _build(config: BoreholeGridConfig) -> Grid:
    grids = build_grids_from_config(config)
    assert list(grids) == ["boreholes"]
    return grids["boreholes"]


def _relabel(*labels: dict[str, Any]) -> list[Site]:
    """The default sites, carrying *labels* instead of their own."""
    return [
        Site(longitude=site.longitude, latitude=site.latitude, labels=label)
        for site, label in zip(SITES, labels, strict=True)
    ]


@pytest.fixture(scope="module")
def grid(synthetic_surface: Path) -> Grid:
    """The default grid, built once. No test mutates it."""
    return _build(_config(synthetic_surface))


# ---------------------------------------------------------------------------
# Site loading
# ---------------------------------------------------------------------------


def _write_sites(path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "longitude": [site.longitude for site in SITES],
            "latitude": [site.latitude for site in SITES],
            "site": NAMES,
            "network": [site.labels["network"] for site in SITES],
        }
    )
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path)
    return path


@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".pq"])
def test_read_sites_round_trips(tmp_path: Path, suffix: str) -> None:
    assert read_sites(_write_sites(tmp_path / f"sites{suffix}")) == SITES


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        pytest.param(
            {
                "longitude": [172.1, 172.2],
                "latitude": [-43.5, -43.6],
                "site": ["B", "A"],
            },
            [
                Site(longitude=172.1, latitude=-43.5, labels={"site": "B"}),
                Site(longitude=172.2, latitude=-43.6, labels={"site": "A"}),
            ],
            id="file-order-is-kept",
        ),
        pytest.param(
            {
                "longitude": [172.1],
                "latitude": [-43.5],
                "site": ["A"],
                "elevation": [12.0],
            },
            [
                Site(
                    longitude=172.1,
                    latitude=-43.5,
                    labels={"site": "A", "elevation": 12.0},
                )
            ],
            id="extra-columns-become-labels",
        ),
        pytest.param(
            {"longitude": [172.1], "latitude": [-43.5]},
            [Site(longitude=172.1, latitude=-43.5, labels={})],
            id="labels-are-optional",
        ),
    ],
)
def test_read_sites(tmp_path: Path, columns: dict, expected: list[Site]) -> None:
    """Longitude and latitude are the only reserved columns."""
    path = tmp_path / "sites.csv"
    pd.DataFrame(columns).to_csv(path, index=False)
    assert read_sites(path) == expected


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("sites.txt", "site,longitude,latitude\n", "expected one of"),
        ("sites.csv", "site,longitude\nA,172.1\n", "missing the latitude"),
    ],
)
def test_read_sites_rejects_bad_files(
    tmp_path: Path, name: str, text: str, message: str
) -> None:
    path = tmp_path / name
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        read_sites(path)


def test_resolve_sites_accepts_inline_sites(synthetic_surface: Path) -> None:
    assert resolve_sites(_config(synthetic_surface)) == SITES


def test_resolve_sites_reads_a_file(synthetic_surface: Path, tmp_path: Path) -> None:
    config = _config(synthetic_surface, sites=_write_sites(tmp_path / "sites.csv"))
    assert resolve_sites(config) == SITES


@pytest.mark.parametrize(
    ("override", "error", "message"),
    [
        pytest.param({"sites": []}, ValueError, "at least one site", id="empty-list"),
        pytest.param(
            {"sites_crs": _NZTM}, InvalidFieldValue, "geographic", id="projected-crs"
        ),
    ],
)
def test_config_rejects_bad_sites(
    synthetic_surface: Path, override: dict, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _config(synthetic_surface, **override)


def test_resolve_sites_rejects_an_empty_file(
    synthetic_surface: Path, tmp_path: Path
) -> None:
    path = tmp_path / "sites.csv"
    path.write_text("name,longitude,latitude\n")
    with pytest.raises(ValueError, match="No sites found"):
        resolve_sites(_config(synthetic_surface, sites=path))


# ---------------------------------------------------------------------------
# Site labels
#
# Longitude and latitude place a site. Every other key or column is a label
# the grid passes through to the output under the name the caller gave it.
# ---------------------------------------------------------------------------


def test_extra_config_keys_become_labels() -> None:
    assert Site.from_dict(
        {"longitude": 172.1, "latitude": -43.5, "site": "A", "depth_drilled": 30.0}
    ).labels == {"site": "A", "depth_drilled": 30.0}
    assert Site.from_dict({"longitude": 172.1, "latitude": -43.5}).labels == {}


def test_labels_become_coordinates_on_the_site_axis(grid: Grid) -> None:
    for label in ("site", "network"):
        assert grid[label].dims == (Coordinate.I,)
        assert list(grid[label].values) == [site.labels[label] for site in SITES]


def test_numeric_labels_keep_a_numeric_dtype(synthetic_surface: Path) -> None:
    """A label is whatever the caller wrote, not necessarily a string."""
    sites = _relabel(*({"cased": n * 10} for n in range(len(SITES))))
    labelled = _build(_config(synthetic_surface, sites=sites))
    assert labelled["cased"].dtype.kind in "iuf"
    assert list(labelled["cased"].values) == [0, 10, 20]


def test_keep_extra_columns_false_drops_the_labels(synthetic_surface: Path) -> None:
    bare = _build(_config(synthetic_surface, keep_extra_columns=False))
    assert set(bare.coords) == {Coordinate.I, Coordinate.J, Coordinate.K}
    # The opt-out leaves the spatial coordinates alone.
    assert bare.x.shape == (len(SITES), 1, NK)


@pytest.mark.parametrize("reserved", ["name", "x", "depth", "vs", "i", "geometry"])
def test_a_label_may_not_shadow_a_grid_name(
    synthetic_surface: Path, reserved: str
) -> None:
    """A coordinate shadows a variable or attribute of the same name, so
    `grid.name` would stop being the grid's name."""
    sites = _relabel(*([{reserved: "x"}] * len(SITES)))
    with pytest.raises(ValueError, match="would shadow"):
        _build(_config(synthetic_surface, sites=sites))


def test_sites_have_to_agree_on_their_labels(synthetic_surface: Path) -> None:
    """A missing label is nearly always a typo, and the alternative is a
    column of nulls."""
    with pytest.raises(ValueError, match="Every site needs the same labels"):
        _build(_config(synthetic_surface, sites=MISMATCHED))


def test_disagreement_is_allowed_once_labels_are_dropped(
    synthetic_surface: Path,
) -> None:
    config = _config(synthetic_surface, sites=MISMATCHED, keep_extra_columns=False)
    assert _build(config).x.shape == (len(MISMATCHED), 1, NK)


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------


def test_grid_is_one_column_per_site(grid: Grid) -> None:
    assert grid.x.shape == (len(SITES), 1, NK)
    assert grid.sizes[Coordinate.J] == 1
    assert list(grid["site"].values) == NAMES


def test_resolution_z_sets_the_sample_count(synthetic_surface: Path) -> None:
    finer = _build(_config(synthetic_surface, depth=600.0, resolution_z=25.0))
    assert finer.sizes[Coordinate.K] == 25
    assert float(finer.depth.max()) == pytest.approx(600.0)


def test_columns_sit_over_the_projected_site(grid: Grid) -> None:
    expected_x, expected_y = _TO_NZTM.transform(
        [site.longitude for site in SITES], [site.latitude for site in SITES]
    )
    assert grid.x.values[:, 0, 0] == pytest.approx(np.float32(expected_x), rel=1e-6)
    assert grid.y.values[:, 0, 0] == pytest.approx(np.float32(expected_y), rel=1e-6)
    # x and y are constant down each column.
    for axis in (grid.x, grid.y):
        assert np.all(axis.values == axis.values[:, :, :1])


def test_columns_run_from_the_topography_down_to_depth(grid: Grid) -> None:
    """Comparable profiles need one shared depth axis. A column starts at the
    ground and ends `depth` below it."""
    depth = grid.depth.values
    assert np.all(depth == depth[:1, :1, :])
    assert depth[0, 0, 0] == pytest.approx(0.0)
    assert depth.max() == pytest.approx(DEPTH)

    # Grids are +z down, so the top of each column is minus the elevation.
    expected = -synthetic.elevation(
        [site.longitude for site in SITES], [site.latitude for site in SITES]
    )
    top = grid.z.values[:, :, 0]
    assert top[:, 0] == pytest.approx(expected, abs=15.0)
    assert (grid.z.values[:, :, -1] - top) == pytest.approx(np.full_like(top, DEPTH))


def test_a_geographic_sites_crs_lands_in_the_same_place(
    synthetic_surface: Path, grid: Grid
) -> None:
    nzgd2000 = _build(_config(synthetic_surface, sites_crs=_NZGD2000))
    # NZGD2000 and WGS84 are within a metre of each other over New Zealand.
    assert nzgd2000.x.values == pytest.approx(grid.x.values, abs=1.0)


def test_geometry_covers_every_site(grid: Grid) -> None:
    """The query layer prunes models against this, so it has to hit each site."""
    assert len(grid.geometry.geoms) == len(SITES)
    assert grid.geometry.bounds == pytest.approx(
        (
            float(grid.x.min()),
            float(grid.y.min()),
            float(grid.x.max()),
            float(grid.y.max()),
        ),
        abs=1.0,
    )


def test_derived_origin_is_the_site_centroid(grid: Grid) -> None:
    """A borehole grid has no configured origin. The attributes come from the
    sites, since downstream writers still expect them to be present."""
    lons = [site.longitude for site in SITES]
    lats = [site.latitude for site in SITES]
    assert grid.origin_lon == pytest.approx(np.mean(lons), abs=1e-2)
    assert grid.origin_lat == pytest.approx(np.mean(lats), abs=1e-2)
    assert grid.bottom_left_lon == pytest.approx(min(lons), abs=1e-2)
    assert grid.bottom_left_lat == pytest.approx(min(lats), abs=1e-2)
    assert grid.azimuth == 0.0
    assert grid.grid_azimuth == 0.0


def test_grid_is_chunked_over_sites(synthetic_surface: Path) -> None:
    chunked = _build(_config(synthetic_surface, chunks={Coordinate.I: 2}))
    assert max(chunked.x.chunksizes[Coordinate.I]) <= 2
    # The pipeline relies on k staying in one piece.
    assert len(chunked.x.chunksizes[Coordinate.K]) == 1


# ---------------------------------------------------------------------------
# Config decoding
# ---------------------------------------------------------------------------

_TOML = """
[grid]
type = "borehole"
surface = "{surface}"
depth = 200.0
resolution_z = 50.0
{sites}

[grid.projection]
crs = 'EPSG:2193'

[[layers]]
type = "query"
model_path = "{surface}"
"""

_INLINE_SITES = """
[[grid.sites]]
longitude = 172.15
latitude = -43.70
site = "GULL"
network = "NZ"
"""


def _decode_grid(path: Path, surface: Path, sites: str) -> BoreholeGridConfig:
    """The grid a written TOML config decodes to."""
    path.write_text(_TOML.format(surface=surface, sites=sites))
    grid = VelocityModelConfig.read_config(path).grid
    assert isinstance(grid, BoreholeGridConfig)
    return grid


def test_toml_config_selects_the_borehole_grid(
    synthetic_surface: Path, tmp_path: Path
) -> None:
    config = _decode_grid(tmp_path / "borehole.toml", synthetic_surface, _INLINE_SITES)

    assert config.sites == [
        Site(
            longitude=172.15,
            latitude=-43.70,
            labels={"site": "GULL", "network": "NZ"},
        )
    ]
    # Unset, so it falls back to WGS84.
    assert config.sites_crs.to_epsg() == 4326
    assert _build(config).sizes[Coordinate.K] == 5


def test_toml_config_reads_sites_from_a_file(
    synthetic_surface: Path, tmp_path: Path
) -> None:
    sites = _write_sites(tmp_path / "sites.csv")
    config = _decode_grid(
        tmp_path / "borehole.toml", synthetic_surface, f'sites = "{sites}"'
    )

    assert config.sites == sites
    assert list(_build(config)["site"].values) == NAMES


# ---------------------------------------------------------------------------
# Through the pipeline
# ---------------------------------------------------------------------------


def _run(grid: Grid) -> VelocityModel:
    model = VelocityModel(grids={"boreholes": grid}, metadata=ModelMetadata())
    return execute_model_pipeline(model, constant(vs=1234.0))


def test_grid_survives_the_chunked_pipeline(synthetic_surface: Path) -> None:
    """`execute_model_pipeline` maps over the chunks, so `map_blocks` has to
    preserve both the singleton j axis and the site labels."""
    chunked = _build(_config(synthetic_surface, chunks={Coordinate.I: 2}))
    result = _run(chunked)

    qualities = result.qualities["boreholes"]
    assert qualities.vs.shape == chunked.x.shape
    assert float(qualities.vs.values.mean()) == pytest.approx(1234.0, rel=1e-4)
    assert list(result.grids["boreholes"]["site"].values) == NAMES


def test_output_round_trips_through_zarr(grid: Grid, tmp_path: Path) -> None:
    """A profile is only useful when a reader can pick out one station."""
    path = tmp_path / "boreholes.zarr"
    write_velocity_model(_run(grid), path, Format.ZARR, quantise_arrays=False)

    with xr.open_datatree(path, engine="zarr") as tree:
        stored = tree["grids/boreholes"].ds
        assert list(stored["site"].values) == NAMES
        gull = stored.set_xindex("site").sel({"site": "GULL"})
        assert float(gull.depth.max()) == pytest.approx(DEPTH)
        assert tree["qualities/boreholes"].ds.vs.shape == stored.x.shape
