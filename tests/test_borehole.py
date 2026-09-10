"""Tests for the borehole grid: config decoding, site loading and the builder.

Every test builds on the synthetic DEM (:mod:`nzcvm.synthetic`), so nothing
reads a real data file, and the expected elevations follow from the analytic
topography rather than from a fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from mashumaro.exceptions import InvalidFieldValue
from pyproj import CRS, Transformer

from nzcvm import synthetic
from nzcvm.config.grids.borehole import (
    DEFAULT_CHUNK_SIZES,
    BoreholeGridConfig,
    Site,
)
from nzcvm.config.grids.model import Model, Projection
from nzcvm.config.metadata import ModelMetadata
from nzcvm.config.velocity_model import VelocityModelConfig
from nzcvm.coordinates import Coordinate
from nzcvm.formats import Format, write_velocity_model
from nzcvm.grids.borehole import read_sites, resolve_sites
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.grids.grid import Grid
from nzcvm.layers.pipeline import execute_model_pipeline
from nzcvm.models.mesh import StructuredMeshSchema
from nzcvm.qualities import Qualities, QualitiesSchema
from nzcvm.query import ModelRange
from nzcvm.velocity_model import VelocityModel

_NZTM = CRS.from_epsg(2193)
_NZGD2000 = CRS.from_epsg(4167)
_WGS84 = CRS.from_epsg(4326)
_TO_NZTM = Transformer.from_crs(4326, _NZTM, always_xy=True)

# Sites in the hills and out to sea, spread across the domain.
SITES = [
    Site(name="GULL", longitude=172.15, latitude=-43.70),
    Site(name="RIDG", longitude=172.10, latitude=-43.45),
    Site(name="SEAB", longitude=172.55, latitude=-43.60),
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


def _config(
    surface: Path,
    sites: list[Site] | Path = SITES,
    depth: float = 400.0,
    resolution_z: float = 100.0,
    sites_crs: CRS = _WGS84,
    chunks: dict[Coordinate, int] = DEFAULT_CHUNK_SIZES,
) -> BoreholeGridConfig:
    return BoreholeGridConfig(
        surface=surface,
        sites=sites,
        depth=depth,
        resolution_z=resolution_z,
        projection=Projection(crs=_NZTM),
        sites_crs=sites_crs,
        chunks=chunks,
    )


def _build(config: BoreholeGridConfig) -> Grid:
    grids = build_grids_from_config(config)
    assert list(grids) == ["boreholes"]
    return grids["boreholes"]


# ---------------------------------------------------------------------------
# Projection / Model split
# ---------------------------------------------------------------------------


def test_projection_needs_no_origin() -> None:
    """The whole point of splitting Projection out of Model."""
    projection = Projection(crs=_NZTM)
    x, y = projection.from_wgs84.transform(172.0, -43.5)
    assert projection.to_wgs84.transform(x, y) == pytest.approx((172.0, -43.5))


def test_model_is_still_a_projection() -> None:
    model = Model(origin_lon=172.0, origin_lat=-43.5, azimuth=39.0, crs=_NZTM)
    assert isinstance(model, Projection)
    assert model.grid_origin_x == pytest.approx(
        model.from_wgs84.transform(172.0, -43.5)[0]
    )


def test_transformer_from_reaches_the_projection() -> None:
    projection = Projection(crs=_NZTM)
    from_nzgd = projection.transformer_from(_NZGD2000)
    assert from_nzgd.transform(172.0, -43.5) == pytest.approx(
        projection.from_wgs84.transform(172.0, -43.5), abs=1.0
    )


# ---------------------------------------------------------------------------
# Site loading
# ---------------------------------------------------------------------------


def _write_sites(path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "name": [site.name for site in SITES],
            "longitude": [site.longitude for site in SITES],
            "latitude": [site.latitude for site in SITES],
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


def test_read_sites_preserves_file_order(tmp_path: Path) -> None:
    path = tmp_path / "sites.csv"
    pd.DataFrame(
        {"name": ["B", "A"], "longitude": [172.1, 172.2], "latitude": [-43.5, -43.6]}
    ).to_csv(path, index=False)
    assert [site.name for site in read_sites(path)] == ["B", "A"]


def test_read_sites_ignores_extra_columns(tmp_path: Path) -> None:
    path = tmp_path / "sites.csv"
    pd.DataFrame(
        {
            "name": ["A"],
            "longitude": [172.1],
            "latitude": [-43.5],
            "elevation": [12.0],
        }
    ).to_csv(path, index=False)
    assert read_sites(path) == [Site(name="A", longitude=172.1, latitude=-43.5)]


def test_read_sites_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "sites.txt"
    path.write_text("name,longitude,latitude\n")
    with pytest.raises(ValueError, match="expected one of"):
        read_sites(path)


def test_read_sites_reports_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "sites.csv"
    pd.DataFrame({"name": ["A"], "longitude": [172.1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing the latitude column"):
        read_sites(path)


def test_resolve_sites_accepts_inline_sites(synthetic_surface: Path) -> None:
    assert resolve_sites(_config(synthetic_surface)) == SITES


def test_resolve_sites_reads_a_file(synthetic_surface: Path, tmp_path: Path) -> None:
    config = _config(synthetic_surface, sites=_write_sites(tmp_path / "sites.csv"))
    assert resolve_sites(config) == SITES


def test_resolve_sites_rejects_an_empty_file(
    synthetic_surface: Path, tmp_path: Path
) -> None:
    path = tmp_path / "sites.csv"
    path.write_text("name,longitude,latitude\n")
    with pytest.raises(ValueError, match="No sites found"):
        resolve_sites(_config(synthetic_surface, sites=path))


def test_config_rejects_an_empty_inline_site_list(synthetic_surface: Path) -> None:
    with pytest.raises(ValueError, match="at least one site"):
        _config(synthetic_surface, sites=[])


def test_config_rejects_a_projected_sites_crs(synthetic_surface: Path) -> None:
    with pytest.raises(InvalidFieldValue, match="geographic"):
        _config(synthetic_surface, sites_crs=_NZTM)


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------


def test_grid_is_one_column_per_site(synthetic_surface: Path) -> None:
    grid = _build(_config(synthetic_surface))
    assert grid.x.shape == (len(SITES), 1, 5)
    assert grid.sizes[Coordinate.J] == 1


def test_grid_labels_columns_by_site(synthetic_surface: Path) -> None:
    grid = _build(_config(synthetic_surface))
    assert list(grid[Coordinate.SITE].values) == [site.name for site in SITES]


def test_depth_is_identical_across_sites(synthetic_surface: Path) -> None:
    """Comparable profiles are the point: every column shares one depth axis."""
    depth = _build(_config(synthetic_surface)).depth.values
    expected = np.linspace(0.0, 400.0, 5, dtype=np.float32)
    for column in depth.reshape(-1, depth.shape[-1]):
        assert column == pytest.approx(expected)


def test_resolution_z_sets_the_sample_count(synthetic_surface: Path) -> None:
    grid = _build(_config(synthetic_surface, depth=600.0, resolution_z=25.0))
    assert grid.sizes[Coordinate.K] == 25
    assert float(grid.depth.max()) == pytest.approx(600.0)


def test_sites_are_projected_into_the_grid_crs(synthetic_surface: Path) -> None:
    grid = _build(_config(synthetic_surface))
    expected_x, expected_y = _TO_NZTM.transform(
        [site.longitude for site in SITES], [site.latitude for site in SITES]
    )
    assert grid.x.values[:, 0, 0] == pytest.approx(np.float32(expected_x), rel=1e-6)
    assert grid.y.values[:, 0, 0] == pytest.approx(np.float32(expected_y), rel=1e-6)


def test_x_and_y_are_constant_down_each_column(synthetic_surface: Path) -> None:
    grid = _build(_config(synthetic_surface))
    for axis in (grid.x, grid.y):
        assert np.all(axis.values == axis.values[:, :, :1])


def test_columns_start_at_the_topography(synthetic_surface: Path) -> None:
    grid = _build(_config(synthetic_surface))
    # Grids are +z down, so the top of each column is minus the elevation.
    expected = -synthetic.elevation(
        [site.longitude for site in SITES], [site.latitude for site in SITES]
    )
    assert grid.z.values[:, 0, 0] == pytest.approx(expected, abs=15.0)


def test_columns_follow_the_topography_down(synthetic_surface: Path) -> None:
    grid = _build(_config(synthetic_surface))
    top = grid.z.values[:, :, 0]
    bottom = grid.z.values[:, :, -1]
    assert (bottom - top) == pytest.approx(np.full_like(top, 400.0))


def test_a_geographic_sites_crs_lands_in_the_same_place(
    synthetic_surface: Path,
) -> None:
    wgs84 = _build(_config(synthetic_surface))
    nzgd2000 = _build(_config(synthetic_surface, sites_crs=_NZGD2000))
    # NZGD2000 and WGS84 are within a metre of each other over New Zealand.
    assert nzgd2000.x.values == pytest.approx(wgs84.x.values, abs=1.0)


def test_geometry_covers_every_site(synthetic_surface: Path) -> None:
    """The query layer prunes models against this, so it has to hit each site."""
    grid = _build(_config(synthetic_surface))
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


def test_derived_origin_is_the_site_centroid(synthetic_surface: Path) -> None:
    """A borehole grid has no configured origin, so the attributes come from
    the sites. Downstream writers still expect them to be present."""
    grid = _build(_config(synthetic_surface))
    lons = [site.longitude for site in SITES]
    lats = [site.latitude for site in SITES]
    assert grid.origin_lon == pytest.approx(np.mean(lons), abs=1e-2)
    assert grid.origin_lat == pytest.approx(np.mean(lats), abs=1e-2)
    assert grid.bottom_left_lon == pytest.approx(min(lons), abs=1e-2)
    assert grid.bottom_left_lat == pytest.approx(min(lats), abs=1e-2)
    assert grid.azimuth == 0.0
    assert grid.grid_azimuth == 0.0


def test_grid_is_chunked_over_sites(synthetic_surface: Path) -> None:
    grid = _build(_config(synthetic_surface, chunks={Coordinate.I: 2}))
    assert grid.x.chunksizes[Coordinate.I] == (2, 1)
    # The pipeline relies on k staying in one piece.
    assert len(grid.x.chunksizes[Coordinate.K]) == 1


# ---------------------------------------------------------------------------
# Config decoding
# ---------------------------------------------------------------------------

_TOML = """
[grid]
type = "borehole"
surface = "{surface}"
depth = 200.0
resolution_z = 50.0

[grid.projection]
crs = 'EPSG:2193'

[[grid.sites]]
name = "GULL"
longitude = 172.15
latitude = -43.70

[[layers]]
type = "query"
model_path = "{surface}"
"""


def test_toml_config_selects_the_borehole_grid(
    synthetic_surface: Path, tmp_path: Path
) -> None:
    path = tmp_path / "borehole.toml"
    path.write_text(_TOML.format(surface=synthetic_surface))

    config = VelocityModelConfig.read_config(path)
    assert isinstance(config.grid, BoreholeGridConfig)
    assert config.grid.sites == [Site(name="GULL", longitude=172.15, latitude=-43.70)]
    # Unset, so it falls back to WGS84.
    assert config.grid.sites_crs.to_epsg() == 4326
    assert _build(config.grid).sizes[Coordinate.K] == 5


def test_toml_config_reads_sites_from_a_file(
    synthetic_surface: Path, tmp_path: Path
) -> None:
    sites = _write_sites(tmp_path / "sites.csv")
    body = _TOML.format(surface=synthetic_surface)
    body = body[: body.index("[[grid.sites]]")] + body[body.index("[[layers]]") :]
    path = tmp_path / "borehole.toml"
    path.write_text(
        body.replace("resolution_z = 50.0", f'resolution_z = 50.0\nsites = "{sites}"')
    )

    config = VelocityModelConfig.read_config(path)
    assert isinstance(config.grid, BoreholeGridConfig)
    assert config.grid.sites == sites
    assert list(_build(config.grid)[Coordinate.SITE].values) == [
        site.name for site in SITES
    ]


# ---------------------------------------------------------------------------
# Through the pipeline
# ---------------------------------------------------------------------------


def _uniform(grid: Grid, model_range: ModelRange = ModelRange.ALL) -> Qualities:
    """A terminal layer that keeps the grid's coordinates.

    `ConstantLayer` builds its output from raw NumPy and so drops them, which
    `map_blocks` rejects for any grid that carries dimension coordinates.
    """
    ones = xr.ones_like(grid.x)
    return QualitiesSchema.new(
        rho=ones * 2700.0,
        vp=ones * 6000.0,
        vs=ones * 1234.0,
        qp=ones * 200.0,
        qs=ones * 100.0,
        alpha=ones,
    )


def _run(grid: Grid) -> VelocityModel:
    model = VelocityModel(grids={"boreholes": grid}, metadata=ModelMetadata())
    return execute_model_pipeline(model, _uniform)


def test_grid_survives_the_chunked_pipeline(synthetic_surface: Path) -> None:
    """`execute_model_pipeline` maps over the chunks, so `map_blocks` has to
    preserve both the singleton j axis and the site labels."""
    grid = _build(_config(synthetic_surface, chunks={Coordinate.I: 2}))
    result = _run(grid)

    qualities = result.qualities["boreholes"]
    assert qualities.vs.shape == grid.x.shape
    assert float(qualities.vs.values.mean()) == pytest.approx(1234.0, rel=1e-4)
    assert list(result.grids["boreholes"][Coordinate.SITE].values) == [
        site.name for site in SITES
    ]


def test_output_round_trips_through_zarr(
    synthetic_surface: Path, tmp_path: Path
) -> None:
    """A profile is only useful when a reader can pick out one station."""
    path = tmp_path / "boreholes.zarr"
    write_velocity_model(
        _run(_build(_config(synthetic_surface))),
        path,
        Format.ZARR,
        quantise_arrays=False,
    )

    with xr.open_datatree(path, engine="zarr") as tree:
        stored = tree["grids/boreholes"].ds
        assert list(stored[Coordinate.SITE].values) == [site.name for site in SITES]
        gull = stored.set_xindex(Coordinate.SITE).sel({Coordinate.SITE: "GULL"})
        assert float(gull.depth.max()) == pytest.approx(400.0)
        assert tree["qualities/boreholes"].ds.vs.shape == stored.x.shape
