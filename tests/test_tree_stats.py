"""Tests for the ``nzcvm tree-stats`` benchmark.

Natural Earth serves the NZ land polygon. These tests swap it for a small
polygon around a two-tetrahedron mesh.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely
from hypothesis import given
from hypothesis import strategies as st
from pyproj import Transformer
from rich.console import Console
from typer.testing import CliRunner

from nzcvm.models.mesh import mesh_encoding
from nzcvm.models.model import ModelTree
from nzcvm.scripts import tree_stats
from tests.test_model import _tetrahedral_mesh

# Wellington and Auckland in NZTM (EPSG:2193), and their WGS84 coordinates.
NZTM_XS = np.array([1748735.0, 1757000.0])
NZTM_YS = np.array([5427916.0, 5920000.0])
LONS_LATS = Transformer.from_crs("epsg:2193", "epsg:4326", always_xy=True).transform(
    NZTM_XS, NZTM_YS
)

# A land footprint over part of the mesh: some samples hit it, some miss it.
LAND = shapely.MultiPolygon([shapely.box(0.1, 0.1, 0.6, 0.6)])


# ---------------------------------------------------------------------------
# sample_land_points
# ---------------------------------------------------------------------------


@given(n=st.integers(min_value=1, max_value=200))
def test_sample_land_points_lie_on_land(n: int) -> None:
    land = shapely.MultiPolygon(
        [shapely.box(0.0, 0.0, 1.0, 1.0), shapely.box(5.0, 5.0, 6.0, 6.0)]
    )
    min_bounds = np.array([0.0, 0.0, 0.0])
    max_bounds = np.array([10.0, 10.0, 2.0])

    xs, ys, zs = tree_stats.sample_land_points(n, min_bounds, max_bounds, land)

    assert xs.shape == ys.shape == zs.shape == (n,)
    assert shapely.contains_xy(land, xs, ys).all()
    # Depths follow a truncated exponential on [0, max_z].
    assert ((zs >= 0.0) & (zs <= max_bounds[2])).all()


def test_sample_land_points_depths_are_shallow_biased() -> None:
    np.random.seed(0)
    land = shapely.MultiPolygon([shapely.box(0.0, 0.0, 1.0, 1.0)])
    _, _, zs = tree_stats.sample_land_points(
        2000, np.zeros(3), np.array([1.0, 1.0, 10.0]), land
    )
    # Exponential with scale 0.1 truncated far from the mean: mean ~0.1.
    assert np.mean(zs) == pytest.approx(0.1, rel=0.1)
    assert zs.max() <= 10.0


# ---------------------------------------------------------------------------
# save_to_parquet
# ---------------------------------------------------------------------------


def test_save_to_parquet_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "bench.parquet"

    returned = tree_stats.save_to_parquet(
        NZTM_XS, NZTM_YS, [1.5, 2.5], [3, 4], [5, 6], output
    )

    assert returned == output
    df = pd.read_parquet(output)
    assert list(df.columns) == [
        "longitude",
        "latitude",
        "query_time",
        "simplex_tests",
        "aabb_tests",
    ]
    np.testing.assert_allclose(df["longitude"], LONS_LATS[0])
    np.testing.assert_allclose(df["latitude"], LONS_LATS[1])
    # Wellington sanity check: catches swapped axes or a wrong CRS.
    assert df["longitude"][0] == pytest.approx(174.78, abs=0.01)
    assert df["latitude"][0] == pytest.approx(-41.29, abs=0.01)
    assert df["query_time"].tolist() == [1.5, 2.5]
    assert df["simplex_tests"].dtype == np.float64
    assert df["aabb_tests"].dtype == np.float64
    assert df["simplex_tests"].tolist() == [5.0, 6.0]
    assert df["aabb_tests"].tolist() == [3.0, 4.0]


# ---------------------------------------------------------------------------
# get_nz_land_polygon
# ---------------------------------------------------------------------------


def test_get_nz_land_polygon_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "geopandas", None)
    with pytest.raises(ImportError, match="geopandas") as excinfo:
        tree_stats.get_nz_land_polygon()
    message = str(excinfo.value)
    assert "nzcvm tree-stats" in message
    assert "nzcvm[bench]" in message
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_get_nz_land_polygon_is_cached_on_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import platformdirs

    calls = []

    def read_polygon():
        calls.append(1)
        return shapely.box(0.0, 0.0, 1.0, 1.0)

    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda _app: str(tmp_path))
    monkeypatch.setattr(tree_stats, "_read_nz_land_polygon", read_polygon)

    first = tree_stats.get_nz_land_polygon()
    second = tree_stats.get_nz_land_polygon()

    assert first.equals(shapely.box(0.0, 0.0, 1.0, 1.0))
    assert second.equals(first)
    assert len(calls) == 1
    assert any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# run_benchmark and the command line
# ---------------------------------------------------------------------------


@pytest.fixture()
def captured_console(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Capture the benchmark report, and use the small land polygon."""
    buffer = io.StringIO()
    monkeypatch.setattr(tree_stats, "console", Console(file=buffer, width=200))
    monkeypatch.setattr(tree_stats, "get_nz_land_polygon", lambda: LAND)
    return buffer


@pytest.fixture()
def mesh_zarr(tmp_path: Path) -> Path:
    mesh = _tetrahedral_mesh()
    path = tmp_path / "mesh.zarr"
    mesh.to_zarr(path, encoding=mesh_encoding(mesh))
    return path


def _expected_hits(model_path: Path, n: int, seed: int) -> int:
    """Count the hits for the points the benchmark samples with `seed`."""
    model = ModelTree.load_models([model_path])
    np.random.seed(seed)
    xs, ys, zs = tree_stats.sample_land_points(n, *model.aabb, LAND)
    return sum(
        model.query_stats(x, y, z, z).hit_count > 0
        for x, y, z in zip(xs, ys, zs, strict=True)
    )


def test_run_benchmark_report(
    captured_console: io.StringIO, mesh_zarr: Path, tmp_path: Path
) -> None:
    n = 60
    expected_hits = _expected_hits(mesh_zarr, n, seed=1)
    # Points in the land box with x + y + z > 1 miss the first tetrahedron;
    # check that both outcomes occur so the hit rate is informative.
    assert 0 < expected_hits < n

    output = tmp_path / "bench.parquet"
    np.random.seed(1)
    tree_stats.run_benchmark([mesh_zarr], n, output)

    df = pd.read_parquet(output)
    assert len(df) == n
    assert (df["simplex_tests"] >= 1).all()
    assert (df["query_time"] >= 0).all()

    report = captured_console.getvalue()
    assert "Loading 1 model(s)" in report
    # The title wraps to the table's width.
    assert "Realistic NZ Land Performance Profile" in report
    assert f"(N={n})" in report
    assert f"{expected_hits} hits" in report
    assert f"{expected_hits / n * 100:.1f}%" in report
    assert f"{df['simplex_tests'].max():.0f}" in report
    assert "Query Time Distribution" in report
    # Each of the 20 histogram bins prints its count. The counts sum to n.
    bin_rows = [line for line in report.splitlines() if "μ " in line]
    assert len(bin_rows) == 20
    assert sum(int(row.split()[-2]) for row in bin_rows) == n
    # Bars scale so the most frequent bin is 20 characters long.
    assert max(row.count("█") for row in bin_rows) == 20


def test_cli_writes_parquet(captured_console: io.StringIO, tmp_path: Path) -> None:
    # The command line only accepts files, so store the mesh as netCDF. netCDF can't
    # store ``None`` attributes.
    mesh = _tetrahedral_mesh()
    mesh.attrs = {k: v for k, v in mesh.attrs.items() if v is not None}
    mesh_path = tmp_path / "mesh.nc"
    mesh.to_netcdf(mesh_path)
    output = tmp_path / "out.parquet"

    result = CliRunner().invoke(
        tree_stats.app, [str(mesh_path), "-n", "7", "--output", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert len(pd.read_parquet(output)) == 7
    assert "(N=7)" in captured_console.getvalue()


@pytest.mark.parametrize("n", ["0", "-3"])
def test_cli_rejects_non_positive_sample_count(tmp_path: Path, n: str) -> None:
    mesh_path = tmp_path / "mesh.nc"
    mesh_path.touch()
    output = tmp_path / "out.parquet"

    result = CliRunner().invoke(
        tree_stats.app, [str(mesh_path), "-n", n, "--output", str(output)]
    )

    assert result.exit_code == 2
    assert not output.exists()
