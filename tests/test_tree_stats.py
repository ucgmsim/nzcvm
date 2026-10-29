"""Tests for the ``nzcvm tree-stats`` BVH benchmark.

The land polygon normally comes from Natural Earth over the network, so
these tests stub the download and use small hand-made polygons around a
one-tetrahedron mesh written to zarr.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import platformdirs
import pytest
import shapely
from rich.console import Console

from nzcvm.models.mesh import ModelType, Vertical, make_mesh, mesh_encoding
from nzcvm.scripts import tree_stats

# NZTM origin of the test mesh, placing the samples at real NZ coordinates.
X0, Y0 = 1_570_000.0, 5_180_000.0
SIDE = 1000.0


def _write_tetrahedron(path: Path) -> Path:
    """Write the right-angled tetrahedron with legs ``SIDE`` at ``(X0, Y0, 0)``."""
    points = np.array(
        [
            [X0, Y0, 0.0],
            [X0 + SIDE, Y0, 0.0],
            [X0, Y0 + SIDE, 0.0],
            [X0, Y0, SIDE],
        ],
        dtype=np.float32,
    )
    connectivity = np.array([[0, 1, 2, 3]], dtype=np.uint64)
    ones = np.ones(len(points), dtype=np.float32)
    mesh = make_mesh(
        name="tetrahedron",
        vertical=Vertical.ELEVATION,
        aabb=[*points.min(axis=0), *points.max(axis=0)],
        points=points,
        connectivity=connectivity,
        cell_data={
            "model_type": np.full(1, ModelType.INTERPOLATE, dtype=np.uint8),
            "priority": np.zeros(1, dtype=np.uint8),
        },
        field_data={k: ones for k in ("rho", "vp", "vs", "qp", "qs", "alpha")},
        geometry=None,
    )
    mesh.to_zarr(path, encoding=mesh_encoding(mesh))
    return path


def _square(lo: float, hi: float) -> shapely.MultiPolygon:
    """A square covering ``[lo, hi]`` of the mesh's legs, as a MultiPolygon."""
    return shapely.MultiPolygon(
        [shapely.box(X0 + lo * SIDE, Y0 + lo * SIDE, X0 + hi * SIDE, Y0 + hi * SIDE)]
    )


@pytest.fixture()
def captured_console(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Redirect the module console to a buffer wide enough for its tables."""
    buffer = io.StringIO()
    monkeypatch.setattr(tree_stats, "console", Console(file=buffer, width=200))
    return buffer


# ---------------------------------------------------------------------------
# Land polygon
# ---------------------------------------------------------------------------


def test_read_nz_land_polygon_keeps_nz_in_nztm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nz_lonlat = shapely.box(172.0, -44.0, 173.0, -43.0)
    world = gpd.GeoDataFrame(
        {
            "ADMIN": ["Australia", "New Zealand"],
            "geometry": [shapely.box(140.0, -35.0, 150.0, -25.0), nz_lonlat],
        },
        crs="EPSG:4326",
    )
    urls = []

    def read_file(url: str) -> gpd.GeoDataFrame:
        urls.append(url)
        return world

    monkeypatch.setattr(gpd, "read_file", read_file)

    with pytest.warns(DeprecationWarning, match="unary_union"):
        land = tree_stats._read_nz_land_polygon()

    assert urls == [
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    ]
    expected = gpd.GeoSeries([nz_lonlat], crs="EPSG:4326").to_crs(epsg=2193)[0]
    # The query drops Australia and reprojects New Zealand to NZTM metres.
    assert land.equals_exact(expected, tolerance=1e-6)
    minx, miny, maxx, maxy = land.bounds
    assert 1_400_000 < minx < maxx < 1_700_000
    assert 5_000_000 < miny < maxy < 5_300_000


@pytest.mark.parametrize("module", ["geopandas", "platformdirs", "joblib"])
def test_missing_bench_extra_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError) as excinfo:
        tree_stats.get_nz_land_polygon()
    message = str(excinfo.value)
    assert message.startswith(f"{module} is required for `nzcvm tree-stats`")
    assert "pip install 'nzcvm[bench]'" in message
    assert "uv sync --extra bench" in message
    assert isinstance(excinfo.value.__cause__, ImportError)


_reads: list[int] = []


def _fake_read() -> shapely.MultiPolygon:
    _reads.append(1)
    return _square(0.0, 1.0)


def test_land_polygon_is_cached_between_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(tree_stats, "_read_nz_land_polygon", _fake_read)
    _reads.clear()

    first = tree_stats.get_nz_land_polygon()
    second = tree_stats.get_nz_land_polygon()

    assert first.equals(_square(0.0, 1.0))
    assert second.equals(first)
    assert _reads == [1]
    # The cache is written to the nzcvm user cache directory.
    assert any((tmp_path / "nzcvm").rglob("*"))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_required", [1, 7, 200])
def test_sample_land_points_lie_on_land(n_required: int) -> None:
    np.random.seed(0)
    land = _square(0.25, 0.5)
    # A wide bounding box, so most candidates fall in the sea and the sampler
    # has to discard them.
    min_bounds = np.array([X0, Y0, 0.0])
    max_bounds = np.array([X0 + SIDE, Y0 + SIDE, 5.0])

    xs, ys, zs = tree_stats.sample_land_points(n_required, min_bounds, max_bounds, land)

    assert xs.shape == ys.shape == zs.shape == (n_required,)
    assert shapely.contains_xy(land, xs, ys).all()
    assert ((zs >= 0.0) & (zs <= 5.0)).all()


def test_sample_land_points_depths_concentrate_near_surface() -> None:
    np.random.seed(1)
    min_bounds = np.array([X0, Y0, 0.0])
    max_bounds = np.array([X0 + SIDE, Y0 + SIDE, 100.0])

    _, _, zs = tree_stats.sample_land_points(
        2000, min_bounds, max_bounds, _square(0.0, 1.0)
    )

    # A truncated exponential with a 0.1 m scale: mean ~0.1, nothing past max.
    assert zs.mean() == pytest.approx(0.1, rel=0.15)
    assert zs.max() <= 100.0


# ---------------------------------------------------------------------------
# Parquet output
# ---------------------------------------------------------------------------


def test_save_to_parquet_converts_to_lonlat(tmp_path: Path) -> None:
    output = tmp_path / "bench.parquet"
    # The NZTM projection origin: 173°E on the equator.
    xs = np.array([1_600_000.0, X0])
    ys = np.array([10_000_000.0, Y0])

    returned = tree_stats.save_to_parquet(xs, ys, [1.5, 2.5], [3, 4], [5, 6], output)

    assert returned == output
    df = pd.read_parquet(output)
    assert list(df.columns) == [
        "longitude",
        "latitude",
        "query_time",
        "simplex_tests",
        "aabb_tests",
    ]
    assert df["longitude"].iloc[0] == pytest.approx(173.0)
    assert df["latitude"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    # Canterbury, not swapped axes.
    assert 171.0 < df["longitude"].iloc[1] < 173.0
    assert -44.0 < df["latitude"].iloc[1] < -43.0
    assert df["query_time"].tolist() == [1.5, 2.5]
    assert df["aabb_tests"].tolist() == [3.0, 4.0]
    assert df["simplex_tests"].tolist() == [5.0, 6.0]
    assert df["aabb_tests"].dtype == np.float64
    assert df["simplex_tests"].dtype == np.float64


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("land", "hits"),
    [
        # Near the right angle the tetrahedron covers the shallow samples.
        (_square(0.0, 0.3), True),
        # Past the hypotenuse, inside the bounding box but outside the mesh.
        (_square(0.6, 0.9), False),
    ],
)
def test_run_benchmark_reports_hits_and_writes_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    captured_console: io.StringIO,
    land: shapely.MultiPolygon,
    hits: bool,
) -> None:
    np.random.seed(2)
    model = _write_tetrahedron(tmp_path / "tetrahedron.zarr")
    monkeypatch.setattr(tree_stats, "get_nz_land_polygon", lambda: land)
    output = tmp_path / "bench.parquet"
    n_samples = 25

    tree_stats.run_benchmark([model], n_samples, output)

    df = pd.read_parquet(output)
    assert len(df) == n_samples
    # The report converts elapsed nanoseconds to microseconds.
    assert (df["query_time"] > 0).all()
    assert (df["query_time"] < 1e6).all()
    # With one leaf, the tree tests the tetrahedron and no inner boxes.
    assert (df["aabb_tests"] == 0.0).all()
    assert (df["simplex_tests"] == 1.0).all()
    # The benchmark reprojected the samples from the land polygon to lon/lat.
    assert df["longitude"].between(171.0, 173.0).all()
    assert df["latitude"].between(-44.0, -43.0).all()

    out = captured_console.getvalue()
    assert "Loading 1 model(s)" in out
    # The title wraps to the table's width.
    assert "Realistic NZ Land Performance Profile" in out
    assert f"(N={n_samples})" in out
    assert "Query Time Distribution" in out
    lines = out.splitlines()

    def row(metric: str) -> list[str]:
        (line,) = [line for line in lines if f"│ {metric} " in line]
        return [cell.strip() for cell in line.split("│")[2:4]]

    assert row("AABB Tests") == ["0.0", "0"]
    assert row("Simplex Tests") == ["1.0", "1"]
    if hits:
        assert row("ModelTree Hit Rate") == ["100.0%", f"{n_samples} hits"]
    else:
        assert row("ModelTree Hit Rate") == ["0.0%", "0 hits"]
    assert row("Query time") == [
        f"{df['query_time'].mean():.2f} μs",
        f"{df['query_time'].max():.2f} μs",
    ]
    # Twenty histogram bins counting every sample; the fullest gets the
    # full 20-character bar and no bar is longer.
    bins = [line for line in lines if line.startswith("│") and "μ " in line]
    bins = [line for line in bins if "Query time" not in line]
    assert len(bins) == 20
    counts = [int(line.split()[-2]) for line in bins]
    assert sum(counts) == n_samples
    bars = [line.count("█") for line in bins]
    assert max(bars) == 20
    for count, bar in zip(counts, bars, strict=True):
        assert bar == int(count / max(counts) * 20)
