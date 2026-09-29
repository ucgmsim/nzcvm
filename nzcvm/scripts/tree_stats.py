"""Benchmark NZCVM BVH performance with realistic land-sampled query points."""

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import scipy as sp
import shapely
import typer
from pyproj import Transformer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from nzcvm.models.model import ModelTree
from nzcvm.scripts._extras import missing_extra

console = Console()

app = typer.Typer(
    help="Benchmark BVH performance with realistic NZ land-sampled queries."
)


def _read_nz_land_polygon():
    import geopandas as gpd

    url = (
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    world = gpd.read_file(url)
    nz = world[world["ADMIN"] == "New Zealand"]
    return nz.to_crs(epsg=2193).geometry.unary_union


def get_nz_land_polygon():
    """Return the NZ land polygon (NZTM), cached on disk between runs."""
    try:
        import geopandas  # noqa: F401
        import platformdirs
        from joblib import Memory
    except ImportError as err:
        raise missing_extra(err.name or "geopandas", "bench", "tree-stats") from err

    memory = Memory(platformdirs.user_cache_dir("nzcvm"), verbose=0)
    return memory.cache(_read_nz_land_polygon)()


def sample_land_points(
    n_required: int,
    min_bounds: np.ndarray,
    max_bounds: np.ndarray,
    nz_poly: shapely.MultiPolygon,
):
    xs_final, ys_final, zs_final = [], [], []

    batch_size = n_required * 5
    shapely.prepare(nz_poly)
    while len(xs_final) < n_required:
        c_x = np.random.uniform(min_bounds[0], max_bounds[0], batch_size)
        c_y = np.random.uniform(min_bounds[1], max_bounds[1], batch_size)
        c_z = sp.stats.truncexpon(b=max_bounds[2] / 0.1, scale=0.1).rvs(size=batch_size)

        points = shapely.multipoints(np.column_stack((c_x, c_y)))
        mask = nz_poly.contains(points.geoms)

        xs_final.extend(c_x[mask])
        ys_final.extend(c_y[mask])
        zs_final.extend(c_z[mask])

    return (
        np.array(xs_final[:n_required]),
        np.array(ys_final[:n_required]),
        np.array(zs_final[:n_required]),
    )


def save_to_parquet(xs, ys, times, aabb_tests, simplex_tests, output_path):
    transformer = Transformer.from_crs("epsg:2193", "epsg:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)

    df = pd.DataFrame(
        {
            "longitude": lons,
            "latitude": lats,
            "query_time": times,
            "simplex_tests": simplex_tests,
            "aabb_tests": aabb_tests,
        }
    )
    df["simplex_tests"] = df["simplex_tests"].astype(float)
    df["aabb_tests"] = df["aabb_tests"].astype(float)
    df.to_parquet(output_path)
    return output_path


def run_benchmark(model_paths: list[Path], n_samples: int, output_path: Path):
    console.print(f"[bold blue]Loading {len(model_paths)} model(s)...[/bold blue]")
    mesh_model = ModelTree.load_models(model_paths)

    min_bounds, max_bounds = mesh_model.aabb
    nz_land = get_nz_land_polygon()

    console.print("[yellow]Generating land-constrained points...[/yellow]")
    xs, ys, zs = sample_land_points(n_samples, min_bounds, max_bounds, nz_land)

    results = []
    times = []
    aabb_tests_list = []
    simplex_tests_list = []
    with Progress() as progress:
        task = progress.add_task("[green]Processing queries...", total=n_samples)
        for i in range(n_samples):
            stats = mesh_model.query_stats(xs[i], ys[i], zs[i])
            results.append(stats)
            times.append(stats.elapsed / 1000)
            aabb_tests_list.append(stats.aabb_tests)
            simplex_tests_list.append(stats.simplex_tests)
            progress.update(task, advance=1)

    console.print(
        f"[bold yellow]Dumping spatial benchmark to {output_path}...[/bold yellow]"
    )
    save_to_parquet(xs, ys, times, aabb_tests_list, simplex_tests_list, output_path)

    total_hits = sum(1 for s in results if s.hit_count > 0)
    times_series = pd.Series(times)
    mean_time = times_series.mean()
    max_time = times_series.max()
    table = Table(title=f"Realistic NZ Land Performance Profile (N={n_samples})")
    table.add_column("Metric", style="magenta")
    table.add_column("Average", justify="right")
    table.add_column("Max", justify="right")
    table.add_row(
        "AABB Tests",
        f"{np.mean([s.aabb_tests for s in results]):.1f}",
        f"{max(s.aabb_tests for s in results)}",
    )
    table.add_row(
        "Simplex Tests",
        f"{np.mean([s.simplex_tests for s in results]):.1f}",
        f"{max(s.simplex_tests for s in results)}",
    )
    table.add_row(
        "ModelTree Hit Rate",
        f"{(total_hits / n_samples) * 100:.1f}%",
        f"{total_hits} hits",
    )
    table.add_row(
        "Query time",
        f"{mean_time:.2f} μs",
        f"{max_time:.2f} μs",
    )

    hist, bin_edges = np.histogram(times_series, bins=20)
    max_freq = max(hist)

    hist_table = Table(box=None, show_header=False, pad_edge=False)
    hist_table.add_column("Bin", justify="right", style="cyan")
    hist_table.add_column("Graph", justify="left")
    hist_table.add_column("Count", justify="right", style="dim")

    for i in range(len(hist)):
        bar_length = int((hist[i] / max_freq) * 20) if max_freq > 0 else 0
        bar = "█" * bar_length
        bin_range = f"{bin_edges[i]:.0f}-{bin_edges[i + 1]:.0f}μ"
        hist_table.add_row(bin_range, bar, str(hist[i]))

    console.print(table)
    console.print(Panel(hist_table, title="Query Time Distribution", expand=False))


@app.command()
def main(
    models: Annotated[
        list[Path],
        typer.Argument(
            help="One or more VTKHDF model files to load.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    n_samples: Annotated[
        int,
        typer.Option(
            "-n", "--n-samples", help="Number of random sample points to query.", min=1
        ),
    ] = 1000,
    output: Annotated[Path, typer.Option(help="Output Parquet path.")] = Path(
        "nzcvm_benchmark.parquet"
    ),
) -> None:
    """Entry point for the ``nzcvm tree-stats`` command."""
    run_benchmark(models, n_samples, output)
