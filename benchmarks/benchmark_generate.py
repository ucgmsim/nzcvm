"""Pipeline benchmark: grid construction + constant-layer execution.

Builds a multi-resolution SW4 curvilinear grid and times pipeline
evaluation (lazy graph construction + dask compute) using the
:func:`constant` functional layer as the terminal layer. The benchmark
constructs the pipeline and evaluates it directly for each concrete grid,
rather than exercising the ``VelocityModel`` → ``execute_model_pipeline``
wrapper path used by production pipelines.

Run with::

    python benchmarks/benchmark_generate.py

The script prints wall-clock timings as a Markdown table suitable for
``$GITHUB_STEP_SUMMARY``.
"""

import tempfile
import time
from pathlib import Path

import numpy as np
from pyproj import CRS

from nzcvm.config.grids.model import Model
from nzcvm.config.grids.sw4 import MeshRefinement, SW4GridConfig
from nzcvm.config.metadata import ModelMetadata
from nzcvm.coordinates import Coordinate
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.layers.dummy import constant
from nzcvm.models.mesh import StructuredMeshSchema
from nzcvm.velocity_model import VelocityModel

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

EXTENT_X = 100_000.0  # metres
EXTENT_Y = 80_000.0  # metres

REFINEMENTS = {
    "near_surface": MeshRefinement(resolution=200.0, bottom=2_000.0),
    "mid_crust": MeshRefinement(resolution=400.0, bottom=20_000.0),
    "lower_crust": MeshRefinement(resolution=800.0, bottom=60_000.0),
}

CHUNK_SIZE = 64  # voxels per chunk along i/j, with k always one chunk


# ---------------------------------------------------------------------------
# Flat-surface helper
# ---------------------------------------------------------------------------


def _write_flat_surface(path: Path) -> None:
    """Write a flat z=0 StructuredGrid to *path* (VTK format)."""
    # NZ-wide bounding box in NZTM2000: origin roughly (1e6, 4.7e6)
    xmin, ymin = 1_000_000.0, 4_700_000.0
    extent = 2_000_000.0  # 2000 km side, which encloses the benchmark domain
    n = 10
    xs = np.linspace(xmin, xmin + extent, n)
    ys = np.linspace(ymin, ymin + extent, n)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = np.zeros_like(xx)
    mesh = StructuredMeshSchema.new(
        x=xx, y=yy, z=zz, i=np.arange(n), j=np.arange(n), name="float"
    )
    mesh.to_zarr(path)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark() -> None:
    """Build grids, run the pipeline, compute results, and print a table."""

    with tempfile.TemporaryDirectory() as tmp:
        surface_path = Path(tmp) / "flat.zarr"
        _write_flat_surface(surface_path)

        # 1. Build SW4 grids -------------------------------------------------
        t0 = time.perf_counter()
        config = SW4GridConfig(
            surface=surface_path,
            extent_x=EXTENT_X,
            extent_y=EXTENT_Y,
            orientation=Model(
                origin_lon=172.0,
                origin_lat=-41.0,
                azimuth=0.0,
                crs=CRS.from_epsg(2193),
            ),
            refinements=REFINEMENTS,
            chunks={
                Coordinate.I: CHUNK_SIZE,
                Coordinate.J: CHUNK_SIZE,
            },
        )
        grids = build_grids_from_config(config)
        t1 = time.perf_counter()
        grid_time = t1 - t0

        metadata = ModelMetadata(title="benchmark")
        vm = VelocityModel(grids=grids, metadata=metadata)

        pipeline = constant(
            rho=2700.0, vp=6000.0, vs=3500.0, qp=200.0, qs=100.0, alpha=1.0
        )

        t2 = time.perf_counter()
        all_qualities = {}
        for name, grid in vm.grids.items():
            concrete = grid.compute()
            all_qualities[name] = pipeline(concrete)
        t3 = time.perf_counter()
        compute_time = t3 - t2

        total_elements = sum(q.rho.size for q in all_qualities.values())
        total_gb = total_elements * 6 * 4 / 1024**3  # 6 components × float32

    throughput = total_gb / max(compute_time, 1e-9)

    grid_shapes = {name: str(g.x.shape) for name, g in grids.items()}
    shapes_str = ", ".join(f"`{n}` {s}" for n, s in grid_shapes.items())

    print("### 📊 Benchmark Results: pipeline execution")
    print()
    print(f"**Grids**: {shapes_str}")
    print()
    print("| Metric | Value |")
    print("| :--- | :--- |")
    print(f"| **Total Quality Data** | {total_gb:.2f} GB |")
    print(f"| **Grid Construction** | {grid_time:.3f} s |")
    print(f"| **Pipeline + Compute** | {compute_time:.3f} s |")
    print(f"| **Throughput** | **{throughput:.2f} GB/s** |")
    print()
    print(f"*Benchmark run on {time.strftime('%Y-%m-%d %H:%M:%S')}*")


if __name__ == "__main__":
    run_benchmark()
