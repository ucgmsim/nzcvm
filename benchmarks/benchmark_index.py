"""Is compiling the BVH to disk worth it? Measure rather than guess.

Builds a synthetic tomography mesh of a chosen size, then times the two ways
of getting a queryable model out of it:

* load the zarr and build the BVH in memory (what every process did before);
* open a compiled ``.nzidx`` and let the pages fault in on demand.

Both are then queried over the same random points so that the mapped model's
first pass (faulting pages in) and second pass (page cache warm) sit next to
the in-memory numbers.

Run from the repository root::

    uv run python benchmarks/benchmark_index.py --n-horizontal 120 --n-depth 40

The default size gives about a million tetrahedra, which is enough to see the
shape of the answer without needing the real models.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from nzcvm import synthetic
from nzcvm.models.model import MB, MeshModel, ModelTree, index_path
from nzcvm.scripts.convert_tomography import (
    DEFAULT_ENCODING_SETTINGS,
    MODEL_COLUMNS,
    ModelType,
    data_frame_to_mesh,
)


def timed(label: str, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"{label:48s} {elapsed * 1000:10.1f} ms")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-horizontal", type=int, default=120)
    parser.add_argument("--n-depth", type=int, default=40)
    parser.add_argument("--points", type=int, default=200_000)
    parser.add_argument("--keep", type=Path, help="Keep the mesh and index here.")
    args = parser.parse_args()

    root = args.keep or Path(tempfile.mkdtemp(prefix="nzcvm-index-bench-"))
    root.mkdir(parents=True, exist_ok=True)
    mesh_path = root / "tomography.zarr"

    frame = synthetic.tomography(n_horizontal=args.n_horizontal, n_depth=args.n_depth)
    mesh = data_frame_to_mesh("tomography", frame, MODEL_COLUMNS[ModelType.EP2020])
    n_tets = mesh.sizes["j"]
    print(f"mesh: {n_tets:,} tetrahedra, {mesh.sizes['i']:,} vertices")
    mesh.to_zarr(mesh_path, mode="w", encoding=DEFAULT_ENCODING_SETTINGS)
    del mesh, frame

    # Points spread through the model's bounding box.
    rng = np.random.default_rng(0)
    built = timed("load zarr + build BVH", lambda: MeshModel._build(mesh_path))
    lo, hi = built.aabb
    xs, ys, zs = (
        rng.uniform(lo[i], hi[i], args.points).astype(np.float32) for i in range(3)
    )
    del built

    # This load runs before the index exists, so it reads the zarr and builds
    # the tree.
    tree_built = timed(
        "ModelTree.load_models, no index", lambda: ModelTree.load_models([mesh_path])
    )

    # `compile_index` builds the mesh again and writes it, so the write alone
    # is this figure less the build figure.
    timed("compile index (build + write)", lambda: MeshModel.compile_index(mesh_path))
    size = index_path(mesh_path).stat().st_size
    print(f"{'index size':48s} {size * MB:10.1f} MB  ({size / n_tets:.0f} B/tet)")

    # A current index now exists beside the mesh, so this load maps it. The
    # figure includes the fingerprint walk over the zarr store.
    tree_mapped = timed(
        "ModelTree.load_models, mapped", lambda: ModelTree.load_models([mesh_path])
    )

    def query(tree: ModelTree):
        return tree.query_many(_xr(xs), _xr(ys), _xr(zs))

    a = timed(f"query {args.points:,} points, built", lambda: query(tree_built))
    b1 = timed(
        f"query {args.points:,} points, mapped, first pass", lambda: query(tree_mapped)
    )
    b2 = timed(
        f"query {args.points:,} points, mapped, second pass", lambda: query(tree_mapped)
    )
    same = all(
        np.array_equal(_values(a, k), _values(b1, k)) for k in ("vs", "vp", "rho")
    )
    print(f"{'built == mapped':48s} {same}")
    same2 = all(np.array_equal(_values(b1, k), _values(b2, k)) for k in ("vs",))
    print(f"{'mapped first == second pass':48s} {same2}")

    if not args.keep:
        shutil.rmtree(root)


def _xr(values: np.ndarray):
    """One column of points on the `(i, j, k)` index `query_many` answers on."""
    import xarray as xr

    return xr.DataArray(values.reshape(-1, 1, 1), dims=["i", "j", "k"])


def _values(qualities, key: str) -> np.ndarray:
    return np.asarray(qualities[key].values)


if __name__ == "__main__":
    main()
