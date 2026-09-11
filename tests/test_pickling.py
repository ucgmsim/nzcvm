"""Tests for moving Rust-backed objects between processes.

A model tree, a surface and a coastline each wrap an index built in Rust with
no Python representation, so pickle had nothing to work with, and
``--distributed`` could only run thread workers inside one interpreter.

They now pickle as the call that built them. These tests check the three
things that has to get right: the payload is the recipe rather than the data,
rebuilding gives the same answers, and a worker rebuilding after unpickling
reads the file once rather than per task.

The cross-process test uses a spawn context rather than fork. Forking a
process with a live Rayon pool in it invites a deadlock, and spawn is what
Dask does anyway.
"""

from __future__ import annotations

import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from nzcvm import synthetic
from nzcvm.layers.coastline import Coastline
from nzcvm.models.model import ModelTree
from nzcvm.models.reconstruct import Reconstructable
from nzcvm.models.surface import Surface

#: A point inland on the synthetic domain, in NZTM.
PROBE = (1_531_509.0, 5_161_095.0)


@pytest.fixture(scope="module")
def resources(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A synthetic DEM, coastline and tomography model on disk."""
    from pyproj import Transformer

    from nzcvm.models.mesh import StructuredMeshSchema
    from nzcvm.scripts.convert_tomography import (
        MODEL_COLUMNS,
        ModelType,
        data_frame_to_mesh,
    )
    from nzcvm.scripts.synthetic import write_wgs84_polygon

    root = tmp_path_factory.mktemp("pickling")
    samples = 16
    to_nztm = Transformer.from_crs(4326, 2193, always_xy=True)
    lon, lat = synthetic.DOMAIN.sample(samples, samples)
    mesh_lon, mesh_lat = np.meshgrid(lon, lat, indexing="ij")
    x, y = to_nztm.transform(mesh_lon, mesh_lat)

    dem = root / "dem.zarr"
    StructuredMeshSchema.new(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        z=(-synthetic.elevation(mesh_lon, mesh_lat)).astype(np.float32),
        i=np.arange(samples),
        j=np.arange(samples),
        name="dem",
    ).to_zarr(dem, mode="w")

    coastline = root / "coastline.wkb.gz"
    write_wgs84_polygon(coastline, synthetic.land())

    models = root / "models"
    models.mkdir()
    data_frame_to_mesh(
        "tomography",
        synthetic.tomography(n_horizontal=6, n_depth=5),
        MODEL_COLUMNS[ModelType.EP2020],
    ).to_zarr(models / "tomography.zarr", mode="w")

    return {"dem": dem, "coastline": coastline, "models": models}


# ---------------------------------------------------------------------------
# What crosses the wire
# ---------------------------------------------------------------------------


def test_a_model_tree_pickles_as_its_paths(resources: dict[str, Path]) -> None:
    """A tree of meshes reduces to a filename, the whole point of this."""
    paths = sorted(resources["models"].glob("*.zarr"))
    tree = ModelTree.load_models(paths)
    assert len(pickle.dumps(tree)) < 1000


def test_a_surface_pickles_as_its_path(resources: dict[str, Path]) -> None:
    assert len(pickle.dumps(Surface.load(resources["dem"]))) < 1000


def test_a_coastline_pickles_as_its_path(resources: dict[str, Path]) -> None:
    assert len(pickle.dumps(Coastline.load(resources["coastline"]))) < 1000


# ---------------------------------------------------------------------------
# What comes back
# ---------------------------------------------------------------------------


def test_a_model_tree_answers_the_same_after_a_round_trip(
    resources: dict[str, Path],
) -> None:
    tree = ModelTree.load_models(sorted(resources["models"].glob("*.zarr")))
    restored = pickle.loads(pickle.dumps(tree))
    before = tree.query(x=PROBE[0], y=PROBE[1], z=100.0)
    after = restored.query(x=PROBE[0], y=PROBE[1], z=100.0)
    assert before is not None and after is not None
    assert (before.rho, before.vp, before.vs) == (after.rho, after.vp, after.vs)


def test_a_surface_answers_the_same_after_a_round_trip(
    resources: dict[str, Path],
) -> None:
    surface = Surface.load(resources["dem"])
    restored = pickle.loads(pickle.dumps(surface))
    x = np.array([PROBE[0]], np.float32)
    y = np.array([PROBE[1]], np.float32)
    assert np.array_equal(surface.transform(x, y), restored.transform(x, y))


def test_a_coastline_answers_the_same_after_a_round_trip(
    resources: dict[str, Path],
) -> None:
    coastline = Coastline.load(resources["coastline"])
    restored = pickle.loads(pickle.dumps(coastline))
    x = np.array([PROBE[0]], np.float32)
    y = np.array([PROBE[1]], np.float32)
    assert np.array_equal(
        coastline.signed_distance(x, y), restored.signed_distance(x, y)
    )


# ---------------------------------------------------------------------------
# Reading once per worker
# ---------------------------------------------------------------------------


def test_loading_the_same_path_twice_returns_one_object(
    resources: dict[str, Path],
) -> None:
    assert Surface.load(resources["dem"]) is Surface.load(resources["dem"])


def test_unpickling_reuses_the_loaded_object(resources: dict[str, Path]) -> None:
    """Dask hands the same layer to every chunk, which makes this the
    difference between reading a file once and reading it per task."""
    surface = Surface.load(resources["dem"])
    blob = pickle.dumps(surface)
    assert all(pickle.loads(blob) is surface for _ in range(20))


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_an_object_built_without_a_factory_says_so() -> None:
    """Rather than pickling into something that fails later, or silently
    produces an object with a dangling handle."""

    class Bare(Reconstructable):
        pass

    with pytest.raises(TypeError, match="no record of how to rebuild it"):
        pickle.dumps(Bare())


def test_a_misaligned_buffer_is_accepted(resources: dict[str, Path]) -> None:
    """Deserialising an array can leave it contiguous but misaligned, which
    `ascontiguousarray` passes through and the Rust binding rejects."""
    coastline = Coastline.load(resources["coastline"])
    buffer = bytearray(4 * 4 + 2)
    misaligned = np.frombuffer(buffer, dtype=np.float32, count=3, offset=2)
    assert not misaligned.flags["ALIGNED"]

    assert coastline.signed_distance(misaligned, misaligned).shape == (3,)


# ---------------------------------------------------------------------------
# Actually in another process
# ---------------------------------------------------------------------------


def _query_in_worker(tree: ModelTree, point: tuple[float, float, float]):
    """Run in a spawned process: the tree arrives as a recipe and rebuilds."""
    import os

    quality = tree.query(x=point[0], y=point[1], z=point[2])
    assert quality is not None
    return os.getpid(), (quality.rho, quality.vp, quality.vs)


def test_a_model_tree_survives_a_real_process_boundary(
    resources: dict[str, Path],
) -> None:
    tree = ModelTree.load_models(sorted(resources["models"].glob("*.zarr")))
    here = tree.query(x=PROBE[0], y=PROBE[1], z=100.0)
    assert here is not None

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        pid, there = pool.submit(
            _query_in_worker, tree, (PROBE[0], PROBE[1], 100.0)
        ).result(timeout=120)

    import os

    assert pid != os.getpid(), "the work has to happen somewhere else"
    assert (here.rho, here.vp, here.vs) == there
