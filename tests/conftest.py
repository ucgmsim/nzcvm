"""Shared fixtures for the nzcvm test suite."""

import os
from pathlib import Path

import numpy as np
import pytest
import shapely
from hypothesis import HealthCheck, settings

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]
from nzcvm.grids.grid import Grid, GridSchema

# ---------------------------------------------------------------------------
# Hypothesis profiles
#
# ``deadline=None`` matters here: several property tests build a fresh
# xarray Dataset per example, which is slow enough to trip the default
# deadline on a loaded machine.
# ---------------------------------------------------------------------------

settings.register_profile(
    "default",
    deadline=None,
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.register_profile("ci", deadline=None, max_examples=300)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))

# ---------------------------------------------------------------------------
# Rust-level helpers
# ---------------------------------------------------------------------------


def _mesh_model(
    rho: float = 2700.0,
    vp: float = 6000.0,
    vs: float = 3500.0,
    qp: float = 200.0,
    qs: float = 100.0,
    alpha: float = 1.0,
    priority: int = 0,
    name: str | None = None,
):
    """Return a raw PyMeshModel wrapping one unit tetrahedron."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2, 3]], dtype=np.uint64)
    types = np.array([0], dtype=np.uint8)
    idx = np.array([0], dtype=np.uint64)
    qualities = np.array([[rho, vp, vs, qp, qs, alpha]], dtype=np.float32)
    return _nzcvm.mesh_model(
        vertices,
        faces,
        types,
        idx,
        qualities,
        np.uint8(priority),
        None,
        name,
    )


@pytest.fixture()
def unit_tetrahedron_mesh():
    """Raw PyMeshModel for one unit tetrahedron."""
    return _mesh_model()


@pytest.fixture()
def unit_tetrahedron_tree():
    """Raw PyModelTree containing one unit tetrahedron."""
    return _nzcvm.model_tree([_mesh_model()])


# ---------------------------------------------------------------------------
# Grid fixture
# ---------------------------------------------------------------------------


def make_grid(
    nx: int = 2,
    ny: int = 2,
    nz: int = 2,
    x0: float = 0.1,
    depth0: float = 0.0,
) -> Grid:
    """Construct a minimal concrete (non-dask) Grid for layer tests."""
    x = np.full((nx, ny, nz), x0, dtype=np.float32)
    y = np.full((nx, ny, nz), x0, dtype=np.float32)
    z = np.full((nx, ny, nz), x0, dtype=np.float32)
    depth = np.zeros((nx, ny, nz), dtype=np.float32) + depth0

    return GridSchema.new(
        x=x,
        y=y,
        z=z,
        depth=depth,
        name="test",
        resolution=100.0,
        geometry=shapely.box(171.9, -43.6, 172.1, -43.4),
        origin_lon=np.float32(172.0),
        origin_lat=np.float32(-43.5),
        azimuth=np.float32(0.0),
        grid_azimuth=np.float32(0.0),
        bottom_left_lon=np.float32(172.0),
        bottom_left_lat=np.float32(-43.5),
    )


@pytest.fixture()
def unit_grid() -> Grid:
    """2×2×2 concrete Grid with all points inside the unit tetrahedron."""
    return make_grid()


@pytest.fixture()
def isolated_layer_registry():
    """Snapshot and restore ``Layer.registry`` after the test."""
    from nzcvm.layers.core import Layer

    original = Layer.registry.copy()
    yield
    Layer.registry.clear()
    Layer.registry.update(original)


def get_model_directory() -> Path | None:
    model_path = os.getenv("MODEL_PATH")
    return Path(model_path) if model_path else None


@pytest.fixture
def model_directory() -> Path:
    directory = get_model_directory()
    if directory is None:
        pytest.skip("MODEL_PATH is not set")
    return directory


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "model_path" not in metafunc.fixturenames:
        return
    directory = get_model_directory()
    if directory is None:
        metafunc.parametrize(
            "model_path",
            [
                pytest.param(
                    None,
                    marks=[
                        pytest.mark.real_data,
                        pytest.mark.skip(reason="MODEL_PATH is not set"),
                    ],
                )
            ],
        )
        return
    paths = sorted(directory.glob("*.zarr"))
    if not paths:
        raise pytest.UsageError(
            f"MODEL_PATH={directory} contains no *.zarr models; the real-data "
            "suite would otherwise collect zero tests and report green."
        )
    metafunc.parametrize(
        "model_path",
        [pytest.param(p, marks=pytest.mark.real_data) for p in paths],
        ids=[p.stem for p in paths],
    )
