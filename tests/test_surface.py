"""Tests for the Surface FFI boundary class.

Cargo tests cover the mathematical correctness of the underlying Rust
interpolator.  These tests pin the Python-level contract:

* :meth:`~nzcvm.models.surface.Surface.from_dataset` produces a
  :class:`~nzcvm.models.surface.Surface` with sensible metadata.
* :meth:`~nzcvm.models.surface.Surface.transform` preserves the input shape
  and returns float32 values.
* The ``bounds`` array reads ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nzcvm.models.mesh import regular_mesh
from nzcvm.models.surface import Surface

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _flat_surface(
    z: float = 5.0, side: float = 10.0, cx: float = 5.0, cy: float = 5.0
) -> Surface:
    """Flat plane at constant elevation *z* covering [cx-side/2, cx+side/2]²."""
    n = 5  # 5x5 grid, matching pv.Plane(i_resolution=4, j_resolution=4)
    step = side / (n - 1)
    mesh = regular_mesh(
        "flat",
        np.full((n, n), z, dtype=np.float32),
        origin=(cx - side / 2, cy - side / 2),
        step_i=(step, 0.0),
        step_j=(0.0, step),
    )
    return Surface.from_dataset(mesh)


@pytest.fixture()
def flat_surface() -> Surface:
    return _flat_surface()


# ---------------------------------------------------------------------------
# Metadata contracts
# ---------------------------------------------------------------------------


def test_n_points_positive(flat_surface: Surface) -> None:
    assert flat_surface.n_points > 0


def test_bounds_length(flat_surface: Surface) -> None:
    assert len(flat_surface.bounds) == 6


def test_bounds_order_min_lt_max(flat_surface: Surface) -> None:
    """bounds = [xmin, ymin, zmin, xmax, ymax, zmax], with mins < maxes."""
    b = flat_surface.bounds
    assert b[0] < b[3]
    assert b[1] < b[4]


def test_bounds_z_constant_for_flat_surface() -> None:
    s = _flat_surface(z=7.0)
    assert s.bounds[2] == pytest.approx(7.0, abs=1e-6)
    assert s.bounds[5] == pytest.approx(7.0, abs=1e-6)
    assert s.bounds[2] == s.bounds[5]


# ---------------------------------------------------------------------------
# Transform output shape and dtype
# ---------------------------------------------------------------------------


@given(
    nx=st.integers(min_value=1, max_value=8),
    ny=st.integers(min_value=1, max_value=8),
)
def test_transform_preserves_shape(nx: int, ny: int) -> None:
    s = _flat_surface()
    x = np.full((nx, ny), 5.0, dtype=np.float32)
    y = np.full((nx, ny), 5.0, dtype=np.float32)
    z = s.transform(x, y)
    assert z.shape == (nx, ny)


def test_transform_returns_float32(flat_surface: Surface) -> None:
    x = np.array([[5.0, 5.0]], dtype=np.float32)
    y = np.array([[5.0, 5.0]], dtype=np.float32)
    assert flat_surface.transform(x, y).dtype == np.float32


def test_transform_1d_input(flat_surface: Surface) -> None:
    x = np.array([5.0, 5.0], dtype=np.float32)
    y = np.array([5.0, 5.0], dtype=np.float32)
    result = flat_surface.transform(x, y)
    assert result.shape == (2,)


# ---------------------------------------------------------------------------
# Pickling (registry round-trip)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "y"),
    [(20.0, 5.0), (5.0, -20.0), (-1e6, 1e6)],
    ids=["east", "south", "far"],
)
def test_transform_is_nan_outside_the_surface(
    flat_surface: Surface, x: float, y: float
) -> None:
    """Points outside the surface come back NaN, not 0.

    The Ely layer skips NaN Vs30. With 0 it tapered a coastal station just
    outside the Vs30 surface (ECLS) to Vs = 0 at the free surface.
    """
    z = flat_surface.transform(np.array([x], np.float32), np.array([y], np.float32))
    assert np.isnan(z).all()


def test_transform_keeps_hits_beside_misses(flat_surface: Surface) -> None:
    x = np.array([5.0, 20.0, 2.5, -3.0], dtype=np.float32)
    y = np.array([5.0, 5.0, 7.5, 5.0], dtype=np.float32)
    z = flat_surface.transform(x, y)
    np.testing.assert_array_equal(np.isnan(z), [False, True, False, True])
    assert z[[0, 2]] == pytest.approx([5.0, 5.0])


def test_surface_pickleable(flat_surface: Surface) -> None:
    import pickle

    x = np.array([5.0], dtype=np.float32)
    y = np.array([5.0], dtype=np.float32)
    np.testing.assert_allclose(flat_surface.transform(x, y), 5.0, rtol=1e-5)
    restored = pickle.loads(pickle.dumps(flat_surface))
    np.testing.assert_allclose(
        flat_surface.transform(x, y),
        restored.transform(x, y),
        rtol=1e-5,
    )


def _unit_mesh(name: str, values: np.ndarray):
    """An upright mesh with node ``[i, j]`` at ``(i, j)``."""
    return regular_mesh(name, values, (0.0, 0.0), (1.0, 0.0), (0.0, 1.0))


def _sloped_values() -> np.ndarray:
    """``z = x + 10 y`` on a 5x5 unit lattice."""
    xx, yy = np.meshgrid(np.arange(5.0), np.arange(5.0), indexing="ij")
    return (xx + 10 * yy).astype(np.float32)


def _sloped_surface() -> Surface:
    """5x5 grid over [0, 4]² with z = x + 10 y, so every vertex is distinct."""
    return Surface.from_dataset(_unit_mesh("sloped", _sloped_values()))


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (10.0, 2.0, 24.0),
        (10.0, 2.5, 29.0),
        (2.0, -7.0, 2.0),
        (1.5, -7.0, 1.5),
        (-3.0, 9.0, 40.0),
        (1e6, 1e6, 44.0),
    ],
    ids=[
        "east",
        "east-between-nodes",
        "south",
        "south-between-nodes",
        "north-west",
        "far",
    ],
)
def test_transform_extends_the_edge_outward(
    x: float, y: float, expected: float
) -> None:
    """An outside point takes the edge value at its clamped position.

    Between two edge nodes that's the line between them, not the nearer
    node's value.
    """
    surface = _sloped_surface()
    z = surface.transform(
        np.array([x], np.float32), np.array([y], np.float32), extrapolate=True
    )
    assert z == pytest.approx([expected])


def test_extrapolate_leaves_hits_alone() -> None:
    surface = _sloped_surface()
    x = np.array([1.5, 2.5], dtype=np.float32)
    y = np.array([0.5, 3.0], dtype=np.float32)
    np.testing.assert_allclose(
        surface.transform(x, y, extrapolate=True), surface.transform(x, y)
    )


def test_extrapolate_leaves_inner_no_data_alone() -> None:
    """A NaN node inside the surface isn't outside it, so nothing fills it."""
    values = _sloped_values()
    values[2, 2] = np.nan
    surface = Surface.from_dataset(_unit_mesh("holed", values))
    z = surface.transform(np.array([2.0, 0.5]), np.array([2.0, 0.5]), extrapolate=True)
    assert np.isnan(z[0])
    assert z[1] == pytest.approx(5.5)


def test_rotated_lattice_reproduces_a_plane() -> None:
    """A lattice at 45 degrees maps each query into its own frame first."""
    step = 30.0 / np.sqrt(2.0)
    origin = (1.5e6, 5.2e6)
    step_i, step_j = (step, step), (-step, step)
    ii, jj = np.meshgrid(np.arange(6.0), np.arange(5.0), indexing="ij")
    node_x = origin[0] + ii * step_i[0] + jj * step_j[0]
    node_y = origin[1] + ii * step_i[1] + jj * step_j[1]

    def plane(x, y):
        return 0.02 * (x - origin[0]) - 0.01 * (y - origin[1]) + 7.0

    surface = Surface.from_dataset(
        regular_mesh(
            "rotated", plane(node_x, node_y).astype(np.float32), origin, step_i, step_j
        )
    )
    fi = np.array([0.3, 2.5, 4.9, 1.0])
    fj = np.array([0.7, 3.2, 0.1, 4.0])
    x = origin[0] + fi * step_i[0] + fj * step_j[0]
    y = origin[1] + fi * step_i[1] + fj * step_j[1]
    np.testing.assert_allclose(surface.transform(x, y), plane(x, y), rtol=1e-6)
    # One step beyond the last row is outside.
    outside_x = origin[0] + 6.0 * step_i[0]
    outside_y = origin[1] + 6.0 * step_i[1]
    assert np.isnan(surface.transform(np.array([outside_x]), np.array([outside_y])))


def test_loaded_surface_reads_lazily_and_matches_memory(tmp_path) -> None:
    """A surface opened from Zarr reads its chunks on demand, with the same values."""
    import pickle

    import zarr

    rng = np.random.default_rng(0)
    values = rng.uniform(0.0, 100.0, (40, 30)).astype(np.float32)
    mesh = regular_mesh("random", values, (1.0e6, 5.0e6), (0.0, -30.0), (30.0, 0.0))
    path = tmp_path / "random.zarr"
    mesh.chunk({"i": 8, "j": 8}).to_zarr(path)

    loaded = Surface.load(path)
    assert isinstance(loaded.values, zarr.Array)
    in_memory = Surface.from_dataset(mesh)
    x = rng.uniform(1.0e6 - 100.0, 1.0e6 + 1000.0, 500)
    y = rng.uniform(5.0e6 - 1300.0, 5.0e6 + 100.0, 500)
    for extrapolate in (False, True):
        np.testing.assert_array_equal(
            loaded.transform(x, y, extrapolate), in_memory.transform(x, y, extrapolate)
        )
    restored = pickle.loads(pickle.dumps(loaded))
    np.testing.assert_array_equal(restored.transform(x, y), loaded.transform(x, y))


def test_windows_average_aligned_blocks(tmp_path) -> None:
    """Overlapping windows of one surface average the same blocks."""
    from nzcvm.models.surface import read_surface_window

    rng = np.random.default_rng(1)
    values = rng.uniform(0.0, 100.0, (60, 60)).astype(np.float32)
    # Over the synthetic domain's longitude and latitude in NZTM.
    origin = (1_550_000.0, 5_150_000.0)
    path = tmp_path / "rough.zarr"
    regular_mesh("rough", values, origin, (0.0, 100.0), (100.0, 0.0)).to_zarr(path)
    to_lon_lat = __import__("pyproj").Transformer.from_crs(2193, 4326, always_xy=True)

    def bbox(x0, y0, x1, y1):
        (lon0, lon1), (lat0, lat1) = to_lon_lat.transform([x0, x1], [y0, y1])
        return (lat0, lat1, lon0, lon1)

    a = read_surface_window(
        path, bbox(1_551_000.0, 5_151_000.0, 1_554_000.0, 5_154_000.0), 300.0
    )
    b = read_surface_window(
        path, bbox(1_551_700.0, 5_151_400.0, 1_555_000.0, 5_155_000.0), 300.0
    )
    keys_a = {
        (round(x), round(y)): v
        for x, y, v in zip(a.x.ravel(), a.y.ravel(), a.values.ravel())
    }
    shared = [
        (keys_a[(round(x), round(y))], v)
        for x, y, v in zip(b.x.ravel(), b.y.ravel(), b.values.ravel())
        if (round(x), round(y)) in keys_a
    ]
    assert len(shared) > 4
    from_a, from_b = np.array(shared).T
    np.testing.assert_allclose(from_b, from_a)
