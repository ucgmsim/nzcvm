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

from nzcvm.models.mesh import StructuredMeshSchema
from nzcvm.models.surface import Surface

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _flat_surface(
    z: float = 5.0, side: float = 10.0, cx: float = 5.0, cy: float = 5.0
) -> Surface:
    """Flat plane at constant elevation *z* covering [cx-side/2, cx+side/2]²."""
    n = 5  # 5x5 grid, matching pv.Plane(i_resolution=4, j_resolution=4)
    xs = np.linspace(cx - side / 2, cx + side / 2, n, dtype=np.float32)
    ys = np.linspace(cy - side / 2, cy + side / 2, n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = np.full_like(xx, z)
    mesh = StructuredMeshSchema.new(
        x=xx, y=yy, z=zz, i=np.arange(n), j=np.arange(n), name="flat"
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


def _sloped_surface() -> Surface:
    """5x5 grid over [0, 4]² with z = x + 10 y, so every vertex is distinct."""
    xs = np.arange(5, dtype=np.float32)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    mesh = StructuredMeshSchema.new(
        x=xx, y=yy, z=xx + 10 * yy, i=np.arange(5), j=np.arange(5), name="sloped"
    )
    return Surface.from_dataset(mesh)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [(10.0, 2.0, 24.0), (2.0, -7.0, 2.0), (-3.0, 9.0, 40.0), (1e6, 1e6, 44.0)],
    ids=["east", "south", "north-west", "far"],
)
def test_transform_extrapolates_nearest_boundary_vertex(
    x: float, y: float, expected: float
) -> None:
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
