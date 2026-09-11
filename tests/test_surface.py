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
# Pickling
#
# A surface built in memory replays `from_dataset`; one read off disk replays
# `load`. `tests/test_pickling.py` covers the path-based form and the process
# boundary.
# ---------------------------------------------------------------------------


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
