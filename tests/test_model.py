"""Tests for the ModelTree / MeshModel FFI boundary.

Cargo tests cover the Rust BVH tree and blending logic.  These
tests focus on the Python-level contracts:

* :class:`~nzcvm.models.model.ModelRange` enumeration values.
* :meth:`~nzcvm.models.model.ModelTree.query_many_raw` shape / dtype contract.
* :meth:`~nzcvm.models.model.ModelTree.query_many` xarray Dataset contract.
* :class:`~nzcvm.models.model.MeshModel` metadata accessors.
* Priority-range filtering observable from the outside.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]
from nzcvm.models.model import MeshModel, ModelRange, ModelTree
from tests.conftest import _mesh_model

# ---------------------------------------------------------------------------
# query_many_raw shape and dtype contract
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=1, max_value=16),
)
def test_query_many_raw_1d_shape(unit_tetrahedron_tree, n: int) -> None:
    model = ModelTree(unit_tetrahedron_tree)
    x = np.full(n, 0.1, dtype=np.float32)
    z = np.zeros(n, dtype=np.float32)
    result = model._query_many_raw(x, z, z)
    assert result.shape == (n, 6)
    assert result.dtype == np.float32


@given(
    nx=st.integers(min_value=1, max_value=4),
    ny=st.integers(min_value=1, max_value=4),
)
def test_query_many_raw_nd_shape(unit_tetrahedron_tree, nx: int, ny: int) -> None:
    model = ModelTree(unit_tetrahedron_tree)
    x = np.full((nx, ny), 0.1, dtype=np.float32)
    result = model._query_many_raw(x, x, x)
    assert result.shape == (nx, ny, 6)


# ---------------------------------------------------------------------------
# MeshModel metadata
# ---------------------------------------------------------------------------


def test_mesh_model_name_round_trips() -> None:
    raw = _mesh_model(name="my_mesh")
    m = MeshModel(raw)
    assert m.name == "my_mesh"


def test_mesh_model_default_name_is_empty() -> None:
    raw = _mesh_model()
    m = MeshModel(raw)
    assert m.name == ""


def test_mesh_model_priority_accessible() -> None:
    raw = _mesh_model(priority=42)
    m = MeshModel(raw)
    assert m.priority == 42


def test_mesh_model_aabb_shape(unit_tetrahedron_mesh) -> None:
    mn, mx = MeshModel(unit_tetrahedron_mesh).aabb
    assert mn.shape == (3,) and mx.shape == (3,)


def test_mesh_model_aabb_min_lt_max(unit_tetrahedron_mesh) -> None:
    mn, mx = MeshModel(unit_tetrahedron_mesh).aabb
    assert all(mn[i] <= mx[i] for i in range(3))


def test_mesh_model_view_label_contains_name() -> None:
    m = MeshModel(_mesh_model(name="crust"))
    view = m.view()
    assert "crust" in str(view.label)


# ---------------------------------------------------------------------------
# Query wrappers: MeshModel and ModelTree share a point-query contract
# ---------------------------------------------------------------------------


def _as_mesh_model(raw) -> MeshModel:
    return MeshModel(raw)


def _as_model_tree(raw) -> ModelTree:
    return ModelTree(_nzcvm.model_tree([raw]))


_WRAPPERS = pytest.mark.parametrize(
    "wrap", [_as_mesh_model, _as_model_tree], ids=["MeshModel", "ModelTree"]
)


@_WRAPPERS
def test_query_inside_returns_quality(wrap) -> None:
    model = wrap(_mesh_model(rho=1234.0))
    q = model.query(0.1, 0.1, 0.1)
    assert q is not None
    assert q.rho == pytest.approx(1234.0)


@_WRAPPERS
def test_query_outside_returns_none(wrap) -> None:
    """A point beyond the mesh AABB must miss for both wrapper types.

    The coordinate comes from the AABB rather than a hardcoded literal, so it
    remains outside even if the shared tetrahedron fixture changes.
    """
    _, aabb_max = MeshModel(_mesh_model()).aabb
    outside = tuple(float(c) + 1.0 for c in aabb_max)

    model = wrap(_mesh_model())
    assert model.query(*outside) is None


def test_model_range_basins_includes_priority_zero() -> None:
    """A priority-0 model should be visible in the BASINS range."""
    vs = 3500.0
    model = ModelTree(_nzcvm.model_tree([_mesh_model(vs=vs, priority=0)]))
    x = 0.1
    z = 0.4
    result = model.query(x, z, z, model_range=ModelRange.BASINS)
    assert result is not None
    assert result.vs == pytest.approx(vs)


def test_model_range_tomography_excludes_priority_zero() -> None:
    """A priority-0 (basin) model must NOT be visible in TOMOGRAPHY range."""
    model = ModelTree(_nzcvm.model_tree([_mesh_model(vs=3500.0, priority=0)]))
    x = 0.1
    z = 1.0
    result = model.query(x, z, z, model_range=ModelRange.TOMOGRAPHY)
    assert result is None


@given(
    x=st.floats(0.0, 1.0, width=32),
    y=st.floats(0.0, 1.0, width=32),
    z=st.floats(0.0, 1.0, width=32),
)
def test_query_matches_query_many(x: float, y: float, z: float) -> None:
    """Asserts that query and query many agree on model outputs."""
    model = ModelTree(_nzcvm.model_tree([_mesh_model(vs=3500.0, priority=0)]))
    x_arr = xr.DataArray(
        np.atleast_3d(x), dims=["i", "j", "k"], coords={"i": [0], "j": [0], "k": [0]}
    )
    y_arr = xr.DataArray(
        np.atleast_3d(y), dims=["i", "j", "k"], coords={"i": [0], "j": [0], "k": [0]}
    )
    z_arr = xr.DataArray(
        np.atleast_3d(z), dims=["i", "j", "k"], coords={"i": [0], "j": [0], "k": [0]}
    )
    result_query = model.query(x, y, z)
    result_query_many = model.query_many(x_arr, y_arr, z_arr)
    assert (result_query is None) == (result_query_many.alpha.item() == 0.0)
    if result_query:
        assert result_query.vs == pytest.approx(result_query_many.vs.item())
        assert result_query.vp == pytest.approx(result_query_many.vp.item())
        assert result_query.rho == pytest.approx(result_query_many.rho.item())
        assert result_query.qp == pytest.approx(result_query_many.qp.item())
        assert result_query.qs == pytest.approx(result_query_many.qs.item())
        assert result_query.alpha == pytest.approx(result_query_many.alpha.item())
