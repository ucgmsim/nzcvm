"""Tests for the depth-profile basin meshes ``nzcvm basin main`` writes.

The gmsh triangulation is too slow for a unit test, so these build the
volumetric mesh from a hand-made triangulation of a square.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st

from nzcvm.models.mesh import (
    DEPTH_PROFILE_VARIABLES,
    ModelType,
    TetrahedralMeshSchema,
    mesh_encoding,
)
from nzcvm.models.model import MeshModel
from nzcvm.scripts import construct_mesh
from nzcvm.scripts.construct_mesh import PAD_DEPTH, Triangulation, profile_mesh

SIDE = 1000.0
# Elevation of the four corners, as depth (positive down): the topography
# slopes so the profile's reference surface isn't flat.
TOPOGRAPHY = np.array([0.0, -40.0, -20.0, -60.0])
THICKNESS = 500.0

PROFILE = pd.DataFrame(
    {
        "z": [0.0, 100.0, 300.0],
        "thickness": [100.0, 200.0, 1e6],
        "rho": [1800.0, 2000.0, 2200.0],
        "vp": [1500.0, 2000.0, 2500.0],
        "vs": [400.0, 800.0, 1200.0],
        "qp": [40.0, 80.0, 120.0],
        "qs": [20.0, 40.0, 60.0],
    }
)


def _square() -> Triangulation:
    """Triangulate a square, numbering the corners 1-4 (gmsh style)."""
    vertices = np.zeros(
        4,
        dtype=[
            ("vertex", np.int64),
            ("x", np.float64),
            ("y", np.float64),
            ("boundary", np.int64),
        ],
    )
    vertices["vertex"] = np.arange(1, 5)
    vertices["x"] = [0.0, SIDE, 0.0, SIDE]
    vertices["y"] = [0.0, 0.0, SIDE, SIDE]
    triangles = np.zeros(
        2,
        dtype=[("vertex", np.int64), ("i", np.int64), ("j", np.int64), ("k", np.int64)],
    )
    triangles["vertex"] = [1, 2]
    triangles["i"] = [1, 2]
    triangles["j"] = [2, 4]
    triangles["k"] = [3, 3]
    return Triangulation(vertices, triangles)


def _topography(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The planar interpolant of TOPOGRAPHY over the triangulation."""
    u, v = x / SIDE, y / SIDE
    lower = u + v <= 1.0
    t0, t1, t2, t3 = TOPOGRAPHY
    return np.where(
        lower,
        t0 + (t1 - t0) * u + (t2 - t0) * v,
        t3 + (t2 - t3) * (1.0 - u) + (t1 - t3) * (1.0 - v),
    )


def _topography_at(x: float, y: float) -> float:
    return float(_topography(np.array(x), np.array(y)))


def _mesh(
    pad_top: bool = False,
    bottom: np.ndarray | None = None,
    model: pd.DataFrame = PROFILE,
):
    return profile_mesh(
        "square",
        shapely.box(0.0, 0.0, SIDE, SIDE),
        _square(),
        TOPOGRAPHY,
        TOPOGRAPHY.copy(),
        TOPOGRAPHY + THICKNESS if bottom is None else bottom,
        model,
        culling_volume=1e-3,
        pad_top=pad_top,
        priority=3,
    )


def _expected_vs(depth: np.ndarray) -> np.ndarray:
    tops = PROFILE["z"].to_numpy()
    return PROFILE["vs"].to_numpy()[np.searchsorted(tops[1:], depth, side="right")]


def test_one_layer_of_depth_profile_tetrahedra() -> None:
    mesh = _mesh()
    # Each triangle gives three tetrahedra however many profile layers there are.
    assert mesh.sizes["j"] == 6
    assert (mesh.model_type.values == ModelType.DEPTH_PROFILE).all()
    assert (mesh.priority.values == 3).all()
    assert mesh.has_depth_profile


def test_vertex_depth_is_below_topography() -> None:
    mesh = _mesh()
    np.testing.assert_allclose(
        mesh.depth.values, mesh.z.values - _topography(mesh.x.values, mesh.y.values)
    )


# Avoid the layer boundaries, where rounding picks either layer.
interior = st.floats(min_value=0.01 * SIDE, max_value=0.99 * SIDE)
depths = st.floats(min_value=1.0, max_value=THICKNESS - 1.0).filter(
    lambda d: abs(d - 100.0) > 0.5 and abs(d - 300.0) > 0.5
)


@given(x=interior, y=interior, depth=depths)
def test_query_evaluates_profile_at_depth(x: float, y: float, depth: float) -> None:
    model = MeshModel.from_mesh(_mesh())
    z = _topography_at(x, y) + depth
    quality = model.query(x, y, z)
    assert quality is not None
    assert quality.vs == pytest.approx(_expected_vs(np.array(depth)))
    assert quality.alpha == pytest.approx(1.0)


def test_pad_top_extends_first_layer_upwards() -> None:
    model = MeshModel.from_mesh(_mesh(pad_top=True))
    x = y = 0.3 * SIDE
    above = _topography_at(x, y) - 0.5 * PAD_DEPTH
    quality = model.query(x, y, above)
    assert quality is not None
    assert quality.vs == PROFILE["vs"].iloc[0]
    assert MeshModel.from_mesh(_mesh(pad_top=False)).query(x, y, above) is None


def test_zero_thickness_column_is_culled() -> None:
    bottom = TOPOGRAPHY + THICKNESS
    # Corner 4 (index 3) belongs only to the upper triangle.
    bottom[3] = TOPOGRAPHY[3]
    mesh = _mesh(bottom=bottom)
    # Its prism keeps the two tetrahedra with one vertex on the pinched column.
    assert mesh.sizes["j"] == 5
    TetrahedralMeshSchema.from_dataset(mesh)


def test_profile_mesh_round_trips_through_zarr(tmp_path: Path) -> None:
    path = tmp_path / "square.zarr"
    mesh = _mesh()
    mesh.to_zarr(path, encoding=mesh_encoding(mesh))
    reloaded = TetrahedralMeshSchema.from_dataset(xr.open_zarr(path))
    assert reloaded.has_depth_profile
    np.testing.assert_array_equal(reloaded.profile_vs.values, PROFILE["vs"])

    model = MeshModel.from_path(path)
    x = y = 0.25 * SIDE
    z = _topography_at(x, y) + 200.0
    quality = model.query(x, y, z)
    assert quality is not None
    assert quality.vs == 800.0


def test_partial_profile_is_rejected() -> None:
    incomplete = _mesh().drop_vars("profile_vs")
    with pytest.raises(TypeError):
        TetrahedralMeshSchema.from_dataset(incomplete)


def test_depth_profile_cells_need_a_profile() -> None:
    mesh = _mesh().drop_vars([*DEPTH_PROFILE_VARIABLES, "layer"])
    with pytest.raises(ValueError, match="has no profile"):
        MeshModel.from_mesh(TetrahedralMeshSchema.from_dataset(mesh))


def test_uniform_model_is_a_one_layer_profile() -> None:
    model = construct_mesh.uniform_model(2000.0, 2500.0, 1000.0, 100.0, 50.0)
    quality = MeshModel.from_mesh(_mesh(model=model)).query(100.0, 100.0, 300.0)
    assert quality is not None
    assert quality.vs == 1000.0


def test_layered_mesh_round_trips_through_zarr(tmp_path: Path) -> None:
    """``--layered`` meshes have no depth profile and still write and query."""
    layers = construct_mesh.slice_with_model(
        _square(),
        TOPOGRAPHY,
        TOPOGRAPHY.copy(),
        TOPOGRAPHY + THICKNESS,
        PROFILE,
        culling_volume=1e-3,
        pad_top=False,
    )
    mesh = construct_mesh.construct_volumetric_mesh(
        "layered", shapely.box(0.0, 0.0, SIDE, SIDE), layers, priority=3
    )
    # One slice of three tetrahedra per triangle per profile layer.
    assert mesh.sizes["j"] == 2 * 3 * len(PROFILE)
    path = tmp_path / "layered.zarr"
    mesh.to_zarr(path, encoding=mesh_encoding(mesh))
    reloaded = TetrahedralMeshSchema.from_dataset(xr.open_zarr(path))
    assert not reloaded.has_depth_profile

    x = y = 0.25 * SIDE
    z = _topography_at(x, y) + 200.0
    quality = MeshModel.from_path(path).query(x, y, z)
    assert quality is not None
    assert quality.vs == pytest.approx(800.0)


def test_basin_stops_at_the_bottom_of_the_model() -> None:
    shallow = PROFILE.assign(thickness=[100.0, 200.0, 50.0])
    mesh = _mesh(model=shallow)
    # The last layer ends 350 m below the topography, above the 500 m basement.
    np.testing.assert_allclose(mesh.depth.values.max(), 350.0, atol=1e-3)


def test_zero_thickness_layers_are_dropped() -> None:
    repeated = pd.concat([PROFILE.iloc[:2], PROFILE.iloc[1:]], ignore_index=True)
    mesh = _mesh(model=repeated)
    np.testing.assert_array_equal(mesh.profile_top.values, PROFILE["z"])


def test_decreasing_layer_tops_are_rejected() -> None:
    with pytest.raises(ValueError, match="increase"):
        _mesh(model=PROFILE.assign(z=[0.0, 300.0, 100.0]))
