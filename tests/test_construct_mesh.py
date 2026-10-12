"""Tests for the basin meshes ``nzcvm basin main`` writes.

The volumetric tests build on a hand-made triangulation of a square, so
they don't depend on the triangulation; ``surface_samples`` and
``triangulate_polygon`` have tests of their own.
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
from nzcvm.models.surface import LatticeWindow, SurfaceGrid
from nzcvm.scripts import construct_mesh
from nzcvm.scripts.construct_mesh import (
    PAD_DEPTH,
    SAMPLES_PER_CELL,
    Triangulation,
    profile_mesh,
    surface_samples,
    triangulate_polygon,
)

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
    """Triangulate a square with its diagonal from corner 1 to corner 2."""
    vertices = np.array([[0.0, 0.0], [SIDE, 0.0], [0.0, SIDE], [SIDE, SIDE]])
    return Triangulation(vertices, np.array([[0, 1, 2], [1, 3, 2]]))


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
        padded_top=TOPOGRAPHY - PAD_DEPTH if pad_top else None,
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
        padded_top=None,
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


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------


def _grid(f, n: int = 21) -> SurfaceGrid:
    """A surface over a grid slightly larger than the square."""
    axis = np.linspace(-0.1 * SIDE, 1.1 * SIDE, n)
    x, y = np.meshgrid(axis, axis)
    return SurfaceGrid(x, y, f(x, y))


SQUARE = shapely.box(0.0, 0.0, SIDE, SIDE)


def _bump(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 300.0 * np.exp(-((x - 400.0) ** 2 + (y - 600.0) ** 2) / 1e5)


def _interpolate(
    triangulation: Triangulation, grid: SurfaceGrid, points: np.ndarray
) -> np.ndarray:
    """The triangulation's linear interpolant of `grid` at `points`."""
    corners = triangulation.vertices[triangulation.triangles]  # (T, 3, 2)
    corner_values = grid(corners[..., 0].ravel(), corners[..., 1].ravel())[0]
    corner_values = corner_values.reshape(-1, 3)
    # Barycentric coordinates of every point in every triangle.
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    basis = np.stack((b - a, c - a), axis=2)  # (T, 2, 2)
    offset = points[:, None, :] - a[None]  # (N, T, 2)
    lb, lc = np.einsum("tij,ntj->nti", np.linalg.inv(basis), offset).transpose(2, 0, 1)
    weights = np.stack((1 - lb - lc, lb, lc), axis=2)  # (N, T, 3)
    inside = (weights >= -1e-9).all(axis=2)
    triangle = inside.argmax(axis=1)
    assert inside.any(axis=1).all()
    rows = np.arange(len(points))
    return (weights[rows, triangle] * corner_values[triangle]).sum(axis=1)


def test_samples_refine_the_grid_inside_the_outline() -> None:
    grid = _grid(_bump)
    points, spacing = surface_samples(grid, SQUARE, per_cell=2)
    assert shapely.contains_xy(SQUARE, points[:, 0], points[:, 1]).all()
    # The lattice is twice as fine as the 60 m grid, and covers the square.
    assert spacing == pytest.approx(30.0)
    assert np.diff(np.unique(points[:, 0])).max() == pytest.approx(30.0)
    assert len(points) == len(np.unique(points[:, 0])) * len(np.unique(points[:, 1]))


def test_plane_needs_only_the_outline() -> None:
    plane = _grid(lambda x, y: 0.3 * x - 0.2 * y + 7.0)
    triangulation = triangulate_polygon(SQUARE, [(plane, 1.0)])
    assert len(triangulation.triangles) == 2


@pytest.mark.parametrize("tolerance", [1.0, 10.0])
def test_triangulation_fits_samples_within_tolerance(tolerance: float) -> None:
    grid = _grid(_bump)
    triangulation = triangulate_polygon(SQUARE, [(grid, tolerance)])
    points, spacing = surface_samples(grid, SQUARE, SAMPLES_PER_CELL)
    # The outline too, at the lattice spacing.
    outline = shapely.segmentize(SQUARE.exterior, spacing)
    points = np.r_[points, np.asarray(outline.coords)[:-1]]
    values, _ = grid(points[:, 0], points[:, 1])
    error = np.abs(_interpolate(triangulation, grid, points) - values)
    assert error.max() <= tolerance + 1e-6
    # A looser tolerance needs fewer triangles than the samples.
    assert len(triangulation.triangles) < len(points)


def test_max_edge_caps_triangle_size() -> None:
    flat = _grid(lambda x, y: np.zeros_like(x))
    triangulation = triangulate_polygon(SQUARE, [(flat, 1.0)], max_edge=300.0)
    corners = triangulation.vertices[triangulation.triangles]
    edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)
    assert edges.max() <= 300.0 + 1e-6


def _rough_window(seed: int = 0) -> LatticeWindow:
    """A 30 m lattice of rough terrain (depth, +z down) over [0, 3000]^2."""
    rng = np.random.default_rng(seed)
    values = (-200.0 * rng.random((101, 101))).astype(np.float32)
    return LatticeWindow(
        values, np.array([0.0, 0.0]), np.array([30.0, 0.0]), np.array([0.0, 30.0])
    )


def test_terrain_cover_puts_the_top_above_the_terrain() -> None:
    """A top at the cover interpolates at or above every terrain node."""
    window = _rough_window()
    vertices = np.array(
        [[0.0, 0.0], [3000.0, 0.0], [3000.0, 3000.0], [0.0, 3000.0], [1400.0, 1700.0]]
    )
    triangles = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    triangulation = construct_mesh.Triangulation(vertices, triangles)
    cover = construct_mesh.terrain_cover(window, triangulation)

    i, j = np.meshgrid(np.arange(101), np.arange(101), indexing="ij")
    nodes = np.c_[30.0 * i.ravel(), 30.0 * j.ravel()]
    for a, b, c in triangles:
        corners = vertices[[a, b, c]]
        basis = np.linalg.inv(
            np.column_stack([corners[1] - corners[0], corners[2] - corners[0]])
        )
        w1, w2 = basis @ (nodes - corners[0]).T
        inside = (w1 >= -1e-9) & (w2 >= -1e-9) & (w1 + w2 <= 1 + 1e-9)
        top = (1 - w1 - w2) * cover[a] + w1 * cover[b] + w2 * cover[c]
        # Depth is +z down, so "above" is shallower.
        assert (top[inside] <= window.values.ravel()[inside] + 1e-6).all()


def test_terrain_cover_finds_a_spike_between_vertices() -> None:
    window = _rough_window()
    window.values[50, 50] = -5000.0
    vertices = np.array([[0.0, 0.0], [3000.0, 0.0], [3000.0, 3000.0], [0.0, 3000.0]])
    triangulation = construct_mesh.Triangulation(
        vertices, np.array([[0, 1, 2], [0, 2, 3]])
    )
    np.testing.assert_array_equal(
        construct_mesh.terrain_cover(window, triangulation), -5000.0
    )
