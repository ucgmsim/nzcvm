"""Tests for the tomography grid-to-tetrahedra conversion.

`tet_connectivity` splits every voxel of a rectilinear grid into five
tetrahedra. Getting the five right for one voxel is easy. The part that goes
wrong is making neighbouring voxels agree on the diagonal they share, and it
goes wrong silently: the cells end up with cracks between them, and a query
lands in one.

So the tests below check the two properties that catch that: the tetrahedra
fill the volume exactly, and exactly two of them meet at every internal
face.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from nzcvm.coordinates import scale, translate
from nzcvm.models.model import ModelTree
from nzcvm.scripts.convert_tomography import (
    TomographyModel,
    data_frame_to_mesh,
    tet_connectivity,
)

SHAPES = [(2, 2, 2), (4, 3, 5), (5, 5, 5), (3, 2, 7), (6, 4, 3)]


def _grid_points(ni: int, nj: int, nk: int) -> np.ndarray:
    """Unit-spaced grid points, flattened in the order the indices assume."""
    i, j, k = np.indices((ni, nj, nk), dtype=np.float64)
    return np.stack([i.ravel(), j.ravel(), k.ravel()], axis=-1)


def _volumes(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    anchor = points[tetrahedra[:, 0]]
    edges = points[tetrahedra[:, 1:]] - anchor[:, np.newaxis, :]
    return np.abs(np.linalg.det(edges)) / 6.0


@pytest.mark.parametrize("shape", SHAPES)
def test_five_tetrahedra_per_voxel(shape: tuple[int, int, int]) -> None:
    ni, nj, nk = shape
    voxels = (ni - 1) * (nj - 1) * (nk - 1)
    assert tet_connectivity(*shape).shape == (5 * voxels, 4)


@pytest.mark.parametrize("shape", SHAPES)
def test_indices_stay_inside_the_grid(shape: tuple[int, int, int]) -> None:
    connectivity = tet_connectivity(*shape)
    assert connectivity.min() >= 0
    assert connectivity.max() < np.prod(shape)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_tetrahedra_fill_the_volume(shape: tuple[int, int, int]) -> None:
    """A unit-spaced grid has one unit of volume per voxel, and the five
    tetrahedra of a voxel have to account for all of it."""
    ni, nj, nk = shape
    points = _grid_points(*shape)
    volumes = _volumes(points, tet_connectivity(*shape))

    assert (volumes > 0).all(), "a degenerate tetrahedron has no interior"
    expected = float((ni - 1) * (nj - 1) * (nk - 1))
    assert volumes.sum() == pytest.approx(expected)


@pytest.mark.parametrize("shape", SHAPES)
def test_every_internal_face_is_shared_by_two_tetrahedra(
    shape: tuple[int, int, int],
) -> None:
    """The watertightness check.

    On the boundary of the grid a face belongs to one tetrahedron, and inside
    it to exactly two. Three of them means an overlap, while one internal face
    on its own means a crack, which is what an unalternated split produces.
    """
    connectivity = tet_connectivity(*shape)
    faces = np.concatenate(
        [
            connectivity[:, list(face)]
            for face in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        ]
    )
    # A face is the same face whatever order its vertices come in.
    _, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    assert set(np.unique(counts)) <= {1, 2}


def test_a_single_voxel_splits_into_a_known_tiling() -> None:
    """The smallest case, checked against the corner indices by hand."""
    connectivity = tet_connectivity(2, 2, 2)
    assert len(connectivity) == 5
    # The tiling uses every corner of the voxel: four tetrahedra each cut off
    # one corner, and a central one joins them.
    assert set(connectivity.ravel().tolist()) == set(range(8))
    volumes = _volumes(_grid_points(2, 2, 2), connectivity)
    # The corner tetrahedra hold 1/6 each and the central one 1/3.
    assert sorted(volumes) == pytest.approx([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 3])


# `data_frame_to_mesh` puts the CSV rows in grid order, then in Morton order,
# and scales the units. A mistake in any of those steps scrambles the
# velocities without raising an error. A field that's linear in x, y and z
# catches it: interpolating over tetrahedra reproduces a linear field exactly,
# and a scrambled one doesn't come back.

_TOMOGRAPHY = TomographyModel(
    latitude="lat",
    longitude="lon",
    x="x",
    y="y",
    z="z",
    rho="rho",
    vp="vp",
    vs="vs",
    affine_inverse=np.eye(4, dtype=np.float32),
)


def _vs(x, y, z):
    return 1.0 + 0.1 * x + 0.2 * y + 0.3 * z


def _vp(x, y, z):
    return 2.0 + 0.5 * x - 0.1 * y + 0.2 * z


def _rho(x, y, z):
    return 2.5 + 0.01 * x + 0.02 * y - 0.03 * z


def _shuffled_grid(nx: int = 4, ny: int = 3, nz: int = 5) -> pd.DataFrame:
    """A rectilinear grid with linear fields (km/s, g/cm³), rows shuffled."""
    z, y, x = np.meshgrid(
        np.arange(nz, dtype=np.float64),
        np.arange(ny, dtype=np.float64),
        np.arange(nx, dtype=np.float64),
        indexing="ij",
    )
    x, y, z = x.ravel(), y.ravel(), z.ravel()
    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "z": z,
            "vs": _vs(x, y, z),
            "vp": _vp(x, y, z),
            "rho": _rho(x, y, z),
        }
    )
    return df.sample(frac=1, random_state=0).reset_index(drop=True)


def _vertex_coordinates(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(mesh[axis].to_numpy().astype(np.float64) for axis in "xyz")


@pytest.mark.parametrize(
    "point", [(1.3, 0.7, 2.2), (2.9, 1.5, 3.9), (0.1, 1.9, 0.4), (2.5, 0.5, 1.5)]
)
def test_shuffled_rows_reproduce_a_linear_field(
    point: tuple[float, float, float],
) -> None:
    mesh = data_frame_to_mesh("test", _shuffled_grid(), _TOMOGRAPHY)
    quality = ModelTree.from_mesh(mesh).query(*point, 0.0)

    assert quality is not None
    assert quality.vs == pytest.approx(1000 * _vs(*point), rel=1e-5)
    assert quality.vp == pytest.approx(1000 * _vp(*point), rel=1e-5)
    assert quality.rho == pytest.approx(1000 * _rho(*point), rel=1e-5)


def test_every_vertex_keeps_its_own_values() -> None:
    """Each field has to follow its vertex through both permutations."""
    mesh = data_frame_to_mesh("test", _shuffled_grid(), _TOMOGRAPHY)
    x, y, z = _vertex_coordinates(mesh)

    np.testing.assert_allclose(mesh["vs"], 1000 * _vs(x, y, z), rtol=1e-6)
    np.testing.assert_allclose(mesh["vp"], 1000 * _vp(x, y, z), rtol=1e-6)
    np.testing.assert_allclose(mesh["rho"], 1000 * _rho(x, y, z), rtol=1e-6)


def test_missing_q_columns_default_from_vs() -> None:
    mesh = data_frame_to_mesh("test", _shuffled_grid(), _TOMOGRAPHY)
    x, y, z = _vertex_coordinates(mesh)

    expected_qs = 0.05 * 1000 * _vs(x, y, z)
    np.testing.assert_allclose(mesh["qs"], expected_qs, rtol=1e-6)
    np.testing.assert_allclose(mesh["qp"], 2.0 * expected_qs, rtol=1e-6)


def test_q_columns_pass_through_unscaled() -> None:
    df = _shuffled_grid()
    df["qs"] = 10.0 + df["x"] + 2.0 * df["y"] + 3.0 * df["z"]
    df["qp"] = 2.5 * df["qs"] + 1.0
    mesh = data_frame_to_mesh("test", df, _TOMOGRAPHY)
    x, y, z = _vertex_coordinates(mesh)

    expected_qs = 10.0 + x + 2.0 * y + 3.0 * z
    np.testing.assert_allclose(mesh["qs"], expected_qs, rtol=1e-6)
    np.testing.assert_allclose(mesh["qp"], 2.5 * expected_qs + 1.0, rtol=1e-6)


def test_a_missing_point_is_rejected() -> None:
    df = _shuffled_grid().drop(index=7)
    with pytest.raises(ValueError, match="not a curvilinear grid"):
        data_frame_to_mesh("test", df, _TOMOGRAPHY)


def test_aabb_is_in_world_coordinates() -> None:
    """The vertices are km from an origin. The aabb is in world metres."""
    ox, oy = 1_500_000.0, 5_000_000.0
    tomography = dataclasses.replace(
        _TOMOGRAPHY,
        affine_inverse=scale(1 / 1000.0, 1 / 1000.0, 1 / 1000.0)
        @ translate(-ox, -oy, z=0.0),
    )
    mesh = data_frame_to_mesh("test", _shuffled_grid(nx=4, ny=3, nz=5), tomography)

    assert mesh.attrs["aabb"] == pytest.approx(
        [ox, oy, 0.0, ox + 3000.0, oy + 2000.0, 4000.0]
    )
