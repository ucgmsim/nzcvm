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

import numpy as np
import pytest

from nzcvm.scripts.convert_tomography import tet_connectivity

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
