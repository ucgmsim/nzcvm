"""Tests for :class:`~nzcvm.layers.query.QueryLayer` model discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import shapely

from nzcvm.config.layers.query import QueryLayerConfig
from nzcvm.layers.query import QueryLayer
from nzcvm.models.mesh import mesh_encoding
from tests.test_construct_mesh import SIDE, _mesh, _topography_at

GEOMETRY = shapely.box(-1e7, -1e7, 1e7, 1e7)


def _write_mesh(path: Path) -> None:
    mesh = _mesh()
    mesh.to_zarr(path, encoding=mesh_encoding(mesh))


def test_default_globs_find_zarr_meshes(tmp_path: Path) -> None:
    """The default ``model_globs`` loads the zarr meshes the package writes."""
    _write_mesh(tmp_path / "basin.zarr")
    layer = QueryLayer(QueryLayerConfig(model_path=tmp_path), GEOMETRY, None)  # ty: ignore[invalid-argument-type]

    x = y = 0.25 * SIDE
    z = _topography_at(x, y) + 200.0
    xs, ys, zs, depths = (np.array([v], dtype=np.float32) for v in (x, y, z, 200.0))
    out = layer.model._query_many_raw(xs, ys, zs, depths)
    assert out[0, 2] == 800.0


def test_no_matching_models_is_an_error(tmp_path: Path) -> None:
    """A query layer with no matching models raises instead of returning zeros."""
    _write_mesh(tmp_path / "basin.zarr")
    config = QueryLayerConfig(model_path=tmp_path, model_globs=["*.vtkhdf"])
    with pytest.raises(ValueError, match=r"\*\.vtkhdf") as excinfo:
        QueryLayer(config, GEOMETRY, None)  # ty: ignore[invalid-argument-type]
    assert str(tmp_path) in str(excinfo.value)
