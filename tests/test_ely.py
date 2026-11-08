"""Numerical tests for the Ely et al. (2010) GTL taper.

Covers the profile in :mod:`nzcvm.ely_taper` and the branch logic of
:class:`~nzcvm.layers.ely.ElyLayer`. The layer tests run over a stub next
layer with constant tomography and (optionally) basin qualities, and a
constant Vs30 surface written to a temporary Zarr store.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import xarray as xr

from nzcvm.config.layers.core import LayerConfig
from nzcvm.config.layers.ely import ElyLayerConfig
from nzcvm.coordinates import Coordinate
from nzcvm.ely_taper import VP_FROM_VS_RELATION, ely_vs_profile
from nzcvm.grids.grid import Grid
from nzcvm.layers.core import Layer
from nzcvm.layers.ely import ElyLayer
from nzcvm.qualities import Qualities, QualitiesSchema
from nzcvm.query import ModelRange
from tests.conftest import make_grid

DEPTH_T = 350.0
VS30 = 400.0
VP_T = 4000.0
VS_T = 2000.0

#: (rho, vp, vs, qp, qs) of the stub tomography and basin.
TOMOGRAPHY = (2400.0, VP_T, VS_T, 200.0, 100.0)
BASIN = (1800.0, 1600.0, 300.0, 60.0, 30.0)
NAMES = ("rho", "vp", "vs", "qp", "qs")


def _profile(depths: np.ndarray) -> Qualities:
    depth = xr.DataArray(
        np.asarray(depths, dtype=np.float64).reshape(1, 1, -1), dims=("i", "j", "k")
    )
    scalar = xr.DataArray(np.ones((1, 1)), dims=("i", "j"))
    result = ely_vs_profile(
        depth, scalar * VS30, scalar * VP_T, scalar * VS_T, depth_t=DEPTH_T
    )
    return result.isel(i=0, j=0)


# ---------------------------------------------------------------------------
# ely_vs_profile
# ---------------------------------------------------------------------------


def test_profile_at_surface_is_half_the_vs30_relation() -> None:
    surface = _profile(np.array([0.0]))
    vp_vs30 = float(VP_FROM_VS_RELATION(xr.DataArray(VS30)))
    assert float(surface.vs[0]) == pytest.approx(0.5 * VS30, rel=1e-6)
    assert float(surface.vp[0]) == pytest.approx(0.5 * vp_vs30, rel=1e-5)


def test_profile_meets_tomography_at_depth_t() -> None:
    bottom = _profile(np.array([DEPTH_T]))
    assert float(bottom.vs[0]) == pytest.approx(VS_T, rel=1e-6)
    assert float(bottom.vp[0]) == pytest.approx(VP_T, rel=1e-6)


def test_profile_interior_values() -> None:
    depths = np.array([0.0, 30.0, 175.0, 350.0])
    result = _profile(depths)
    # Pinned values from the current code.
    np.testing.assert_allclose(
        result.vs.values, [200.0, 660.222, 1531.8615, 2000.0], rtol=1e-5
    )
    np.testing.assert_allclose(
        result.vp.values, [832.00635, 2150.552, 3492.5498, 4000.0], rtol=1e-5
    )
    np.testing.assert_allclose(
        result.rho.values, [1091.9526, 1969.374, 2317.0644, 2393.344], rtol=1e-5
    )


def test_profile_q_scales_with_vs() -> None:
    result = _profile(np.array([0.0, 30.0, 175.0, 350.0]))
    np.testing.assert_allclose(result.qs.values, 0.05 * result.vs.values, rtol=1e-6)
    np.testing.assert_allclose(result.qp.values, 0.1 * result.vs.values, rtol=1e-6)
    np.testing.assert_array_equal(result.alpha.values, 1.0)


# ---------------------------------------------------------------------------
# ElyLayer
# ---------------------------------------------------------------------------

#: Node depths of the test grid. The last two are at and below DEPTH_T.
Z = np.array([0.0, 30.0, 175.0, DEPTH_T, 450.0])

#: x of a column outside the Vs30 surface, where Vs30 is NaN.
X_OUTSIDE = 5000.0


@dataclass
class _StubConfig(LayerConfig):
    type: Literal["_ely_stub"] = "_ely_stub"


class _Stub(Layer[_StubConfig]):
    """Constant tomography, with a basin in the columns ``i`` in *basin_columns*.

    Records the model range of every call in :attr:`calls`.
    """

    def __init__(self, basin_columns: tuple[int, ...] = ()) -> None:
        super().__init__(_StubConfig(), None, None)  # ty: ignore[invalid-argument-type]
        self.basin_columns = basin_columns
        self.calls: list[ModelRange] = []

    def __call__(
        self, grid: Grid, model_range: ModelRange = ModelRange.ALL
    ) -> Qualities:
        self.calls.append(model_range)
        like = xr.zeros_like(grid.z, dtype=np.float32)
        basin = like.copy(data=np.zeros(like.shape, dtype=bool))
        for i in self.basin_columns:
            basin[{Coordinate.I.value: i}] = True
        values: dict[str, xr.DataArray] = {}
        for name, tomo, bas in zip(NAMES, TOMOGRAPHY, BASIN, strict=True):
            if model_range == ModelRange.TOMOGRAPHY:
                values[name] = like + tomo
            elif model_range == ModelRange.BASINS:
                values[name] = like + xr.where(basin, bas, 0.0)
            else:
                values[name] = like + xr.where(basin, bas, tomo)
        if model_range == ModelRange.BASINS:
            alpha = like + xr.where(basin, 1.0, 0.0)
        else:
            alpha = like + 1.0
        return QualitiesSchema.new(**values, alpha=alpha)


@pytest.fixture()
def vs30_path(tmp_path: Path) -> Path:
    """A constant Vs30 surface over -1000 <= x, y <= 1000 (identity frame)."""
    nodes = np.array([-1000.0, 0.0, 1000.0])
    surface = xr.Dataset(
        {"z": (("y", "x"), np.full((3, 3), VS30, dtype=np.float32))},
        coords={"x": nodes, "y": nodes},
        attrs={"frame": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]},
    )
    path = tmp_path / "vs30.zarr"
    surface.to_zarr(path, mode="w")
    return path


def _grid(
    coastline: list[float],
    x: list[float] | None = None,
    z: np.ndarray = Z,
) -> Grid:
    """Columns along ``i`` at depths *z*, with per-column coastline and x."""
    n = len(coastline)
    grid = make_grid(nx=n, ny=1, nz=len(z))
    shape = (n, 1, len(z))
    depth = np.broadcast_to(z.astype(np.float32), shape).copy()
    grid["depth"] = grid.depth.copy(data=depth)
    grid["z"] = grid.z.copy(data=-depth)
    xs = np.zeros(n) if x is None else np.asarray(x)
    grid["x"] = grid.x.copy(
        data=np.broadcast_to(xs.astype(np.float32)[:, None, None], shape).copy()
    )
    grid[Coordinate.COASTLINE] = (
        (Coordinate.I, Coordinate.J),
        np.asarray(coastline, dtype=np.float32).reshape(n, 1),
    )
    return grid.assign_coords(k=np.arange(len(z)))


def _layer(vs30_path: Path, stub: _Stub) -> ElyLayer:
    return ElyLayer(ElyLayerConfig(vs30=vs30_path, depth_t=DEPTH_T), None, stub)


def _column(result: Qualities, i: int, name: str) -> np.ndarray:
    return result[name].isel(i=i, j=0).values


def test_layer_tapers_onshore_column_above_depth_t(vs30_path: Path) -> None:
    result = _layer(vs30_path, _Stub())(_grid(coastline=[0.0]))
    expected = _profile(Z[:3])
    for name in NAMES:
        np.testing.assert_allclose(
            _column(result, 0, name)[:3], expected[name].values, rtol=1e-5
        )
    # At and below depth_t the next layer's values stand.
    for name, value in zip(NAMES, TOMOGRAPHY, strict=True):
        np.testing.assert_allclose(_column(result, 0, name)[3:], value, rtol=1e-6)


def test_layer_leaves_offshore_column_untouched(vs30_path: Path) -> None:
    result = _layer(vs30_path, _Stub())(_grid(coastline=[0.0, 5000.0]))
    assert _column(result, 0, "vs")[0] == pytest.approx(0.5 * VS30, rel=1e-5)
    for name, value in zip(NAMES, TOMOGRAPHY, strict=True):
        np.testing.assert_allclose(_column(result, 1, name), value, rtol=1e-6)


def test_layer_leaves_nan_vs30_column_untouched(vs30_path: Path) -> None:
    grid = _grid(coastline=[0.0, 0.0], x=[0.0, X_OUTSIDE])
    result = _layer(vs30_path, _Stub())(grid)
    assert _column(result, 0, "vs")[0] == pytest.approx(0.5 * VS30, rel=1e-5)
    for name, value in zip(NAMES, TOMOGRAPHY, strict=True):
        column = _column(result, 1, name)
        assert not np.isnan(column).any(), name
        np.testing.assert_allclose(column, value, rtol=1e-6)


def test_layer_skips_chunk_below_depth_t(vs30_path: Path) -> None:
    stub = _Stub()
    grid = _grid(coastline=[0.0], z=np.array([DEPTH_T, 450.0]))
    result = _layer(vs30_path, stub)(grid)
    assert stub.calls == [ModelRange.ALL]
    for name, value in zip(NAMES, TOMOGRAPHY, strict=True):
        np.testing.assert_allclose(_column(result, 0, name), value, rtol=1e-6)


def test_layer_skips_offshore_chunk(vs30_path: Path) -> None:
    stub = _Stub()
    result = _layer(vs30_path, stub)(_grid(coastline=[5000.0]))
    assert stub.calls == [ModelRange.ALL]
    np.testing.assert_allclose(_column(result, 0, "vs"), VS_T, rtol=1e-6)


def test_layer_basins_range_bypasses_taper(vs30_path: Path) -> None:
    stub = _Stub(basin_columns=(0,))
    result = _layer(vs30_path, stub)(_grid(coastline=[0.0]), ModelRange.BASINS)
    assert stub.calls == [ModelRange.BASINS]
    np.testing.assert_allclose(_column(result, 0, "vs"), BASIN[2], rtol=1e-6)
    np.testing.assert_allclose(_column(result, 0, "alpha"), 1.0)


def test_layer_skips_chunk_all_in_basin(vs30_path: Path) -> None:
    stub = _Stub(basin_columns=(0, 1))
    result = _layer(vs30_path, stub)(_grid(coastline=[0.0, 0.0]))
    assert stub.calls[-1] == ModelRange.ALL
    for i in range(2):
        for name, value in zip(NAMES, BASIN, strict=True):
            np.testing.assert_allclose(_column(result, i, name), value, rtol=1e-6)


def test_layer_blends_basins_and_tapers_the_rest(vs30_path: Path) -> None:
    result = _layer(vs30_path, _Stub(basin_columns=(0,)))(_grid(coastline=[0.0, 0.0]))
    # The basin column keeps the basin, unaffected by the taper.
    for name, value in zip(NAMES, BASIN, strict=True):
        np.testing.assert_allclose(_column(result, 0, name), value, rtol=1e-6)
    # The taper acts on the basin-free column above depth_t.
    expected = _profile(Z[:3])
    for name in NAMES:
        np.testing.assert_allclose(
            _column(result, 1, name)[:3], expected[name].values, rtol=1e-5
        )
    np.testing.assert_allclose(_column(result, 1, "vs")[3:], VS_T, rtol=1e-6)
