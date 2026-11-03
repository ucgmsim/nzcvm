"""Numerical tests for :class:`~nzcvm.layers.backus.BackusAveragedLayer`.

A stub next layer returns qualities that depend on the ``z`` that
the layer passes it, so each test can work out the Backus average by hand
and compare.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest
import xarray as xr

from nzcvm.config.layers.backus import BackusAveragedLayerConfig
from nzcvm.config.layers.core import LayerConfig
from nzcvm.grids.grid import Grid
from nzcvm.layers.backus import BackusAveragedLayer
from nzcvm.layers.core import Layer
from nzcvm.qualities import Qualities, QualitiesSchema
from nzcvm.query import ModelRange
from tests.conftest import make_grid

#: Node z values of the one-column grid the tests use (+z down).
Z = np.array([0.0, 100.0, 200.0, 300.0])

#: (rho, vp, vs, qp, qs) of two media.
UPPER = (1800.0, 2000.0, 800.0, 80.0, 40.0)
LOWER = (2600.0, 5000.0, 2900.0, 300.0, 150.0)

Medium = tuple[float, float, float, float, float]


@dataclass
class _StubConfig(LayerConfig):
    type: Literal["_backus_stub"] = "_backus_stub"


class _Stub(Layer[_StubConfig]):
    """A next layer whose qualities are a function of the ``z`` it receives.

    It records every ``z`` array passed to it in :attr:`calls`.
    """

    def __init__(self, medium: Callable[[np.ndarray], tuple[np.ndarray, ...]]):
        super().__init__(_StubConfig(), None, None)  # ty: ignore[invalid-argument-type]
        self.medium = medium
        self.calls: list[np.ndarray] = []

    def __call__(
        self, grid: Grid, model_range: ModelRange = ModelRange.ALL
    ) -> Qualities:
        z = grid.z.values.astype(np.float64)
        self.calls.append(z)
        rho, vp, vs, qp, qs, alpha = self.medium(z)
        like = xr.zeros_like(grid.z)
        return QualitiesSchema.new(
            rho=like + rho,
            vp=like + vp,
            vs=like + vs,
            qp=like + qp,
            qs=like + qs,
            alpha=like + alpha,
        )


def _uniform(medium: Medium, alpha: float = 1.0):
    def f(z: np.ndarray) -> tuple[np.ndarray, ...]:
        return tuple(np.full_like(z, value) for value in (*medium, alpha))

    return f


def _step(at: float, alpha_upper: float = 1.0):
    """UPPER above *at*, LOWER at and below it."""

    def f(z: np.ndarray) -> tuple[np.ndarray, ...]:
        below = z >= at
        values = [np.where(below, lo, up) for up, lo in zip(UPPER, LOWER, strict=True)]
        return (*values, np.where(below, 1.0, alpha_upper))

    return f


def _column(z: np.ndarray = Z, k0: int = 0) -> Grid:
    """A 1×1 column at nodes *z*, with depth equal to z and ``k`` from *k0*."""
    grid = make_grid(nx=1, ny=1, nz=len(z))
    grid["z"] = grid.z.copy(data=z.astype(np.float32).reshape(1, 1, -1))
    grid["depth"] = grid.depth.copy(data=z.astype(np.float32).reshape(1, 1, -1))
    return grid.assign_coords(k=np.arange(k0, k0 + len(z)))


def _backus(media: list[Medium]) -> dict[str, float]:
    """The closed-form Backus average of equally weighted *media*."""
    rho, vp, vs, qp, qs = (np.array(column) for column in zip(*media, strict=True))
    return {
        "rho": rho.mean(),
        "vp": 1.0 / np.sqrt(np.mean(1.0 / (rho * vp**2)) * rho.mean()),
        "vs": 1.0 / np.sqrt(np.mean(1.0 / (rho * vs**2)) * rho.mean()),
        "qp": np.mean(1.0 / vp) / np.mean(1.0 / (vp * qp)),
        "qs": np.mean(1.0 / vs) / np.mean(1.0 / (vs * qs)),
    }


def _layer(stub: _Stub, samples: int) -> BackusAveragedLayer:
    return BackusAveragedLayer(
        BackusAveragedLayerConfig(samples=samples),
        None,
        stub,
    )


def _node(result: Qualities, k: int) -> dict[str, float]:
    return {
        name: float(result[name].isel(i=0, j=0, k=k))
        for name in ("rho", "vp", "vs", "qp", "qs", "alpha")
    }


def test_uniform_medium_is_unchanged() -> None:
    result = _layer(_Stub(_uniform(UPPER)), samples=5)(_column())
    names = ("rho", "vp", "vs", "qp", "qs")
    for k in range(len(Z)):
        node = _node(result, k)
        assert node["alpha"] == pytest.approx(1.0)
        for name, value in zip(names, UPPER, strict=True):
            assert node[name] == pytest.approx(value, rel=1e-5)


def test_two_layer_step_matches_closed_form() -> None:
    # Node 1 spans z = 50..150, so a step at 100 puts half its samples in
    # each medium.
    result = _layer(_Stub(_step(100.0)), samples=4)(_column())
    node = _node(result, 1)
    expected = _backus([UPPER, LOWER, UPPER, LOWER])
    assert node["alpha"] == pytest.approx(1.0)
    for name, value in expected.items():
        assert node[name] == pytest.approx(value, rel=1e-5), name
    # Nodes wholly on one side see one medium.
    assert _node(result, 0)["vp"] == pytest.approx(UPPER[1], rel=1e-5)
    assert _node(result, 2)["vp"] == pytest.approx(LOWER[1], rel=1e-5)


def test_samples_truncate_at_free_surface_on_top_grid() -> None:
    stub = _Stub(_uniform(UPPER))
    _layer(stub, samples=2)(_column())
    sampled = np.stack([z[0, 0] for z in stub.calls])  # (sample, k)
    # Node 0 spans z[0]..halfway to z[1]; inner nodes span halfway to each
    # neighbour. The last node mirrors its upper half-span below.
    np.testing.assert_allclose(sampled[:, 0], [12.5, 37.5])
    np.testing.assert_allclose(sampled[:, 1], [75.0, 125.0])
    np.testing.assert_allclose(sampled[:, 2], [175.0, 225.0])
    np.testing.assert_allclose(sampled[:, 3], [275.0, 325.0])


def test_samples_mirror_at_interface_below_top_grid() -> None:
    stub = _Stub(_uniform(UPPER))
    _layer(stub, samples=2)(_column(k0=4))
    sampled = np.stack([z[0, 0] for z in stub.calls])
    # Node 0 of a lower grid mirrors its lower half-span above.
    np.testing.assert_allclose(sampled[:, 0], [-25.0, 25.0])
    np.testing.assert_allclose(sampled[:, 3], [275.0, 325.0])


def test_partial_coverage_averages_covered_samples_only() -> None:
    # Node 1's two upper samples have alpha 0, its two lower samples alpha 1.
    result = _layer(_Stub(_step(100.0, alpha_upper=0.0)), samples=4)(_column())
    node = _node(result, 1)
    assert node["alpha"] == pytest.approx(0.5)
    for name, value in zip(("rho", "vp", "vs", "qp", "qs"), LOWER, strict=True):
        assert node[name] == pytest.approx(value, rel=1e-5), name


def test_no_coverage_gives_zero_qualities_without_nan() -> None:
    result = _layer(_Stub(_uniform(UPPER, alpha=0.0)), samples=3)(_column())
    for name in ("rho", "vp", "vs", "qp", "qs", "alpha"):
        values = result[name].values
        assert not np.isnan(values).any(), name
        np.testing.assert_array_equal(values, 0.0)
