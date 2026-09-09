"""Tests for pipeline composition logic.

:func:`~nzcvm.layers.pipeline.build_pipeline` and
:func:`~nzcvm.layers.pipeline.execute_model_pipeline` are the public-facing
pipeline APIs.  These tests use dummy layers to isolate composition from the
inner workings of the layers themselves.
"""

from __future__ import annotations

import pytest
import shapely

from nzcvm.components import Component
from nzcvm.config.layers.clamp import Bound, ClampLayerConfig
from nzcvm.layers.dummy import ConstantLayer
from nzcvm.layers.pipeline import build_pipeline
from tests.conftest import make_grid

# Layers now carry a spatial domain; these unit tests don't exercise masking,
# so any covering geometry works.
GEOM = shapely.box(171.9, -43.6, 172.1, -43.4)

# ---------------------------------------------------------------------------
# build_pipeline guard-rail
# ---------------------------------------------------------------------------


def test_build_pipeline_empty_list_raises() -> None:
    with pytest.raises(ValueError):
        build_pipeline(GEOM, [])


# ---------------------------------------------------------------------------
# Ordering contract
# ---------------------------------------------------------------------------


def test_build_pipeline_single_config_produces_callable() -> None:
    """A single-element config list must produce a callable layer that raises
    ValueError when the grid is out of bounds (sentinel reached)."""
    cfg = ClampLayerConfig()
    pipeline = build_pipeline(GEOM, [cfg])
    with pytest.raises(ValueError, match="out of bounds"):
        pipeline(make_grid())


# ---------------------------------------------------------------------------
# build_pipeline ordering: first config is outermost
#
# These go through build_pipeline itself rather than hand-assembling a chain,
# so that the ``reversed(configs)`` at ``pipeline.py:52`` runs.
# ---------------------------------------------------------------------------


def _constant_cfg(**kwargs: float):
    """A real registered terminal config, built through the layer registry."""
    return ConstantLayer.config_cls(**kwargs)


def test_build_pipeline_first_config_is_outermost() -> None:
    """The first config in the list must wrap the rest.

    A clamp listed *before* the constant terminal bounds the terminal's
    output. Listed after, it would never see it.
    """
    pipeline = build_pipeline(
        GEOM,
        [
            ClampLayerConfig(clamps={Component.VS: Bound(min=4000.0)}),
            _constant_cfg(vs=3000.0),
        ],
    )
    result = pipeline(make_grid())
    assert float(result.vs.min()) == pytest.approx(4000.0, rel=1e-4)


def test_build_pipeline_order_is_not_symmetric() -> None:
    """Reversing the config list must change the result.

    This is the assertion that catches ``build_pipeline`` dropping its
    ``reversed()``: with the clamp innermost the terminal's constant output
    overwrites it, leaving vs at 3000 rather than lifting it to 4000.
    """
    clamp = ClampLayerConfig(clamps={Component.VS: Bound(min=4000.0)})
    terminal = _constant_cfg(vs=3000.0)

    clamp_outermost = build_pipeline(GEOM, [clamp, terminal])(make_grid())
    clamp_innermost = build_pipeline(GEOM, [terminal, clamp])(make_grid())

    assert float(clamp_outermost.vs.min()) == pytest.approx(4000.0, rel=1e-4)
    assert float(clamp_innermost.vs.min()) == pytest.approx(3000.0, rel=1e-4)
    assert float(clamp_outermost.vs.min()) != float(clamp_innermost.vs.min())


def test_build_pipeline_terminates_in_sentinel() -> None:
    """A chain of non-terminal layers must bottom out in the sentinel.

    Both clamps delegate downstream forever, so without the ``_SentinelLayer``
    the innermost would call ``None``.
    """
    pipeline = build_pipeline(
        GEOM,
        [
            ClampLayerConfig(clamps={Component.VS: Bound(min=4000.0)}),
            ClampLayerConfig(clamps={Component.VP: Bound(max=4500.0)}),
        ],
    )
    with pytest.raises(ValueError, match="out of bounds"):
        pipeline(make_grid())


# Hand-assembled ``Layer`` composition tests live in ``test_layers.py``: they
# exercise the Layer chaining contract, not the pipeline builder.


# ---------------------------------------------------------------------------
# execute_model_pipeline maps over all grids
# ---------------------------------------------------------------------------


def test_execute_pipeline_populates_all_grids() -> None:
    from nzcvm.config.metadata import ModelMetadata
    from nzcvm.layers.pipeline import execute_model_pipeline
    from nzcvm.velocity_model import VelocityModel

    # A pair of named grids of different sizes
    grids = {
        "a": make_grid(2, 2, 2),
        "b": make_grid(3, 3, 2),
    }
    vm = VelocityModel(grids=grids, metadata=ModelMetadata())

    pipeline = ConstantLayer(rho=1234.0)
    result = execute_model_pipeline(vm, pipeline)

    assert set(result.qualities.keys()) == {"a", "b"}
    # Dask-backed: compute before asserting values
    rho_a = float(result.qualities["a"].rho.values.mean())
    rho_b = float(result.qualities["b"].rho.values.mean())
    assert rho_a == pytest.approx(1234.0, rel=1e-4)
    assert rho_b == pytest.approx(1234.0, rel=1e-4)
