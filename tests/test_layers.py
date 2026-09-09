"""Tests for individual pipeline layers.

Layers are tested in isolation using the dummy layers from
:mod:`nzcvm.layers.dummy` as inner stubs.  No real model files or
surface grids are required.

Test strategy (in descending preference):
1. Hypothesis property tests where the property is expressible symbolically.
2. Contract / unit tests for non-hypothesis-friendly properties.
3. Behavioural integration using composed dummy layers.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from hypothesis import assume, given
from hypothesis import strategies as st

from nzcvm.components import Component
from nzcvm.config.layers.clamp import Bound, ClampLayerConfig
from nzcvm.config.layers.offshore import (
    DepthModel,
    OffshoreBasinConfig,
    VelocityModel1D,
)
from nzcvm.coordinates import Coordinate
from nzcvm.grids.grid import Grid
from nzcvm.layers.clamp import ClampLayer
from nzcvm.layers.core import Layer
from nzcvm.layers.dummy import ConstantLayer, CountingLayer, RecordingLayer
from nzcvm.layers.offshore import OffshoreBasinLayer, step_interpolator
from nzcvm.qualities import Qualities
from nzcvm.query import ModelRange
from tests.conftest import make_grid

# Layers carry a spatial domain (``Layer.geometry``), but no layer currently
# consults it when evaluating a grid — see
# ``test_layer_geometry_masks_output`` below.  Any covering geometry works.
GEOM = shapely.box(171.9, -43.6, 172.1, -43.4)

# A geometry that shares no area with the grids built by ``make_grid``.
DISJOINT_GEOM = shapely.box(0.0, 0.0, 1.0, 1.0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_over_constant(
    config: ClampLayerConfig,
    **constants,
) -> Qualities:
    """Apply *config* to a 2×2×2 grid backed by a ConstantLayer(*constants*)."""
    grid = make_grid()
    inner = ConstantLayer(**constants)
    layer = ClampLayer(config, GEOM, inner)
    return layer(grid)


# ---------------------------------------------------------------------------
# ClampLayer: per-component bounds
# ---------------------------------------------------------------------------


@given(
    vs_min=st.floats(100.0, 2000.0, allow_nan=False, width=32),
    vs_max=st.floats(2001.0, 6000.0, allow_nan=False, width=32),
    vs_val=st.floats(10.0, 10000.0, allow_nan=False, width=32),
)
def test_clamp_vs_is_the_projection_onto_the_interval(
    vs_min: float, vs_max: float, vs_val: float
) -> None:
    cfg = ClampLayerConfig(clamps={Component.VS: Bound(min=vs_min, max=vs_max)})
    expected = np.float32(min(max(vs_val, vs_min), vs_max))
    assert np.all(_clamp_over_constant(cfg, vs=vs_val).vs.values == expected)
    assert np.all(_clamp_over_constant(cfg, vs=float(expected)).vs.values == expected)


def test_clamp_qs_floored_to_multiple_of_vs() -> None:
    """A relative min bound floors Qs at ``factor * Vs`` (EMOD3D Qs = 50*Vs)."""
    cfg = ClampLayerConfig(
        clamps={Component.QS: Bound(min=0.05, min_ref="vs")}  # ty: ignore[invalid-argument-type]
    )
    # Vs = 3000 m/s -> floor 0.05 * 3000 = 150; input Qs = 40 is below it.
    result = _clamp_over_constant(cfg, vs=3000.0, qs=40.0)
    assert float(result.qs.mean()) == pytest.approx(150.0, rel=1e-4)


def test_clamp_qp_capped_to_multiple_of_vp() -> None:
    """A relative max bound caps Qp at ``factor * Vp``; below the cap is untouched."""
    cfg = ClampLayerConfig(
        clamps={Component.QP: Bound(max=0.1, max_ref="vp")}  # ty: ignore[invalid-argument-type]
    )
    # Vp = 5000 m/s -> cap 0.1 * 5000 = 500; input Qp = 900 is above it, 200 is not.
    capped = _clamp_over_constant(cfg, vp=5000.0, qp=900.0)
    assert float(capped.qp.mean()) == pytest.approx(500.0, rel=1e-4)
    below = _clamp_over_constant(cfg, vp=5000.0, qp=200.0)
    assert float(below.qp.mean()) == pytest.approx(200.0, rel=1e-4)


def test_clamp_relative_bound_uses_clamped_vs() -> None:
    """Qs floored to k*Vs tracks the *clamped* Vs, since Vs is finalised first."""
    cfg = ClampLayerConfig(
        clamps={
            Component.VS: Bound(min=4000.0),  # forces Vs 3000 -> 4000
            Component.QS: Bound(min=0.05, min_ref="vs"),  # ty: ignore[invalid-argument-type]
        }
    )
    result = _clamp_over_constant(cfg, vs=3000.0, qs=40.0)
    assert float(result.vs.mean()) == pytest.approx(4000.0, rel=1e-4)
    assert float(result.qs.mean()) == pytest.approx(0.05 * 4000.0, rel=1e-4)


def test_clamp_bound_rejects_unknown_ref() -> None:
    """A ``*_ref`` naming a non-component is rejected at config time."""
    from mashumaro.exceptions import InvalidFieldValue

    with pytest.raises(InvalidFieldValue):
        Bound(min=0.05, min_ref="not_a_component")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "vs, expected_vp, expected_rho",
    [
        (1500.0, 3015.044, 2227.043),
        (3000.0, 5050.601, 2542.342),
        (4000.0, 6935.700, 2949.165),
        (4500.0, 7906.168, 3257.309),
    ],
)
def test_clamp_vs_snaps_vp_and_rho_onto_manifold(
    vs: float,
    expected_vp: float,
    expected_rho: float,
) -> None:
    """Clamping Vs regenerates Vp and density from Brocher/Nafe-Drake.

    Golden values are derived from Brocher (2005) eq. 9 and Nafe-Drake
    eq. 1; the test uses hardcoded expectations so that a coefficient
    change requires deliberate update, not silent pass-through.
    """
    cfg = ClampLayerConfig(clamps={Component.VS: Bound(min=vs)})
    result = _clamp_over_constant(cfg, vs=vs - 1.0, vp=1.0, rho=1.0)
    assert float(result.vp.mean()) == pytest.approx(expected_vp, rel=1e-4)
    assert float(result.rho.mean()) == pytest.approx(expected_rho, rel=1e-4)


def test_clamp_vs_untouched_leaves_vp_rho_alone() -> None:
    """Where Vs is not moved by the clamp, Vp and density are left as-is."""
    cfg = ClampLayerConfig(clamps={Component.VS: Bound(min=1000.0)})
    # vs 3500 already satisfies the bound, so nothing is snapped.
    result = _clamp_over_constant(cfg, vs=3500.0, vp=6000.0, rho=2700.0)
    assert float(result.vp.mean()) == pytest.approx(6000.0, rel=1e-4)
    assert float(result.rho.mean()) == pytest.approx(2700.0, rel=1e-4)


# ---------------------------------------------------------------------------
# ClampLayer: vp/vs ratio
# ---------------------------------------------------------------------------


# The two ratio bounds are the same contract with the comparison reversed, so
# they are parametrised over ``(config field, comparison)`` rather than
# duplicated.  Note @pytest.mark.parametrize must sit *outside* @given.
@pytest.mark.parametrize(
    "field, compare",
    [
        ("min_vp_vs_ratio", np.greater_equal),
        ("max_vp_vs_ratio", np.less_equal),
    ],
)
@given(
    # 1.125 rather than 1.1: `width=32` requires exactly-representable bounds.
    ratio=st.floats(min_value=1.125, max_value=3.0, allow_nan=False, width=32),
    vs=st.floats(min_value=500.0, max_value=4000.0, allow_nan=False, width=32),
    vp=st.floats(min_value=100.0, max_value=8000.0, allow_nan=False, width=32),
)
def test_clamp_vp_vs_ratio_enforced(
    field: str, compare, ratio: float, vs: float, vp: float
) -> None:
    """After clamping, ``vp`` sits on the correct side of ``ratio * vs``."""
    cfg = ClampLayerConfig(**{field: ratio})  # ty: ignore[invalid-argument-type]
    result = _clamp_over_constant(cfg, vp=vp, vs=vs)
    vp_arr = result.vp.values.astype(np.float64)
    vs_arr = result.vs.values.astype(np.float64)
    bound = float(np.float32(ratio)) * vs_arr
    # Small absolute slack for the float32 -> float64 round-trip.
    slack = 1e-3 if compare is np.greater_equal else -1e-3
    assert np.all(compare(vp_arr, bound - slack)), (
        f"{field}={ratio}: vp={vp_arr.min()} violates the bound {bound.min()}"
    )


@pytest.mark.parametrize(
    "field, ratio, vs, vp_in, vp_out",
    [
        # vp starts inside the bound and must be left alone...
        ("min_vp_vs_ratio", 1.5, 2000.0, 6000.0, 6000.0),
        ("max_vp_vs_ratio", 3.0, 2000.0, 4000.0, 4000.0),
        # ...and starts outside, so it must be moved exactly onto it.
        ("min_vp_vs_ratio", 2.0, 2000.0, 1000.0, 4000.0),
        ("max_vp_vs_ratio", 2.0, 2000.0, 8000.0, 4000.0),
    ],
)
def test_clamp_vp_vs_ratio_moves_vp_onto_the_bound(
    field: str, ratio: float, vs: float, vp_in: float, vp_out: float
) -> None:
    """The ratio clamp is a projection: exact on the bound, identity inside it.

    The property test above only pins the inequality, which a clamp that
    over-corrects would also satisfy.
    """
    cfg = ClampLayerConfig(**{field: ratio})  # ty: ignore[invalid-argument-type]
    result = _clamp_over_constant(cfg, vp=vp_in, vs=vs)
    assert float(result.vp.mean()) == pytest.approx(vp_out, rel=1e-4)


# ---------------------------------------------------------------------------
# ClampLayer: chaining contracts
# ---------------------------------------------------------------------------


def test_clamp_delegates_to_next_layer() -> None:
    """ClampLayer must call next_layer exactly once per grid call."""
    cfg = ClampLayerConfig()
    inner = ConstantLayer()
    counter = CountingLayer(GEOM, inner)
    clamp = ClampLayer(cfg, GEOM, counter)
    clamp(make_grid())
    assert counter.call_count == 1


def test_clamp_propagates_model_range() -> None:
    """The model_range kwarg is forwarded to next_layer unchanged."""
    cfg = ClampLayerConfig()
    inner = ConstantLayer()
    recorder = RecordingLayer(GEOM, inner)
    clamp = ClampLayer(cfg, GEOM, recorder)
    clamp(make_grid(), model_range=ModelRange.BASINS)
    assert recorder.calls[0][1] == ModelRange.BASINS


def test_outer_clamp_applies_to_inner_layer_result() -> None:
    """The outermost ClampLayer governs the inner ConstantLayer's output.

    Hand-assembled rather than built through ``build_pipeline``: this is a
    ``Layer`` composition contract, so it belongs here.  The equivalent
    property *through* ``build_pipeline`` lives in ``test_pipeline.py``.
    """
    terminal = ConstantLayer(vs=3000.0)
    counter = CountingLayer(GEOM, terminal)
    clamp = ClampLayer(
        ClampLayerConfig(clamps={Component.VS: Bound(min=4000.0)}), GEOM, counter
    )

    result = clamp(make_grid())

    assert counter.call_count == 1
    assert float(result.vs.min()) == pytest.approx(4000.0, rel=1e-4)


def test_two_clamp_layers_compose_correctly() -> None:
    """Stacking two ClampLayers applies both constraints."""
    terminal = ConstantLayer(vs=3000.0, vp=5000.0)
    clamp_inner = ClampLayer(
        ClampLayerConfig(clamps={Component.VS: Bound(min=3500.0)}), GEOM, terminal
    )
    clamp_outer = ClampLayer(
        ClampLayerConfig(clamps={Component.VP: Bound(max=4500.0)}), GEOM, clamp_inner
    )

    result = clamp_outer(make_grid())
    # vs raised from 3000 -> 3500 by the inner clamp
    assert float(result.vs.min()) >= 3500.0 - 1e-4
    # vp lowered from 5000 -> 4500 by the outer clamp
    assert float(result.vp.max()) <= 4500.0 + 1e-4


# ---------------------------------------------------------------------------
# ConstantLayer contracts (testing the test helper itself)
# ---------------------------------------------------------------------------


@given(
    rho=st.floats(min_value=1.0, max_value=5000.0, allow_nan=False),
    nx=st.integers(min_value=1, max_value=4),
    ny=st.integers(min_value=1, max_value=4),
    nz=st.integers(min_value=1, max_value=4),
)
def test_constant_layer_shape(rho: float, nx: int, ny: int, nz: int) -> None:
    grid = make_grid(nx, ny, nz)
    layer = ConstantLayer(rho=rho)
    result = layer(grid)
    assert result.rho.shape == (nx, ny, nz)


@given(rho=st.floats(min_value=1.0, max_value=5000.0, allow_nan=False))
def test_constant_layer_value(rho: float) -> None:
    grid = make_grid()
    result = ConstantLayer(rho=rho)(grid)
    assert float(result.rho.mean()) == pytest.approx(rho, rel=1e-4)


@st.composite
def _step_segment(draw):
    breaks = draw(st.lists(st.integers(0, 10_000), min_size=2, max_size=8, unique=True))
    xp = np.array(sorted(breaks), dtype=np.float32)
    fp = np.array(
        draw(
            st.lists(
                st.floats(-1e6, 1e6, allow_nan=False, width=32),
                min_size=len(xp),
                max_size=len(xp),
            )
        ),
        dtype=np.float32,
    )
    m = draw(st.integers(0, len(xp) - 1))
    frac = draw(st.floats(0.0, 1.0, exclude_max=True, allow_nan=False))
    upper = xp[m + 1] if m + 1 < len(xp) else xp[m] + np.float32(10.0)
    x = np.float32(xp[m] + frac * (upper - xp[m]))
    assume(xp[m] <= x < upper)
    return xp, fp, m, x


@given(_step_segment())
def test_step_interpolator_is_right_continuous_on_each_segment(args: tuple) -> None:
    xp, fp, m, x = args
    assert step_interpolator(np.array([x], np.float32), xp, fp)[0] == fp[m]
    assert step_interpolator(np.array([xp[m]], np.float32), xp, fp)[0] == fp[m]


def test_step_interpolator_clip_below_first() -> None:
    from nzcvm.layers.offshore import step_interpolator

    xp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fp = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    # x below first break → returns first value
    result = step_interpolator(np.array([0.0], dtype=np.float32), xp, fp)
    assert result[0] == pytest.approx(10.0)


def test_step_interpolator_clip_above_last() -> None:
    from nzcvm.layers.offshore import step_interpolator

    xp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    fp = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    result = step_interpolator(np.array([100.0], dtype=np.float32), xp, fp)
    assert result[0] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# functional_layer deserialization
# ---------------------------------------------------------------------------
# The @functional_layer decorator generates a LayerConfig subclass dynamically
# via make_dataclass.  The non-trivial invariants are:
#   - the generated config has the right type tag
#   - it round-trips through dict serialisation
#   - mashumaro's discriminator (include_subtypes=True) finds the subclass
#     and reconstructs the correct type from the base LayerConfig.from_dict
#   - the layer class is registered in Layer.registry under its config class
#   - a layer instantiated from the reconstructed config produces correct output


def test_functional_layer_config_has_correct_type_tag() -> None:
    from nzcvm.layers.dummy import constant

    cfg = constant.config_cls()
    assert cfg.type == "constant"  # ty: ignore[unresolved-attribute]


def test_functional_layer_config_accepts_custom_parameters() -> None:
    from nzcvm.layers.dummy import constant

    cfg = constant.config_cls(vs=1234.0, vp=5678.0)  # ty: ignore[unknown-argument]
    assert cfg.vs == pytest.approx(1234.0)  # ty: ignore[unresolved-attribute]
    assert cfg.vp == pytest.approx(5678.0)  # ty: ignore[unresolved-attribute]


def test_functional_layer_config_serialises_to_dict() -> None:
    from nzcvm.layers.dummy import constant

    cfg = constant.config_cls(vs=999.0)  # ty: ignore[unknown-argument]
    d = cfg.to_dict()
    assert d["type"] == "constant"
    assert d["vs"] == pytest.approx(999.0)


def test_functional_layer_config_deserialises_via_base_class() -> None:
    """LayerConfig.from_dict must reconstruct the correct generated subclass."""
    from nzcvm.config.layers.core import LayerConfig
    from nzcvm.layers.dummy import constant

    cfg = constant.config_cls(vs=3210.0)  # ty: ignore[unknown-argument]
    d = cfg.to_dict()

    cfg2 = LayerConfig.from_dict(d)
    assert type(cfg2) is type(cfg)
    assert cfg2.vs == pytest.approx(3210.0)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    assert cfg2.type == "constant"  # ty: ignore[unresolved-attribute]


def test_functional_layer_registered_in_layer_registry() -> None:
    """The layer class must appear in Layer.registry under its config class."""
    from nzcvm.layers.core import Layer
    from nzcvm.layers.dummy import constant

    assert constant.config_cls in Layer.registry
    assert Layer.registry[constant.config_cls] is constant


def test_functional_layer_instantiation_from_deserialised_config() -> None:
    """End-to-end: config → dict → LayerConfig.from_dict → Layer → call."""
    from nzcvm.config.layers.core import LayerConfig
    from nzcvm.layers.core import Layer
    from nzcvm.layers.dummy import constant
    from tests.conftest import make_grid

    d = constant.config_cls(vs=2000.0).to_dict()  # ty: ignore[unknown-argument]
    cfg = LayerConfig.from_dict(d)

    LayerCls = Layer.registry[type(cfg)]
    # Rebuild kwargs from the deserialised config (excluding the 'type' tag)
    kwargs = {k: v for k, v in cfg.to_dict().items() if k != "type"}
    layer = LayerCls(**kwargs)

    result = layer(make_grid())
    assert float(result.vs.mean()) == pytest.approx(2000.0, rel=1e-4)


def test_adhoc_functional_layer_deserialises(isolated_layer_registry: None) -> None:
    """An ad-hoc @functional_layer defined inside a test must round-trip through
    LayerConfig.from_dict — verifying that layers created outside dummy.py work."""
    from nzcvm.config.layers.core import LayerConfig
    from nzcvm.layers.core import Layer
    from nzcvm.layers.functional import functional_layer
    from nzcvm.qualities import QualitiesSchema

    @functional_layer
    def zeros(
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
        *,
        next_layer: Layer | None = None,
    ) -> Qualities:
        """Return all-zero qualities for every point in the grid."""
        import numpy as np

        shape = grid.x.shape
        z = np.zeros(shape, dtype=np.float32)
        return QualitiesSchema.new(rho=z, vp=z, vs=z, qp=z, qs=z, alpha=z)

    # Config must carry the right type tag
    cfg = zeros.config_cls()
    assert cfg.type == "zeros"  # ty: ignore[unresolved-attribute]

    # Must survive a dict round-trip via the base LayerConfig discriminator
    d = cfg.to_dict()
    assert d["type"] == "zeros"
    cfg2 = LayerConfig.from_dict(d)
    assert type(cfg2) is type(cfg)

    # Must be findable in Layer.registry and produce the right output
    assert zeros.config_cls in Layer.registry
    layer = zeros()
    result = layer(make_grid())
    assert float(result.vp.max()) == pytest.approx(0.0)
    assert float(result.vs.max()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# OffshoreBasinLayer: a basin outcropping at the surface vetoes the column
# ---------------------------------------------------------------------------

BASIN_BOTTOM_DEPTH = 50.0
BASIN_VS = 111.0
TOMO_VS = 3333.0
OFFSHORE_VS = 777.0


class _OutcroppingBasinLayer(Layer):
    """Terminal stub: a basin outcrops over ``i == 0`` but bottoms out at 50 m.

    This is precisely the case the lateral veto exists for.  Below the basin
    bottom the basin is absent (``alpha == 0``), so alpha compositing on its
    own would happily let the offshore profile through.
    """

    def __init__(self, geometry) -> None:
        from nzcvm.layers.dummy import _NullConfig

        super().__init__(_NullConfig(), geometry, None)  # ty: ignore[invalid-argument-type]

    def __call__(
        self, grid: Grid, model_range: ModelRange = ModelRange.ALL
    ) -> Qualities:
        from nzcvm.qualities import QualitiesSchema

        shape = grid.x.shape
        in_basin = np.zeros(shape, dtype=bool)
        in_basin[0] = True
        in_basin &= grid.depth.values < BASIN_BOTTOM_DEPTH

        background = 0.0 if model_range == ModelRange.BASINS else TOMO_VS
        vs = np.where(in_basin, BASIN_VS, background).astype(np.float32)
        ones = np.ones(shape, dtype=np.float32)
        return QualitiesSchema.new(
            rho=ones * 2000.0,
            vp=ones * 6000.0,
            vs=vs,
            qp=ones * 100.0,
            qs=ones * 50.0,
            alpha=in_basin.astype(np.float32),
        )


def _offshore_config() -> OffshoreBasinConfig:
    """Offshore profile with one distinctive velocity over the top 1000 m."""
    return OffshoreBasinConfig(
        basin_depth=[
            DepthModel(distance=0.0, bottom_depth=1000.0),
            DepthModel(distance=50_000.0, bottom_depth=1000.0),
        ],
        # ``_build_model_interpolator`` keeps only layers shallower than the
        # deepest basin_depth entry, so the first must sit above 1000 m.
        model=[
            VelocityModel1D(
                bottom_depth=500.0,
                rho=1810.0,
                vp=1800.0,
                vs=OFFSHORE_VS,
                qp=100.0,
                qs=50.0,
                alpha=1.0,
            ),
            VelocityModel1D(
                bottom_depth=5000.0,
                rho=1810.0,
                vp=1800.0,
                vs=OFFSHORE_VS,
                qp=100.0,
                qs=50.0,
                alpha=1.0,
            ),
        ],
    )


def _run_offshore(depth0: float) -> Qualities:
    """Run the offshore layer over an entirely-offshore grid at *depth0*."""
    grid = make_grid(nx=2, ny=2, nz=2, depth0=depth0)
    grid[Coordinate.COASTLINE] = (
        (Coordinate.I, Coordinate.J),
        np.full((2, 2), 5_000.0, dtype=np.float32),  # 5 km offshore everywhere
    )
    layer = OffshoreBasinLayer(_offshore_config(), GEOM, _OutcroppingBasinLayer(GEOM))
    return layer(grid)


def test_offshore_suppressed_under_a_basin_that_reaches_the_surface() -> None:
    """A surface basin disables the offshore profile for the whole column."""
    vs = _run_offshore(depth0=100.0).vs.values

    # i == 0 outcrops a basin.  The basin has bottomed out well above 100 m,
    # but the offshore profile must still stay out of the column.
    assert np.all(np.isclose(vs[0], TOMO_VS))

    # i == 1 is basin-free and offshore, so the profile applies.
    assert np.all(np.isclose(vs[1], OFFSHORE_VS))


@pytest.mark.parametrize("depth0", [100.0, 500.0, 900.0])
def test_offshore_veto_holds_below_the_basin_bottom(depth0: float) -> None:
    """The veto is column-wide: it does not lapse below the basin surface."""
    vs = _run_offshore(depth0=depth0).vs.values

    assert not np.any(np.isclose(vs[0], OFFSHORE_VS))
    assert np.all(np.isclose(vs[1], OFFSHORE_VS))
