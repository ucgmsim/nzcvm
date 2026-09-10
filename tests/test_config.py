"""Tests for configuration validation logic and config dispatch.

Covers two concerns:

1. The validator functions in :mod:`nzcvm.config.validation`, checked
   with Hypothesis property tests where natural and contract-based unit
   tests otherwise.
2. The layer and grid config dispatch that maps a :class:`LayerConfig`
   subclass to the corresponding :class:`Layer` subclass via
   :func:`~nzcvm.layers.core.layer_from_config`.

These tests leave Mashumaro TOML/YAML/JSON decoding alone, since that's the
library's responsibility.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from mashumaro.config import BaseConfig
from mashumaro.exceptions import ExtraKeysError, InvalidFieldValue

from nzcvm.config.core import ConfigObject
from nzcvm.config.grids.core import GridConfig
from nzcvm.config.grids.model import Model
from nzcvm.config.grids.regular import RegularGridConfig
from nzcvm.config.layers.clamp import Bound, ClampLayerConfig
from nzcvm.config.layers.core import LayerConfig
from nzcvm.config.layers.ely import ElyLayerConfig
from nzcvm.config.layers.offshore import VelocityModel1D
from nzcvm.config.metadata import ModelMetadata
from nzcvm.config.validation import (
    ge,
    gt,
    in_choices,
    latitude,
    le,
    longitude,
    lt,
    max_len,
    min_len,
    regex,
    validate_non_negative,
    validate_positive,
)
from nzcvm.layers.clamp import ClampLayer
from nzcvm.layers.core import layer_from_config

# ---------------------------------------------------------------------------
# Numeric validators
# ---------------------------------------------------------------------------


@given(st.floats(min_value=1e-9, max_value=1e9, allow_nan=False))
def test_validate_positive_accepts_positive(v: float) -> None:
    assert validate_positive(v) == v


@given(st.floats(max_value=0.0, allow_nan=False))
def test_validate_positive_rejects_non_positive(v: float) -> None:
    with pytest.raises(ValueError):
        validate_positive(v)


@given(st.floats(min_value=0.0, allow_nan=False, allow_infinity=False))
def test_validate_non_negative_accepts(v: float) -> None:
    assert validate_non_negative(v) == v


@given(st.floats(max_value=-1e-9, allow_nan=False))
def test_validate_non_negative_rejects_negative(v: float) -> None:
    with pytest.raises(ValueError):
        validate_non_negative(v)


@given(
    limit=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    v=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
)
def test_gt_accepts_strictly_greater(limit: float, v: float) -> None:
    if v > limit:
        assert gt(limit)(v) == v
    else:
        with pytest.raises(ValueError):
            gt(limit)(v)


@given(
    limit=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    v=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
)
def test_ge_accepts_greater_or_equal(limit: float, v: float) -> None:
    if v >= limit:
        assert ge(limit)(v) == v
    else:
        with pytest.raises(ValueError):
            ge(limit)(v)


@given(
    limit=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    v=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
)
def test_lt_accepts_strictly_less(limit: float, v: float) -> None:
    if v < limit:
        assert lt(limit)(v) == v
    else:
        with pytest.raises(ValueError):
            lt(limit)(v)


@given(
    limit=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    v=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
)
def test_le_accepts_less_or_equal(limit: float, v: float) -> None:
    if v <= limit:
        assert le(limit)(v) == v
    else:
        with pytest.raises(ValueError):
            le(limit)(v)


def test_validate_positive_passes_none() -> None:
    assert validate_positive(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# String / sequence validators
# ---------------------------------------------------------------------------


@given(st.integers(min_value=1, max_value=10))
def test_min_len_accepts_long_enough(n: int) -> None:
    v = "x" * n
    assert min_len(n)(v) == v


@given(st.integers(min_value=1, max_value=10))
def test_min_len_rejects_too_short(n: int) -> None:
    with pytest.raises(ValueError):
        min_len(n + 1)("x" * n)


@given(st.integers(min_value=1, max_value=10))
def test_max_len_accepts_short_enough(n: int) -> None:
    v = "x" * n
    assert max_len(n)(v) == v


@given(st.integers(min_value=1, max_value=10))
def test_max_len_rejects_too_long(n: int) -> None:
    with pytest.raises(ValueError):
        max_len(n)("x" * (n + 1))


def test_regex_accepts_match() -> None:
    assert regex(r"^\d+$")("123") == "123"


def test_regex_rejects_non_match() -> None:
    with pytest.raises(ValueError):
        regex(r"^\d+$")("abc")


# ---------------------------------------------------------------------------
# Geographic validators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "validator, value, range_str",
    [
        (latitude, -90.0, "[-90, 90]"),
        (latitude, 0.0, "[-90, 90]"),
        (latitude, 90.0, "[-90, 90]"),
        (longitude, -180.0, "[-180, 180]"),
        (longitude, 0.0, "[-180, 180]"),
        (longitude, 180.0, "[-180, 180]"),
    ],
)
def test_coordinate_accepts_valid(validator, value, range_str) -> None:
    validator(value)


@pytest.mark.parametrize(
    "validator, value, range_str",
    [
        (latitude, -91.0, "[-90, 90]"),
        (latitude, 91.0, "[-90, 90]"),
        (longitude, -181.0, "[-180, 180]"),
        (longitude, 181.0, "[-180, 180]"),
    ],
)
def test_coordinate_rejects_out_of_range(validator, value, range_str) -> None:
    with pytest.raises(
        ValueError, match=range_str.replace("[", r"\[").replace("]", r"\]")
    ):
        validator(value)


def test_model_wiring_origin_lon_lat_validated() -> None:
    from pyproj import CRS

    from nzcvm.config.grids.model import Model

    Model(origin_lon=172.0, origin_lat=-43.5, azimuth=0.0, crs=CRS.from_epsg(2193))


# ---------------------------------------------------------------------------
# in_choices validator
# ---------------------------------------------------------------------------


@given(st.sampled_from(["a", "b", "c"]))
def test_in_choices_accepts_member(v: str) -> None:
    assert in_choices(["a", "b", "c"])(v) == v


def test_in_choices_rejects_non_member() -> None:
    with pytest.raises(ValueError):
        in_choices(["a", "b"])("z")


# ---------------------------------------------------------------------------
# ClampLayerConfig cross-field validation
# ---------------------------------------------------------------------------


def test_clamp_bound_rejects_inverted_range() -> None:
    with pytest.raises(InvalidFieldValue):
        Bound(min=5.0, max=1.0)


def test_clamp_bound_rejects_non_positive_max() -> None:
    with pytest.raises(InvalidFieldValue):
        Bound(max=-1.0)


def test_clamp_bound_accepts_min_only() -> None:
    b = Bound(min=1.0)
    assert b.min == 1.0 and b.max is None


def test_clamp_bound_accepts_max_only() -> None:
    b = Bound(max=5.0)
    assert b.max == 5.0 and b.min is None


def test_clamp_config_rejects_inverted_vp_vs_ratio() -> None:
    with pytest.raises(InvalidFieldValue):
        ClampLayerConfig(min_vp_vs_ratio=3.0, max_vp_vs_ratio=1.0)


def test_clamp_config_accepts_valid_vp_vs_ratio() -> None:
    c = ClampLayerConfig(min_vp_vs_ratio=1.5, max_vp_vs_ratio=2.5)
    assert c.min_vp_vs_ratio == 1.5


# ---------------------------------------------------------------------------
# VelocityModel1D physical constraint
# ---------------------------------------------------------------------------


def test_velocity_model_1d_rejects_vp_less_than_vs() -> None:
    with pytest.raises(ValueError, match="vp > vs"):
        VelocityModel1D(
            bottom_depth=100.0,
            rho=2000.0,
            vp=2000.0,
            vs=3000.0,  # vs > vp: illegal
            qp=100.0,
            qs=50.0,
            alpha=1.0,
        )


def test_velocity_model_1d_rejects_equal_vp_vs() -> None:
    with pytest.raises(ValueError):
        VelocityModel1D(
            bottom_depth=0.0,
            rho=2000.0,
            vp=3000.0,
            vs=3000.0,
            qp=100.0,
            qs=50.0,
            alpha=1.0,
        )


# ---------------------------------------------------------------------------
# Layer config dispatch
# ---------------------------------------------------------------------------


def test_layer_from_config_clamp() -> None:
    cfg = ClampLayerConfig()
    assert layer_from_config(cfg) is ClampLayer


# ---------------------------------------------------------------------------
# Mashumaro reads the settings on ConfigObject
#
# Mashumaro looks for a nested class named `Config`. Naming it anything else
# leaves every setting on it inert, which is a silent failure: the decoder
# keeps working and simply stops enforcing what the class asked for.
# ---------------------------------------------------------------------------


def _config_of(cls: type) -> type[BaseConfig]:
    """The config class mashumaro resolves for *cls*, however it inherits it."""
    return getattr(cls, "Config", BaseConfig)


@pytest.mark.parametrize(
    "config_cls",
    [ConfigObject, LayerConfig, GridConfig, ClampLayerConfig, RegularGridConfig],
)
def test_mashumaro_sees_the_project_settings(config_cls: type) -> None:
    """The discriminated bases declare their own `Config`, so they have to
    inherit the project one rather than replace it."""
    resolved = _config_of(config_cls)
    assert resolved.forbid_extra_keys is True
    assert resolved.omit_none is True
    assert resolved.serialize_by_alias is True


@pytest.mark.parametrize(
    "config_cls, payload",
    [
        (
            Model,
            {
                "origin_lon": 172.0,
                "origin_lat": -43.5,
                "azimuth": 0.0,
                "crs": 2193,
                "azimuth_deg": 39.0,
            },
        ),
        (ClampLayerConfig, {"type": "clamp", "min_vp_vs_ratio": 1.7, "typo": 1}),
        (LayerConfig, {"type": "clamp", "min_vp_vs_ratio": 1.7, "typo": 1}),
    ],
)
def test_an_unknown_key_is_reported(
    config_cls: type[ConfigObject], payload: dict
) -> None:
    """A misspelled key used to vanish. The default stayed in place."""
    with pytest.raises(ExtraKeysError, match="typo|azimuth_deg"):
        config_cls.from_dict(payload)


def test_a_known_key_still_decodes() -> None:
    clamp = ClampLayerConfig.from_dict({"type": "clamp", "min_vp_vs_ratio": 1.7})
    assert clamp.min_vp_vs_ratio == pytest.approx(1.7)


def test_derived_layer_fields_survive_a_round_trip() -> None:
    """`provides` and `requires` are `init=False`, so mashumaro can't feed
    them back in. Serialising them would produce output that fails to decode."""
    config = ElyLayerConfig(vs30=Path("vs30.zarr"))
    assert config.requires == ["coastline"]

    serialised = config.to_dict()
    assert "requires" not in serialised
    assert "provides" not in serialised

    assert LayerConfig.from_dict(serialised).requires == ["coastline"]


def test_omit_none_drops_unset_metadata() -> None:
    """The metadata goes onto the output as dataset attributes, where a null
    is worth nothing."""
    metadata = ModelMetadata(title="A title").to_dict()
    assert metadata["title"] == "A title"
    assert "creator_name" not in metadata
