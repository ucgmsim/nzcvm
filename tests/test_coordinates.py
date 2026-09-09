"""Tests for nzcvm.coordinates affine transform factories.

The tests come in two complementary layers.

1. **Structural**: each factory produces a matrix with the correct entries in
   the correct cells.  Cheap, and the shape assertions are real.
2. **Behavioural**: the matrices do the right thing to actual points when fed
   through :func:`~nzcvm.coordinates.apply_affine_transform`, which is what the
   grid builders actually call.

The second layer exists because structural assertions can't separate a row-vector convention
change that still composes correctly (all structural tests fail, nothing
actually wrong) from a matrix that's structurally right but composes wrong
(all structural tests pass, everything actually wrong).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nzcvm.coordinates import apply_affine_transform, reflect_x, scale, translate

# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------


@given(
    dx=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    dy=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
)
def test_translate_2d_encodes_offsets(dx: float, dy: float) -> None:
    """translate(dx, dy) must place dx at [0,2] and dy at [1,2].

    The matrix stores values as float32, so the assertion compares against the
    float32-rounded input rather than the original float64.
    """
    T = translate(dx, dy)
    assert T.shape == (3, 3)
    assert T[0, 2] == np.float32(dx)
    assert T[1, 2] == np.float32(dy)


@given(
    dx=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    dy=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    dz=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
)
def test_translate_3d_encodes_offsets(dx: float, dy: float, dz: float) -> None:
    T = translate(dx, dy, z=dz)
    assert T.shape == (4, 4)
    assert T[0, 3] == np.float32(dx)
    assert T[1, 3] == np.float32(dy)
    assert T[2, 3] == np.float32(dz)


def test_translate_2d_linear_part_is_identity() -> None:
    T = translate(5.0, 7.0)
    np.testing.assert_array_equal(T[:2, :2], np.eye(2, dtype=np.float32))


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------


@given(
    sx=st.floats(0.1, 10.0, allow_nan=False),
    sy=st.floats(0.1, 10.0, allow_nan=False),
)
def test_scale_2d_encodes_factors(sx: float, sy: float) -> None:
    S = scale(sx, sy)
    assert S.shape == (3, 3)
    assert S[0, 0] == np.float32(sx)
    assert S[1, 1] == np.float32(sy)


@given(
    sx=st.floats(0.1, 10.0, allow_nan=False),
    sy=st.floats(0.1, 10.0, allow_nan=False),
    sz=st.floats(0.1, 10.0, allow_nan=False),
)
def test_scale_3d_encodes_factors(sx: float, sy: float, sz: float) -> None:
    S = scale(sx, sy, sz=sz)
    assert S.shape == (4, 4)
    assert S[0, 0] == np.float32(sx)
    assert S[1, 1] == np.float32(sy)
    assert S[2, 2] == np.float32(sz)


def test_scale_2d_off_diagonal_is_zero() -> None:
    S = scale(2.0, 3.0)
    assert float(S[0, 1]) == 0.0
    assert float(S[1, 0]) == 0.0


# ---------------------------------------------------------------------------
# reflect_x
# ---------------------------------------------------------------------------


def test_reflect_x_2d_negates_x_diagonal() -> None:
    R = reflect_x()
    assert R.shape == (3, 3)
    assert float(R[0, 0]) == -1.0
    assert float(R[1, 1]) == 1.0


def test_reflect_x_3d_negates_x_diagonal() -> None:
    R = reflect_x(dims=3)
    assert R.shape == (4, 4)
    assert float(R[0, 0]) == -1.0
    assert float(R[1, 1]) == 1.0
    assert float(R[2, 2]) == 1.0


# ---------------------------------------------------------------------------
# Behavioural layer: what the matrices do to points
# ---------------------------------------------------------------------------

_COORD = st.floats(-1e4, 1e4, allow_nan=False, allow_infinity=False, width=32)


def _apply(transform, x: float, y: float) -> tuple[float, float]:
    xs = np.array([x], dtype=np.float32)
    ys = np.array([y], dtype=np.float32)
    xo, yo = apply_affine_transform(transform, xs, ys)
    return float(xo[0]), float(yo[0])


@given(x=_COORD, y=_COORD, dx=_COORD, dy=_COORD)
def test_translate_shifts_points(x: float, y: float, dx: float, dy: float) -> None:
    got = _apply(translate(dx, dy), x, y)
    assert got == pytest.approx((x + dx, y + dy), rel=1e-5, abs=1e-2)


@given(
    x=_COORD,
    y=_COORD,
    # 0.125 rather than 0.1: `width=32` requires exactly representable bounds.
    sx=st.floats(0.125, 10.0, allow_nan=False, width=32),
    sy=st.floats(0.125, 10.0, allow_nan=False, width=32),
)
def test_scale_multiplies_points(x: float, y: float, sx: float, sy: float) -> None:
    got = _apply(scale(sx, sy), x, y)
    assert got == pytest.approx((x * sx, y * sy), rel=1e-5, abs=1e-2)


@given(x=_COORD, y=_COORD)
def test_reflect_x_is_an_involution(x: float, y: float) -> None:
    """Applying reflect_x twice must return the original point.

    Both as a composed matrix and as two successive applications.
    """
    R = reflect_x()
    once = _apply(R, x, y)
    assert once == pytest.approx((-x, y), rel=1e-5, abs=1e-2)

    twice = _apply(R, *once)
    assert twice == pytest.approx((x, y), rel=1e-5, abs=1e-2)

    composed = _apply(R @ R, x, y)
    assert composed == pytest.approx((x, y), rel=1e-5, abs=1e-2)


@given(x=_COORD, y=_COORD, dx=_COORD, dy=_COORD)
def test_matrix_product_applies_right_factor_first(
    x: float, y: float, dx: float, dy: float
) -> None:
    """``A @ B`` must apply ``B`` and then ``A``.

    This is the composition order the grid builders rely on:
    ``translate(origin) @ rotation`` rotates about the origin and *then*
    translates.  Swapping the operands translates first and then rotates the
    translation itself, putting the grid in the wrong place entirely.
    """
    A = translate(dx, dy)
    B = scale(2.0, 3.0)

    # A @ B: scale first, then translate.
    assert _apply(A @ B, x, y) == pytest.approx(
        (x * 2.0 + dx, y * 3.0 + dy), rel=1e-5, abs=1e-2
    )
    # B @ A translates first and then scales, which scales the translation too.
    assert _apply(B @ A, x, y) == pytest.approx(
        ((x + dx) * 2.0, (y + dy) * 3.0), rel=1e-5, abs=1e-2
    )


@given(x=_COORD, y=_COORD, dx=_COORD, dy=_COORD)
def test_composition_matches_successive_application(
    x: float, y: float, dx: float, dy: float
) -> None:
    """Composing then applying equals applying one at a time."""
    A = translate(dx, dy)
    B = reflect_x()

    composed = _apply(A @ B, x, y)
    stepwise = _apply(A, *_apply(B, x, y))
    assert composed == pytest.approx(stepwise, rel=1e-5, abs=1e-2)
