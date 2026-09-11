"""Tests for the Rust coastline index and the layer that drives it.

Shapely is the oracle throughout. It solves both halves of the problem
independently (`shapely.distance` for the magnitude, `shapely.covers` for the
sign), and this code replaced it, so agreeing with shapely is the whole
specification.

The extension works in `Real`, which is float32 unless someone builds it with
the `high_precision` feature. Over New Zealand a projected easting runs to
about 1.6e6 m, where consecutive float32 values sit 0.125 m apart, so the
assertions below measure against that resolution rather than against an
absolute tolerance.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
import shapely
import shapely.ops
import xarray as xr

from nzcvm import synthetic
from nzcvm.config.layers.coastline import CoastlineConfig
from nzcvm.coordinates import Coordinate
from nzcvm.layers.coastline import CoastlineLayer, _extract_segments
from nzcvm.nzcvm import coastline as build_coastline  # ty: ignore[unresolved-import]
from tests.conftest import Terminal, make_grid

#: Largest coordinate the synthetic coastline uses, in metres.
NZTM_MAGNITUDE = np.float32(1.6e6)
#: Distance below which float32 can't say which side of the coast a point is.
RESOLUTION = float(np.spacing(NZTM_MAGNITUDE))


def _segments(polygon: shapely.Polygon) -> np.ndarray:
    return _extract_segments(shapely.ops.orient(polygon, sign=1.0)).astype(np.float32)


@pytest.fixture(scope="module")
def land() -> shapely.Polygon:
    """The synthetic land polygon, projected into NZTM."""
    from pyproj import Transformer

    to_nztm = Transformer.from_crs(4326, 2193, always_xy=True)
    return shapely.ops.transform(to_nztm.transform, shapely.Polygon(synthetic.land()))


@pytest.fixture(scope="module")
def indexed(land: shapely.Polygon):
    return build_coastline(_segments(land))


@pytest.fixture(scope="module")
def sample(land: shapely.Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Points spread over the polygon's bounds and well beyond them."""
    min_x, min_y, max_x, max_y = land.bounds
    rng = np.random.default_rng(0)
    pad = 5000.0
    x = rng.uniform(min_x - pad, max_x + pad, 4000).astype(np.float32)
    y = rng.uniform(min_y - pad, max_y + pad, 4000).astype(np.float32)
    return x, y


# ---------------------------------------------------------------------------
# Against shapely
# ---------------------------------------------------------------------------


def test_magnitude_matches_shapely(
    indexed, land: shapely.Polygon, sample: tuple[np.ndarray, np.ndarray]
) -> None:
    """Distance to the boundary, ignoring which side it falls on."""
    x, y = sample
    got = np.abs(indexed.signed_distance(x, y))
    want = shapely.distance(shapely.points(x, y), land.boundary)
    error = np.abs(got - want) / RESOLUTION

    # Rounding the closest point at NZTM magnitude costs half a step per
    # coordinate, and the rest comes from solving for it, so half a step is
    # the expected middle of the distribution. Measured over 40,000 points the
    # median is 0.48 steps and the maximum 3.5; the bounds below leave room
    # for a different sample without leaving room for a wrong answer.
    #
    # The median is the assertion that bites: picking the wrong segment would
    # be wrong by metres, not by a rounding step.
    assert np.median(error) < 1.0
    assert error.max() < 8.0


def test_sign_matches_shapely(
    indexed, land: shapely.Polygon, sample: tuple[np.ndarray, np.ndarray]
) -> None:
    """Negative inside, positive outside.

    This skips points nearer the coast than float32 can resolve, since there
    is no fact of the matter about which side of the line they fall on.
    """
    x, y = sample
    signed = indexed.signed_distance(x, y)
    resolved = np.abs(signed) > RESOLUTION
    assert resolved.sum() > 0.99 * len(x), "the sample should be mostly unambiguous"

    inside = shapely.covers(land, shapely.points(x, y))
    assert np.array_equal(signed[resolved] < 0, inside[resolved])


def test_both_sides_are_represented(
    indexed, sample: tuple[np.ndarray, np.ndarray]
) -> None:
    """A test that only ever saw one side would pass without proving much."""
    signed = indexed.signed_distance(*sample)
    assert (signed < 0).any() and (signed > 0).any()


# ---------------------------------------------------------------------------
# Closed form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "point, expected",
    [
        ((0.5, 0.5), -0.5),  # centre of the unit square
        ((0.25, 0.5), -0.25),  # nearer the west edge
        ((-1.0, 0.5), 1.0),  # due west
        ((2.0, 0.5), 1.0),  # due east
        ((0.5, 2.0), 1.0),  # due north
    ],
)
def test_unit_square_has_the_analytic_distance(
    point: tuple[float, float], expected: float
) -> None:
    square = build_coastline(_segments(shapely.box(0.0, 0.0, 1.0, 1.0)))
    got = square.signed_distance(
        np.array([point[0]], np.float32), np.array([point[1]], np.float32)
    )
    assert got[0] == pytest.approx(expected, abs=1e-5)


def test_a_hole_reads_as_outside() -> None:
    """Parity, not winding, decides the sign, so a ring inside a ring is sea."""
    outer = shapely.box(0.0, 0.0, 10.0, 10.0)
    donut = shapely.difference(outer, shapely.box(4.0, 4.0, 6.0, 6.0))
    indexed = build_coastline(_segments(donut))
    signed = indexed.signed_distance(
        np.array([5.0, 2.0], np.float32), np.array([5.0, 2.0], np.float32)
    )
    assert signed[0] > 0, "the hole is outside the land"
    assert signed[1] < 0, "the ring itself is land"


# ---------------------------------------------------------------------------
# Binding contract
# ---------------------------------------------------------------------------


def test_length_mismatch_is_rejected(indexed) -> None:
    with pytest.raises(ValueError, match="same length"):
        indexed.signed_distance(np.zeros(3, np.float32), np.zeros(4, np.float32))


def test_a_bad_segment_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\(S, 2, 2\)"):
        build_coastline(np.zeros((4, 3, 2), np.float32))


def test_len_reports_the_segment_count(indexed, land: shapely.Polygon) -> None:
    assert len(indexed) == len(_segments(land))


def test_an_empty_coastline_puts_everything_offshore() -> None:
    empty = build_coastline(np.zeros((0, 2, 2), np.float32))
    signed = empty.signed_distance(np.zeros(2, np.float32), np.zeros(2, np.float32))
    assert np.isinf(signed).all() and (signed > 0).all()


# ---------------------------------------------------------------------------
# Through the layer
# ---------------------------------------------------------------------------


def test_layer_writes_the_coastline_coordinate(
    tmp_path: Path, land: shapely.Polygon
) -> None:
    path = tmp_path / "coastline.wkb.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(shapely.to_wkb(land))

    layer = CoastlineLayer(CoastlineConfig(coastline=path), land, Terminal())
    # `make_grid` puts every point at one location. Move it well inland.
    inland_x, inland_y = land.representative_point().coords[0]
    grid = make_grid()
    grid[Coordinate.X] = xr.full_like(grid.x, inland_x)
    grid[Coordinate.Y] = xr.full_like(grid.y, inland_y)

    layer(grid)

    distance = grid[Coordinate.COASTLINE]
    assert distance.shape == grid.x.isel({Coordinate.K: 0}).shape
    assert (distance < 0).all(), "an inland point is negative"
