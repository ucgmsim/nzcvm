"""Tests for the `VelocityModel` container in `nzcvm.velocity_model`."""

from pathlib import Path

import numpy as np
import pytest
import shapely
from pyproj import CRS

from nzcvm.config.grids.emod3d import EMOD3DGrid
from nzcvm.config.grids.model import Model
from nzcvm.config.grids.terrain import SquashedDecay
from nzcvm.config.metadata import ModelMetadata
from nzcvm.config.velocity_model import VelocityModelConfig
from nzcvm.coordinates import Coordinate
from nzcvm.grids.grid import Grid, GridSchema
from nzcvm.models.mesh import surface_mesh
from nzcvm.qualities import Qualities, QualitiesSchema
from nzcvm.velocity_model import VelocityModel

SHAPE = (2, 3, 4)


def _ramp(shape: tuple[int, int, int] = SHAPE, offset: float = 0.0) -> np.ndarray:
    """An array whose value at (i, j, k) is 100 i + 10 j + k + offset."""
    ii, jj, kk = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    return (100 * ii + 10 * jj + kk + offset).astype(np.float32)


def _grid(
    name: str,
    geometry: shapely.Geometry = shapely.box(170.0, -43.0, 171.0, -42.0),
    shape: tuple[int, int, int] = SHAPE,
) -> Grid:
    ramp = _ramp(shape)
    return GridSchema.new(
        x=ramp,
        y=ramp + 1000.0,
        z=ramp + 2000.0,
        depth=ramp + 3000.0,
        nominal_depth=np.arange(shape[2], dtype=np.float32) * 10.0,
        name=name,
        resolution=100.0,
        solver="physical",
        geometry=geometry,
        origin_lon=np.float32(170.0),
        origin_lat=np.float32(-43.0),
        azimuth=np.float32(0.0),
        grid_azimuth=np.float32(0.0),
        bottom_left_lon=np.float32(170.0),
        bottom_left_lat=np.float32(-43.0),
    )


def _qualities(shape: tuple[int, int, int] = SHAPE, offset: float = 0.0) -> Qualities:
    ramp = _ramp(shape, offset)
    return QualitiesSchema.new(
        rho=ramp + 2000.0,
        vp=ramp + 3000.0,
        vs=ramp + 1500.0,
        qp=ramp + 100.0,
        qs=ramp + 50.0,
        alpha=np.zeros(shape, dtype=np.float32),
    )


@pytest.fixture
def model() -> VelocityModel:
    return VelocityModel(
        grids={"a": _grid("a"), "b": _grid("b", shape=(3, 2, 5))},
        qualities={"a": _qualities(), "b": _qualities((3, 2, 5), offset=7.0)},
        metadata=ModelMetadata(title="test"),
    )


# ---------------------------------------------------------------------------
# Construction invariant
# ---------------------------------------------------------------------------


def test_qualities_must_cover_the_same_grids() -> None:
    with pytest.raises(
        ValueError,
        match=r"qualities \(a, c\) does not match number of grids \(a, b\)",
    ):
        VelocityModel(
            grids={"b": _grid("b"), "a": _grid("a")},
            qualities={"c": _qualities(), "a": _qualities()},
            metadata=ModelMetadata(),
        )


def test_a_model_without_qualities_is_valid() -> None:
    model = VelocityModel(grids={"a": _grid("a")}, metadata=ModelMetadata())
    assert model.qualities == {}


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------


def test_geometry_is_the_union_of_the_grid_outlines() -> None:
    model = VelocityModel(
        grids={
            "west": _grid("west", shapely.box(170.0, -43.0, 171.0, -42.0)),
            "east": _grid("east", shapely.box(170.5, -43.0, 172.0, -42.0)),
        },
        metadata=ModelMetadata(),
    )
    assert model.geometry.equals(shapely.box(170.0, -43.0, 172.0, -42.0))


def test_pairwise_matches_each_grid_with_its_qualities(model: VelocityModel) -> None:
    pairs = model.pairwise
    assert list(pairs) == ["a", "b"]
    for name, (grid, qualities) in pairs.items():
        assert grid is model.grids[name]
        assert qualities is model.qualities[name]


def test_pairwise_needs_qualities() -> None:
    model = VelocityModel(grids={"a": _grid("a")}, metadata=ModelMetadata())
    with pytest.raises(ValueError, match="assigned qualities are empty"):
        _ = model.pairwise


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coordinate", [Coordinate.I, Coordinate.J, Coordinate.K])
def test_flip_reverses_every_dataset_along_one_axis(
    model: VelocityModel, coordinate: Coordinate
) -> None:
    flipped = model.flip(coordinate)
    axis = "ijk".index(coordinate.value)

    assert flipped.metadata == model.metadata
    for name in ("a", "b"):
        for original, result, var in (
            (model.grids[name], flipped.grids[name], "x"),
            (model.qualities[name], flipped.qualities[name], "vs"),
        ):
            # Same shape: every point is kept, none dropped.
            assert result[var].shape == original[var].shape
            np.testing.assert_array_equal(
                result[var].values, np.flip(original[var].values, axis=axis)
            )


def test_flip_leaves_the_original_model_untouched(model: VelocityModel) -> None:
    before = model.grids["a"].x.values.copy()
    flipped = model.flip(Coordinate.K)
    np.testing.assert_array_equal(model.grids["a"].x.values, before)
    np.testing.assert_array_equal(
        flipped.flip(Coordinate.K).grids["a"].x.values, before
    )
    # The nominal depth coordinate follows the k axis.
    np.testing.assert_array_equal(
        flipped.grids["a"].nominal_depth.values, [30.0, 20.0, 10.0, 0.0]
    )


def test_orient_transposes_grids_and_qualities(model: VelocityModel) -> None:
    oriented = model.orient(Coordinate.K, Coordinate.I, Coordinate.J)
    assert oriented.grids["a"].x.dims == ("k", "i", "j")
    assert oriented.qualities["b"].vs.dims == ("k", "i", "j")
    np.testing.assert_array_equal(
        oriented.grids["a"].x.values,
        np.transpose(model.grids["a"].x.values, (2, 0, 1)),
    )


def test_datatree_round_trip(model: VelocityModel) -> None:
    restored = VelocityModel.from_datatree(model.to_datatree())
    assert restored.metadata == model.metadata
    assert set(restored.grids) == {"a", "b"}
    np.testing.assert_array_equal(
        restored.qualities["b"].vs.values, model.qualities["b"].vs.values
    )


# ---------------------------------------------------------------------------
# Building from a configuration
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_surface(tmp_path: Path) -> Path:
    """A flat z=0 surface wide enough to cover the configured grid."""
    path = tmp_path / "flat.zarr"
    n = 8
    xs = np.linspace(1_000_000.0, 3_000_000.0, n, dtype=np.float32)
    ys = np.linspace(4_700_000.0, 6_700_000.0, n, dtype=np.float32)
    surface_mesh("flat", np.zeros((n, n), dtype=np.float32), xs, ys).to_zarr(
        path, mode="w"
    )
    return path


def test_from_config_builds_the_grids_and_keeps_the_metadata(
    flat_surface: Path,
) -> None:
    metadata = ModelMetadata(title="Configured", version="1.2")
    config = VelocityModelConfig(
        grid=EMOD3DGrid(
            surface=flat_surface,
            nx=4,
            ny=6,
            nz=8,
            resolution=1000.0,
            orientation=Model(
                origin_lon=172.0,
                origin_lat=-41.0,
                azimuth=0.0,
                crs=CRS.from_epsg(2193),
            ),
            decay=SquashedDecay(),
            chunks={Coordinate.I: 4, Coordinate.J: 6},
        ),
        metadata=metadata,
    )

    model = VelocityModel.from_config(config)

    assert model.metadata == metadata
    assert model.qualities == {}
    assert len(model.grids) == 1
    grid = next(iter(model.grids.values()))
    assert dict(grid.sizes) == {"i": 4, "j": 6, "k": 8}
    assert model.geometry.equals(grid.geometry)
