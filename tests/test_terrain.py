"""Tests for the hybrid terrain-following coordinate and its decay profiles.

The profile tests are pure NumPy.  The builder tests use synthetic surfaces
written to zarr, so nothing reads a real data file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pyproj import CRS

from nzcvm.config.grids.emod3d import EMOD3DGrid
from nzcvm.config.grids.model import Model
from nzcvm.config.grids.sw4 import MeshRefinement, SW4GridConfig
from nzcvm.config.grids.terrain import (
    Decay,
    LinearDecay,
    SleveDecay,
    Solver,
    SquashedDecay,
    TaperedDecay,
)
from nzcvm.coordinates import Coordinate
from nzcvm.formats.sfile import _solver_z
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.grids.terrain import decay_weight, terrain_z_depth
from nzcvm.models.mesh import StructuredMeshSchema

_MODEL = Model(
    origin_lon=172.0, origin_lat=-41.0, azimuth=30.0, crs=CRS.from_epsg(2193)
)


def _write_surface(path: Path, z) -> Path:
    """Write a surface over all of New Zealand with ``z = z(x, y)``, positive down."""
    n = 200
    xs = np.linspace(1_000_000.0, 3_000_000.0, n, dtype=np.float32)
    ys = np.linspace(4_700_000.0, 6_700_000.0, n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = np.broadcast_to(np.asarray(z(xx, yy), dtype=np.float32), xx.shape).copy()
    StructuredMeshSchema.new(
        x=xx, y=yy, z=zz, i=np.arange(n), j=np.arange(n), name="surface"
    ).to_zarr(path, mode="w")
    return path


@pytest.fixture(scope="module")
def bumpy_surface(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Peaks up to 2500 m above sea level and seafloor down to 1500 m."""
    return _write_surface(
        tmp_path_factory.mktemp("surfaces") / "bumpy.zarr",
        lambda x, y: -500 + 2000 * np.sin(x / 3e3) * np.cos(y / 5e3),
    )


@pytest.fixture(scope="module")
def deep_seafloor(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_surface(
        tmp_path_factory.mktemp("surfaces") / "deep.zarr", lambda x, y: 1000.0
    )


# ---------------------------------------------------------------------------
# Decay profiles
# ---------------------------------------------------------------------------

_LEVELS = np.linspace(0.0, 5000.0, 501)
_SURFACE = np.array([-2500.0, -300.0, 0.0, 400.0])[:, np.newaxis]

_RESOLVED = [
    SquashedDecay(),
    LinearDecay(length=5000.0),
    TaperedDecay(ratio=1.0),
    TaperedDecay(ratio=2.5),
    SleveDecay(scale=1000.0, length=5000.0),
]
_IDS = ["squashed", "linear", "tapered", "tapered-2.5", "sleve"]


class TestDecayWeight:
    @pytest.mark.parametrize("decay", _RESOLVED, ids=_IDS)
    def test_surface_level_follows_topography(self, decay: Decay) -> None:
        weight = np.broadcast_to(decay_weight(decay, _LEVELS, _SURFACE), (4, 501))
        # Tapered doesn't decay offshore.  Every column still starts at 1.
        assert weight[:, 0] == pytest.approx(1.0)

    @pytest.mark.parametrize("decay", _RESOLVED, ids=_IDS)
    def test_weight_never_increases_with_depth(self, decay: Decay) -> None:
        weight = np.broadcast_to(decay_weight(decay, _LEVELS, _SURFACE), (4, 501))
        assert np.all(np.diff(weight, axis=1) <= 1e-12)
        assert np.all((weight >= 0) & (weight <= 1))

    @pytest.mark.parametrize(
        "decay",
        [LinearDecay(length=3000.0), SleveDecay(scale=500.0, length=3000.0)],
        ids=["linear", "sleve"],
    )
    def test_flat_below_length(self, decay: Decay) -> None:
        weight = decay_weight(decay, _LEVELS, _SURFACE)
        assert np.all(weight[..., _LEVELS >= 3000.0] == 0)

    def test_tapered_length_is_ratio_times_elevation(self) -> None:
        decay = TaperedDecay(ratio=2.0)
        surface = np.array([[-1000.0]])
        weight = decay_weight(decay, np.array([0.0, 1000.0, 2000.0, 3000.0]), surface)
        assert weight.ravel() == pytest.approx([1.0, 0.5, 0.0, 0.0])

    def test_sleve_approaches_linear_for_large_scale(self) -> None:
        sleve = decay_weight(SleveDecay(scale=1e6, length=5000.0), _LEVELS, _SURFACE)
        linear = decay_weight(LinearDecay(length=5000.0), _LEVELS, _SURFACE)
        assert sleve == pytest.approx(np.broadcast_to(linear, sleve.shape), abs=1e-4)

    def test_sleve_does_not_overflow_for_short_scales(self) -> None:
        weight = decay_weight(SleveDecay(scale=1.0, length=5000.0), _LEVELS, _SURFACE)
        assert np.all(np.isfinite(weight))
        assert weight[..., 0] == pytest.approx(1.0)


class TestTerrainZ:
    """The unified formula reproduces each grid's original construction."""

    def test_linear_is_sw4_stretch(self) -> None:
        bottom = 5000.0
        z, _ = terrain_z_depth(_SURFACE, _LEVELS, LinearDecay(length=bottom))
        zeta = _LEVELS / bottom
        assert z == pytest.approx(_SURFACE * (1 - zeta) + bottom * zeta)

    def test_squashed_is_shift(self) -> None:
        z, depth = terrain_z_depth(_SURFACE, _LEVELS, SquashedDecay())
        assert z == pytest.approx(_SURFACE + _LEVELS)
        assert np.all(depth == _LEVELS), "squashed depth must be exactly nominal"

    def test_tapered_matches_velocity_modelling(self) -> None:
        """``velocity_modelling.velocity3d`` SQUASHED_TAPERED, in its z-up form."""
        elevation = -_SURFACE
        d = _LEVELS
        taper = np.ones(np.broadcast_shapes(elevation.shape, d.shape))
        above = np.broadcast_to(elevation > 0, taper.shape)
        taper[above] = (1.0 - d / elevation)[above]
        taper = np.clip(taper, 0.0, None)
        z_up = elevation * taper - d

        z, _ = terrain_z_depth(_SURFACE, _LEVELS, TaperedDecay(ratio=1.0))
        assert z == pytest.approx(-z_up)

    @pytest.mark.parametrize("decay", _RESOLVED, ids=_IDS)
    def test_flat_levels_are_exactly_nominal(self, decay: Decay) -> None:
        levels = np.array([6000.0, 7000.0], dtype=np.float32)
        z, depth = terrain_z_depth(_SURFACE.astype(np.float32), levels, decay)
        weight = np.broadcast_to(decay_weight(decay, levels, _SURFACE), z.shape)
        flat = weight == 0
        assert np.all(z[flat] == np.broadcast_to(levels, z.shape)[flat])
        assert depth.dtype == np.float32


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _sw4(
    surface: Path, decay: Decay | None = None, **refinements: MeshRefinement
) -> SW4GridConfig:
    refinements = refinements or {
        "a": MeshRefinement(resolution=100.0, bottom=3000.0),
        "b": MeshRefinement(resolution=200.0, bottom=6000.0),
        "c": MeshRefinement(resolution=400.0, bottom=10000.0),
    }
    return SW4GridConfig(
        surface=surface,
        extent_x=8000.0,
        extent_y=6000.0,
        orientation=_MODEL,
        refinements=refinements,
        chunks={Coordinate.I: 16, Coordinate.J: 16},
        decay=decay,
    )


_SW4_DECAYS = pytest.mark.parametrize(
    "decay",
    # Seafloor reaches 1500 m, so SLEVE needs s * tanh(L / s) deeper than that.
    [
        None,
        LinearDecay(),
        LinearDecay(length=2000.0),
        SquashedDecay(),
        TaperedDecay(ratio=1.0),
        SleveDecay(scale=2000.0),
    ],
    ids=["none", "linear", "linear-2000", "squashed", "tapered", "sleve"],
)
_NOMINAL_DECAYS = pytest.mark.parametrize(
    "decay",
    [LinearDecay(), SquashedDecay(), TaperedDecay(ratio=1.0), SleveDecay(scale=2000.0)],
    ids=["linear", "squashed", "tapered", "sleve"],
)


class TestSW4Decays:
    @_SW4_DECAYS
    def test_seams_conform(self, bumpy_surface: Path, decay: Decay | None) -> None:
        grids = build_grids_from_config(_sw4(bumpy_surface, decay))
        for upper, lower in [("a", "b"), ("b", "c")]:
            bottom = grids[upper].z.isel(k=-1)
            top = grids[lower].z.isel(k=0)
            # The coarse block uses every second fine node.
            assert np.all(bottom.sel(i=top.i, j=top.j).values == top.values)

    @_SW4_DECAYS
    def test_top_follows_surface(
        self, bumpy_surface: Path, decay: Decay | None
    ) -> None:
        grid = build_grids_from_config(_sw4(bumpy_surface, decay))["a"]
        assert np.all(grid.depth.isel(k=0).values == 0)
        assert np.all(np.diff(grid.z.values, axis=-1) > 0)

    @_SW4_DECAYS
    def test_solver_is_physical_only_without_a_decay(
        self, bumpy_surface: Path, decay: Decay | None
    ) -> None:
        config = _sw4(bumpy_surface, decay)
        expected = Solver.PHYSICAL if decay is None else Solver.NOMINAL
        assert config.solver == expected
        grids = build_grids_from_config(config)
        assert all(grid.attrs["solver"] == expected for grid in grids.values())

    def test_physical_grid_is_the_sw4_stretch(self, bumpy_surface: Path) -> None:
        grids = build_grids_from_config(_sw4(bumpy_surface))
        assert np.all(grids["a"].z.isel(k=-1).values == 3000.0)
        assert np.all(grids["c"].z.isel(k=-1).values == 10000.0)
        # Sized from the thickest column, as SW4 needs.
        spacing = np.diff(grids["a"].z.values, axis=-1)
        assert spacing.max() == pytest.approx(100.0, rel=0.05)

    @_NOMINAL_DECAYS
    def test_nominal_grid_is_sized_at_the_resolution(
        self, bumpy_surface: Path, decay: Decay
    ) -> None:
        grids = build_grids_from_config(_sw4(bumpy_surface, decay))
        for grid, top, bottom, resolution in [
            (grids["a"], 0.0, 3000.0, 100.0),
            (grids["b"], 3000.0, 6000.0, 200.0),
        ]:
            nominal = grid.nominal_depth.values
            assert nominal[0] == top and nominal[-1] == bottom
            assert np.diff(nominal) == pytest.approx(resolution)

    def test_squashed_bottom_follows_the_topography(self, bumpy_surface: Path) -> None:
        grid = build_grids_from_config(_sw4(bumpy_surface, SquashedDecay()))["c"]
        assert np.all(grid.depth.isel(k=-1).values == 10000.0)

    def test_every_block_keeps_the_configured_chunks(self, bumpy_surface: Path) -> None:
        for grid in build_grids_from_config(_sw4(bumpy_surface)).values():
            for var in ("x", "y", "z", "depth"):
                chunks = grid[var].chunksizes
                assert chunks["i"][0] == 16 and chunks["j"][0] == 16
                assert len(chunks["k"]) == 1

    def test_fold_over_is_rejected(self, deep_seafloor: Path) -> None:
        # dz/dA is about 1 - S/s at the surface, negative for S = 1000, s = 200.
        with pytest.raises(ValueError, match="folds over"):
            build_grids_from_config(_sw4(deep_seafloor, SleveDecay(scale=200.0)))


class TestSolverZ:
    def test_physical_solver_uses_z(self, bumpy_surface: Path) -> None:
        grid = build_grids_from_config(_sw4(bumpy_surface))["a"]
        assert _solver_z(grid).identical(grid.z)

    def test_nominal_solver_uses_nominal_depth(self, bumpy_surface: Path) -> None:
        grid = build_grids_from_config(_sw4(bumpy_surface, LinearDecay()))["a"]
        z = _solver_z(grid)
        assert z.dims == grid.z.dims
        assert z.chunks == grid.z.chunks
        assert np.all(z.values == grid.nominal_depth.values)


def _emod3d(surface: Path, **kwargs) -> EMOD3DGrid:
    return EMOD3DGrid(
        surface=surface,
        nx=40,
        ny=30,
        nz=50,
        resolution=100.0,
        orientation=_MODEL,
        chunks={Coordinate.I: 16, Coordinate.J: 16},
        **kwargs,
    )


class TestEMOD3DDecays:
    def test_emod3d_grids_are_nominal(self, bumpy_surface: Path) -> None:
        grid = build_grids_from_config(_emod3d(bumpy_surface, decay=SquashedDecay()))
        assert grid["grid_0"].attrs["solver"] == Solver.NOMINAL

    def test_sleve_decays_over_the_whole_grid(self, bumpy_surface: Path) -> None:
        grid = build_grids_from_config(
            _emod3d(bumpy_surface, decay=SleveDecay(scale=1000.0))
        )["grid_0"]
        assert np.all(grid.z.isel(k=-1).values == 4900.0)
        assert np.all(grid.depth.isel(k=0).values == 0)


class TestDecayConfig:
    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ({"type": "squashed"}, SquashedDecay()),
            ({"type": "linear"}, LinearDecay()),
            ({"type": "linear", "length": 2000.0}, LinearDecay(length=2000.0)),
            ({"type": "tapered", "ratio": 1.5}, TaperedDecay(ratio=1.5)),
            ({"type": "sleve", "scale": 800.0}, SleveDecay(scale=800.0)),
        ],
    )
    def test_discriminated_by_type(self, data: dict, expected: Decay) -> None:
        assert Decay.from_dict(data) == expected

    def test_negative_length_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            LinearDecay(length=-1.0)
