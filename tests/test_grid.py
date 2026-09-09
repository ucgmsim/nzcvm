"""Tests for EMOD3D and SW4 grid builders.

Every test here builds on a synthetic flat surface, z=0 everywhere, so nothing
reads a real data file.
"""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from pyproj import CRS

from nzcvm.config.grids.emod3d import EMOD3DGrid, TopographyType
from nzcvm.config.grids.model import Model
from nzcvm.config.grids.sw4 import MeshRefinement, SW4GridConfig
from nzcvm.coordinates import Coordinate
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.grids.grid import Grid
from nzcvm.models.mesh import StructuredMeshSchema

# ---------------------------------------------------------------------------
# Shared fixture: flat surface file
# ---------------------------------------------------------------------------

# NZ-wide bounding box in NZTM2000 (approx): roughly 1e6 m x 1.7e6 m
_SURFACE_XMIN = 1_000_000.0
_SURFACE_YMIN = 4_700_000.0
_SURFACE_EXTENT = 2_000_000.0  # 2000 km side, wide enough to cover every test grid


def _write_flat_surface(path: Path) -> None:
    """Write a flat z=0 StructuredMesh zarr surface to *path*."""
    n = 8  # 8x8 grid of points, enough for interpolation
    xs = np.linspace(
        _SURFACE_XMIN, _SURFACE_XMIN + _SURFACE_EXTENT, n, dtype=np.float32
    )
    ys = np.linspace(
        _SURFACE_YMIN, _SURFACE_YMIN + _SURFACE_EXTENT, n, dtype=np.float32
    )
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = np.zeros_like(xx)
    mesh = StructuredMeshSchema.new(
        x=xx, y=yy, z=zz, i=np.arange(n), j=np.arange(n), name="flat"
    )
    mesh.to_zarr(path, mode="w")


@pytest.fixture(scope="module")
def flat_surface(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("surfaces") / "flat.zarr"
    _write_flat_surface(p)
    return p


# ---------------------------------------------------------------------------
# Shared Model orientation (centred over New Zealand)
# ---------------------------------------------------------------------------

_NZTM = CRS.from_epsg(2193)
_ORIGIN_LON = 172.0
_ORIGIN_LAT = -41.0


def _model(azimuth: float = 0.0) -> Model:
    return Model(
        origin_lon=_ORIGIN_LON,
        origin_lat=_ORIGIN_LAT,
        azimuth=azimuth,
        crs=_NZTM,
    )


# ---------------------------------------------------------------------------
# EMOD3D grid builder
# ---------------------------------------------------------------------------


def _emod3d_config(
    flat_surface: Path,
    *,
    nx: int = 4,
    ny: int = 6,
    nz: int = 8,
    resolution: float = 1000.0,
    azimuth: float = 0.0,
    topo_type: TopographyType = TopographyType.SQUASHED,
    chunks: dict | None = None,
) -> EMOD3DGrid:
    if chunks is None:
        chunks = {Coordinate.I: nx, Coordinate.J: ny}
    return EMOD3DGrid(
        surface=flat_surface,
        nx=nx,
        ny=ny,
        nz=nz,
        resolution=resolution,
        orientation=_model(azimuth),
        topo_type=topo_type,
        chunks=chunks,
    )


def _first_grid_of(config) -> Grid:
    grids = build_grids_from_config(config)
    return next(iter(grids.values()))


def _sw4_config(
    flat_surface: Path,
    *,
    extent_x: float = 4000.0,
    extent_y: float = 4000.0,
    azimuth: float = 0.0,
    refinements: dict[str, MeshRefinement] | None = None,
) -> SW4GridConfig:
    if refinements is None:
        refinements = {
            "top": MeshRefinement(resolution=1000.0, bottom=2000.0),
        }
    return SW4GridConfig(
        surface=flat_surface,
        extent_x=extent_x,
        extent_y=extent_y,
        orientation=_model(azimuth),
        refinements=refinements,
        chunks={Coordinate.I: 8, Coordinate.J: 8},
    )


_BUILDERS = pytest.mark.parametrize(
    "make_config", [_emod3d_config, _sw4_config], ids=["emod3d", "sw4"]
)


class TestGridShape:
    @_BUILDERS
    def test_coordinate_order_is_ijk(self, flat_surface: Path, make_config) -> None:
        grid = _first_grid_of(make_config(flat_surface))
        assert grid.x.dims == ("i", "j", "k")

    @_BUILDERS
    def test_all_coordinates_are_dask(self, flat_surface: Path, make_config) -> None:
        grid = _first_grid_of(make_config(flat_surface))
        for var in ("x", "y", "z", "depth"):
            assert isinstance(grid[var].data, da.Array), f"{var} is not dask"


class TestEMOD3DShape:
    def test_all_dims_chunked_according_to_config(self, flat_surface: Path) -> None:
        """After ensure_chunks, i/j follow the config chunking and k stays one chunk."""
        nx, ny, nz = 8, 8, 8
        ci, cj = 4, 4
        chunks = {Coordinate.I: ci, Coordinate.J: cj}
        grid = _first_grid_of(
            _emod3d_config(flat_surface, nx=nx, ny=ny, nz=nz, chunks=chunks)
        )
        for var in ("x", "y", "z", "depth"):
            csizes = grid[var].chunksizes
            assert csizes["i"] == (ci, ci), f"{var}: i not chunked as {ci}×{ci}"
            assert csizes["j"] == (cj, cj), f"{var}: j not chunked as {cj}×{cj}"
            assert csizes["k"] == (nz,), f"{var}: k must be a single chunk"


class TestEMOD3DCentreRegistration:
    """EMOD3D is cell-centred in x/y with no offset in z: depth[0] == 0."""

    def test_first_depth_is_zero(self, flat_surface: Path) -> None:
        grid = _first_grid_of(_emod3d_config(flat_surface))
        depth_0 = float(grid.depth.isel(i=0, j=0, k=0).compute())
        assert depth_0 == pytest.approx(0.0, abs=1e-4)

    @_BUILDERS
    def test_depth_increases_with_k(self, flat_surface: Path, make_config) -> None:
        grid = _first_grid_of(make_config(flat_surface))
        depths = grid.depth.isel(i=0, j=0).compute().values
        assert np.all(np.diff(depths) > 0), "depth must increase monotonically in k"

    def test_depth_spacing_matches_resolution(self, flat_surface: Path) -> None:
        res = 500.0
        grid = _first_grid_of(_emod3d_config(flat_surface, resolution=res))
        depths = grid.depth.isel(i=0, j=0).compute().values.astype(np.float64)
        diffs = np.diff(depths)
        assert list(diffs) == pytest.approx([res] * len(diffs), rel=1e-4)


class TestEMOD3DRotation:
    def test_zero_azimuth_x_increases_along_i(self, flat_surface: Path) -> None:
        """With azimuth=0 and origin at NZ centre, x should increase with i."""
        grid = _first_grid_of(_emod3d_config(flat_surface, azimuth=0.0))
        x_slice = grid.x.isel(j=0, k=0).compute().values
        assert x_slice[-1] > x_slice[0], "x must increase along i for azimuth=0"


class TestEMOD3DMetadata:
    def test_resolution_stored_as_attribute(self, flat_surface: Path) -> None:
        cfg = _emod3d_config(flat_surface, resolution=250.0)
        grid = _first_grid_of(cfg)
        assert grid.attrs["resolution"] == 250.0

    def test_origin_lon_lat_stored(self, flat_surface: Path) -> None:
        grid = _first_grid_of(_emod3d_config(flat_surface))
        assert "origin_lon" in grid.attrs
        assert "origin_lat" in grid.attrs
        assert grid.attrs["origin_lon"] == pytest.approx(_ORIGIN_LON)
        assert grid.attrs["origin_lat"] == pytest.approx(_ORIGIN_LAT)


def _i_axis(grid: Grid) -> np.ndarray:
    """The physical vector spanned by the i-axis at j=k=0."""
    x = grid.x.isel(j=0, k=0).compute().values.astype(np.float64)
    y = grid.y.isel(j=0, k=0).compute().values.astype(np.float64)
    return np.array([x[-1] - x[0], y[-1] - y[0]])


class TestGridRotation:
    """Azimuth must rotate the i-axis by exactly -azimuth, preserving length.

    The comparison is against the azimuth=0 grid rather than an absolute
    bearing: ``grid_azimuth`` differs from ``azimuth`` by the meridian
    convergence (``config/grids/model.py``), which is constant for a fixed
    origin but depends on the pyproj version.  The *difference* between two
    azimuths is exact, so a pyproj upgrade can't break the comparison, which
    still pins the sign.  An assertion that the profiles merely differ would pass
    for a rotation in the wrong direction, which is what this replaces.
    """

    @pytest.mark.parametrize("azimuth", [0, 30, 90, 180, 270])
    @_BUILDERS
    def test_i_axis_rotates_by_negative_azimuth(
        self, flat_surface: Path, make_config, azimuth: float
    ) -> None:
        v0 = _i_axis(_first_grid_of(make_config(flat_surface, azimuth=0.0)))
        va = _i_axis(_first_grid_of(make_config(flat_surface, azimuth=azimuth)))

        bearing_0 = np.degrees(np.arctan2(v0[1], v0[0]))
        bearing_a = np.degrees(np.arctan2(va[1], va[0]))
        # The rotation matrix comes from -grid_azimuth, so increasing the
        # azimuth turns the i-axis clockwise.
        assert (bearing_a - bearing_0) % 360 == pytest.approx(
            (-azimuth) % 360, abs=0.1
        ), f"i-axis rotated the wrong way or by the wrong amount for {azimuth=}"

    @pytest.mark.parametrize("azimuth", [0, 30, 90, 180, 270])
    @_BUILDERS
    def test_rotation_preserves_i_axis_length(
        self, flat_surface: Path, make_config, azimuth: float
    ) -> None:
        v0 = _i_axis(_first_grid_of(make_config(flat_surface, azimuth=0.0)))
        va = _i_axis(_first_grid_of(make_config(flat_surface, azimuth=azimuth)))
        assert np.linalg.norm(va) == pytest.approx(np.linalg.norm(v0), rel=1e-4)


# SW4 grid tests
# ---------------------------------------------------------------------------


class TestSW4Shape:
    def test_single_refinement_produces_one_grid(self, flat_surface: Path) -> None:
        grids = build_grids_from_config(_sw4_config(flat_surface))
        assert len(grids) == 1

    def test_two_refinements_produce_two_grids(self, flat_surface: Path) -> None:
        cfg = _sw4_config(
            flat_surface,
            extent_x=4000.0,
            extent_y=4000.0,
            refinements={
                "top": MeshRefinement(resolution=500.0, bottom=2000.0),
                "bottom": MeshRefinement(resolution=1000.0, bottom=4000.0),
            },
        )
        grids = build_grids_from_config(cfg)
        assert len(grids) == 2

    def test_ni_matches_extent_over_resolution(self, flat_surface: Path) -> None:
        """ni == round(extent_x / resolution) + 1 for the finest refinement."""
        extent_x = 4000.0
        res = 1000.0
        cfg = _sw4_config(
            flat_surface,
            extent_x=extent_x,
            refinements={"top": MeshRefinement(resolution=res, bottom=2000.0)},
        )
        grids = build_grids_from_config(cfg)
        grid = grids["top"]
        expected_ni = round(extent_x / res) + 1
        assert grid.x.shape[0] == expected_ni


class TestSW4CornerRegistration:
    """SW4 is corner-registered: the first horizontal point sits at the boundary."""

    def test_depth_at_surface_is_zero_or_negative(self, flat_surface: Path) -> None:
        """Top k slice depth should be >= 0 (at or shallower than surface = 0 for flat topo)."""
        grids = build_grids_from_config(_sw4_config(flat_surface))
        grid = next(iter(grids.values()))
        top_depth = float(grid.depth.isel(i=0, j=0, k=0).compute())
        assert top_depth == pytest.approx(0.0, abs=1e-4)


class TestSW4RefinementSeams:
    """Bottom of layer n must equal top of layer n+1 (no gaps)."""

    def test_no_gap_between_refinements(self, flat_surface: Path) -> None:
        top_bottom = 2000.0
        cfg = _sw4_config(
            flat_surface,
            # Shallowest refinement is finest, and the deeper layer 2x coarser.
            refinements={
                "top": MeshRefinement(resolution=1000.0, bottom=top_bottom),
                "bot": MeshRefinement(resolution=2000.0, bottom=4000.0),
            },
        )
        grids = build_grids_from_config(cfg)
        top_grid = grids["top"]
        bot_grid = grids["bot"]

        top_z_bottom = float(top_grid.z.isel(i=0, j=0, k=-1).compute())
        bot_z_top = float(bot_grid.z.isel(i=0, j=0, k=0).compute())

        assert top_z_bottom == pytest.approx(bot_z_top, abs=1.0)

    def test_refinement_resolution_respected(self, flat_surface: Path) -> None:
        """Vertical spacing in each refinement layer matches its resolution."""
        res = 1000.0
        cfg = _sw4_config(
            flat_surface,
            refinements={"top": MeshRefinement(resolution=res, bottom=3000.0)},
        )
        grids = build_grids_from_config(cfg)
        grid = grids["top"]
        depths = grid.z.isel(i=0, j=0).compute().values.astype(np.float64)
        diffs = np.abs(np.diff(depths))
        assert list(diffs) == pytest.approx([res] * len(diffs), rel=1e-3)
