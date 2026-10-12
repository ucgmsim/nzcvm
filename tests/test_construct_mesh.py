"""Tests for the basin meshes ``nzcvm basin main`` writes.

The volumetric tests build on a hand-made triangulation of a square, so
they don't depend on the triangulation; ``surface_samples`` and
``triangulate_polygon`` have tests of their own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely
import typer
import xarray as xr
from hypothesis import assume, given
from hypothesis import strategies as st

from nzcvm.models.mesh import (
    DEPTH_PROFILE_VARIABLES,
    ModelType,
    TetrahedralMeshSchema,
    mesh_encoding,
)
from nzcvm.models.model import MeshModel
from nzcvm.models.surface import SurfaceGrid
from nzcvm.scripts import construct_mesh
from nzcvm.scripts.construct_mesh import (
    PAD_DEPTH,
    SAMPLES_PER_CELL,
    Triangulation,
    profile_mesh,
    surface_samples,
    triangulate_polygon,
)

SIDE = 1000.0
# Elevation of the four corners, as depth (positive down): the topography
# slopes so the profile's reference surface isn't flat.
TOPOGRAPHY = np.array([0.0, -40.0, -20.0, -60.0])
THICKNESS = 500.0

PROFILE = pd.DataFrame(
    {
        "z": [0.0, 100.0, 300.0],
        "thickness": [100.0, 200.0, 1e6],
        "rho": [1800.0, 2000.0, 2200.0],
        "vp": [1500.0, 2000.0, 2500.0],
        "vs": [400.0, 800.0, 1200.0],
        "qp": [40.0, 80.0, 120.0],
        "qs": [20.0, 40.0, 60.0],
    }
)


def _square() -> Triangulation:
    """Triangulate a square with its diagonal from corner 1 to corner 2."""
    vertices = np.array([[0.0, 0.0], [SIDE, 0.0], [0.0, SIDE], [SIDE, SIDE]])
    return Triangulation(vertices, np.array([[0, 1, 2], [1, 3, 2]]))


def _topography(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The planar interpolant of TOPOGRAPHY over the triangulation."""
    u, v = x / SIDE, y / SIDE
    lower = u + v <= 1.0
    t0, t1, t2, t3 = TOPOGRAPHY
    return np.where(
        lower,
        t0 + (t1 - t0) * u + (t2 - t0) * v,
        t3 + (t2 - t3) * (1.0 - u) + (t1 - t3) * (1.0 - v),
    )


def _topography_at(x: float, y: float) -> float:
    return float(_topography(np.array(x), np.array(y)))


def _mesh(
    pad_top: bool = False,
    bottom: np.ndarray | None = None,
    model: pd.DataFrame = PROFILE,
):
    return profile_mesh(
        "square",
        shapely.box(0.0, 0.0, SIDE, SIDE),
        _square(),
        TOPOGRAPHY.copy(),
        TOPOGRAPHY + THICKNESS if bottom is None else bottom,
        model,
        culling_volume=1e-3,
        pad_top=pad_top,
        priority=3,
        topography_range=(0.0, 0.0),
    )


def _expected_vs(depth: np.ndarray) -> np.ndarray:
    tops = PROFILE["z"].to_numpy()
    return PROFILE["vs"].to_numpy()[np.searchsorted(tops[1:], depth, side="right")]


def test_one_layer_of_depth_profile_tetrahedra() -> None:
    mesh = _mesh()
    # Each triangle gives three tetrahedra however many profile layers there are.
    assert mesh.sizes["j"] == 6
    assert (mesh.model_type.values == ModelType.DEPTH_PROFILE).all()
    assert (mesh.priority.values == 3).all()
    assert mesh.has_depth_profile


def test_vertex_depth_is_its_z() -> None:
    """A basin mesh is in depth below the topography, so depth is z."""
    mesh = _mesh()
    np.testing.assert_allclose(mesh.depth.values, mesh.z.values)


# Avoid the layer boundaries, where rounding picks either layer.
interior = st.floats(min_value=0.01 * SIDE, max_value=0.99 * SIDE)
depths = st.floats(min_value=1.0, max_value=THICKNESS - 1.0).filter(
    lambda d: abs(d - 100.0) > 0.5 and abs(d - 300.0) > 0.5
)


@given(x=interior, y=interior, depth=depths)
def test_query_evaluates_profile_at_depth(x: float, y: float, depth: float) -> None:
    """A basin's z is depth below the topography. The profile reads it."""
    z = _topography_at(x, y) + depth
    # Avoid the layer boundaries, where rounding picks either layer.
    assume(abs(z - 100.0) > 0.5 and abs(z - 300.0) > 0.5)
    model = MeshModel.from_mesh(_mesh())
    quality = model.query(x, y, z)
    assert quality is not None
    assert quality.vs == pytest.approx(_expected_vs(np.array(z)))
    assert quality.alpha == pytest.approx(1.0)


def test_pad_top_extends_first_layer_upwards() -> None:
    model = MeshModel.from_mesh(_mesh(pad_top=True))
    x = y = 0.3 * SIDE
    above = _topography_at(x, y) - 0.5 * PAD_DEPTH
    quality = model.query(x, y, above)
    assert quality is not None
    assert quality.vs == PROFILE["vs"].iloc[0]
    assert MeshModel.from_mesh(_mesh(pad_top=False)).query(x, y, above) is None


def test_zero_thickness_column_is_culled() -> None:
    bottom = TOPOGRAPHY + THICKNESS
    # Corner 4 (index 3) belongs only to the upper triangle.
    bottom[3] = TOPOGRAPHY[3]
    mesh = _mesh(bottom=bottom)
    # Its prism keeps the two tetrahedra with one vertex on the pinched column.
    assert mesh.sizes["j"] == 5
    TetrahedralMeshSchema.from_dataset(mesh)


def test_profile_mesh_round_trips_through_zarr(tmp_path: Path) -> None:
    path = tmp_path / "square.zarr"
    mesh = _mesh()
    mesh.to_zarr(path, encoding=mesh_encoding(mesh))
    reloaded = TetrahedralMeshSchema.from_dataset(xr.open_zarr(path))
    assert reloaded.has_depth_profile
    np.testing.assert_array_equal(reloaded.profile_vs.values, PROFILE["vs"])

    model = MeshModel.from_path(path)
    x = y = 0.25 * SIDE
    z = _topography_at(x, y) + 200.0
    quality = model.query(x, y, z)
    assert quality is not None
    assert quality.vs == 800.0


def test_partial_profile_is_rejected() -> None:
    incomplete = _mesh().drop_vars("profile_vs")
    with pytest.raises(TypeError):
        TetrahedralMeshSchema.from_dataset(incomplete)


def test_depth_profile_cells_need_a_profile() -> None:
    mesh = _mesh().drop_vars([*DEPTH_PROFILE_VARIABLES, "layer"])
    with pytest.raises(ValueError, match="has no profile"):
        MeshModel.from_mesh(TetrahedralMeshSchema.from_dataset(mesh))


def test_uniform_model_is_a_one_layer_profile() -> None:
    model = construct_mesh.uniform_model(2000.0, 2500.0, 1000.0, 100.0, 50.0)
    quality = MeshModel.from_mesh(_mesh(model=model)).query(100.0, 100.0, 300.0)
    assert quality is not None
    assert quality.vs == 1000.0


def test_uniform_model_with_layer_bottoms_matches_the_file_layout(
    tmp_path: Path,
) -> None:
    """Generated layers match the same layers read from a 1-D model file."""
    bottoms = [50.0, 100.0, 200.0]
    path = tmp_path / "uniform.fd_modfile"
    path.write_text(
        "DEF HST\n"
        + "".join(f"1.8 2.0 1.81 38 19 {bottom / 1000.0}\n" for bottom in bottoms)
    )
    generated = construct_mesh.uniform_model(
        1810.0, 1800.0, 2000.0, 38.0, 19.0, layer_bottoms=bottoms
    )
    pd.testing.assert_frame_equal(
        generated, construct_mesh.read_layered_model(path), check_dtype=False
    )


@pytest.mark.parametrize("bottoms", [[], [100.0, 50.0], [50.0, 50.0]])
def test_uniform_model_rejects_bad_layer_bottoms(bottoms: list[float]) -> None:
    with pytest.raises(ValueError, match="increasing"):
        construct_mesh.uniform_model(1.0, 1.0, 1.0, 1.0, 1.0, layer_bottoms=bottoms)


def test_layered_mesh_round_trips_through_zarr(tmp_path: Path) -> None:
    """``--layered`` meshes have no depth profile and still write and query."""
    layers = construct_mesh.slice_with_model(
        _square(),
        TOPOGRAPHY.copy(),
        TOPOGRAPHY + THICKNESS,
        PROFILE,
        culling_volume=1e-3,
        pad_top=False,
    )
    mesh = construct_mesh.construct_volumetric_mesh(
        "layered",
        shapely.box(0.0, 0.0, SIDE, SIDE),
        layers,
        priority=3,
        topography_range=(0.0, 0.0),
    )
    # One slice of three tetrahedra per triangle per profile layer.
    assert mesh.sizes["j"] == 2 * 3 * len(PROFILE)
    path = tmp_path / "layered.zarr"
    mesh.to_zarr(path, encoding=mesh_encoding(mesh))
    reloaded = TetrahedralMeshSchema.from_dataset(xr.open_zarr(path))
    assert not reloaded.has_depth_profile

    x = y = 0.25 * SIDE
    z = _topography_at(x, y) + 200.0
    quality = MeshModel.from_path(path).query(x, y, z)
    assert quality is not None
    assert quality.vs == pytest.approx(800.0)


def test_basin_stops_at_the_bottom_of_the_model() -> None:
    shallow = PROFILE.assign(thickness=[100.0, 200.0, 50.0])
    mesh = _mesh(model=shallow)
    # The last layer ends 350 m below the topography, above the 500 m basement.
    np.testing.assert_allclose(mesh.depth.values.max(), 350.0, atol=1e-3)


def test_zero_thickness_layers_are_dropped() -> None:
    repeated = pd.concat([PROFILE.iloc[:2], PROFILE.iloc[1:]], ignore_index=True)
    mesh = _mesh(model=repeated)
    np.testing.assert_array_equal(mesh.profile_top.values, PROFILE["z"])


def test_decreasing_layer_tops_are_rejected() -> None:
    with pytest.raises(ValueError, match="increase"):
        _mesh(model=PROFILE.assign(z=[0.0, 300.0, 100.0]))


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------


def _grid(f, n: int = 21) -> SurfaceGrid:
    """A surface over a grid slightly larger than the square."""
    axis = np.linspace(-0.1 * SIDE, 1.1 * SIDE, n)
    x, y = np.meshgrid(axis, axis)
    return SurfaceGrid(x, y, f(x, y))


SQUARE = shapely.box(0.0, 0.0, SIDE, SIDE)


def _bump(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 300.0 * np.exp(-((x - 400.0) ** 2 + (y - 600.0) ** 2) / 1e5)


def _interpolate(
    triangulation: Triangulation, grid: SurfaceGrid, points: np.ndarray
) -> np.ndarray:
    """The triangulation's linear interpolant of `grid` at `points`."""
    corners = triangulation.vertices[triangulation.triangles]  # (T, 3, 2)
    corner_values = grid(corners[..., 0].ravel(), corners[..., 1].ravel())[0]
    corner_values = corner_values.reshape(-1, 3)
    # Barycentric coordinates of every point in every triangle.
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    basis = np.stack((b - a, c - a), axis=2)  # (T, 2, 2)
    offset = points[:, None, :] - a[None]  # (N, T, 2)
    lb, lc = np.einsum("tij,ntj->nti", np.linalg.inv(basis), offset).transpose(2, 0, 1)
    weights = np.stack((1 - lb - lc, lb, lc), axis=2)  # (N, T, 3)
    inside = (weights >= -1e-9).all(axis=2)
    triangle = inside.argmax(axis=1)
    assert inside.any(axis=1).all()
    rows = np.arange(len(points))
    return (weights[rows, triangle] * corner_values[triangle]).sum(axis=1)


def test_samples_refine_the_grid_inside_the_outline() -> None:
    grid = _grid(_bump)
    points, spacing = surface_samples(grid, SQUARE, per_cell=2)
    assert shapely.contains_xy(SQUARE, points[:, 0], points[:, 1]).all()
    # The lattice is twice as fine as the 60 m grid, and covers the square.
    assert spacing == pytest.approx(30.0)
    assert np.diff(np.unique(points[:, 0])).max() == pytest.approx(30.0)
    assert len(points) == len(np.unique(points[:, 0])) * len(np.unique(points[:, 1]))


def test_plane_needs_only_the_outline() -> None:
    plane = _grid(lambda x, y: 0.3 * x - 0.2 * y + 7.0)
    triangulation = triangulate_polygon(SQUARE, [(plane, 1.0)])
    assert len(triangulation.triangles) == 2


@pytest.mark.parametrize("tolerance", [1.0, 10.0])
def test_triangulation_fits_samples_within_tolerance(tolerance: float) -> None:
    grid = _grid(_bump)
    triangulation = triangulate_polygon(SQUARE, [(grid, tolerance)])
    points, spacing = surface_samples(grid, SQUARE, SAMPLES_PER_CELL)
    # The outline too, at the lattice spacing.
    outline = shapely.segmentize(SQUARE.exterior, spacing)
    points = np.r_[points, np.asarray(outline.coords)[:-1]]
    values, _ = grid(points[:, 0], points[:, 1])
    error = np.abs(_interpolate(triangulation, grid, points) - values)
    assert error.max() <= tolerance + 1e-6
    # A looser tolerance needs fewer triangles than the samples.
    assert len(triangulation.triangles) < len(points)


def test_max_edge_caps_triangle_size() -> None:
    flat = _grid(lambda x, y: np.zeros_like(x))
    triangulation = triangulate_polygon(SQUARE, [(flat, 1.0)], max_edge=300.0)
    corners = triangulation.vertices[triangulation.triangles]
    edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)
    assert edges.max() <= 300.0 + 1e-6


def test_depth_below_measures_from_the_topography() -> None:
    x, y = np.meshgrid(np.arange(4.0), np.arange(3.0), indexing="ij")
    topography = SurfaceGrid(x, y, -10.0 - x)  # rising to the east
    basement = SurfaceGrid(x, y, np.full_like(x, 200.0))
    depth = construct_mesh.depth_below(basement, topography)
    np.testing.assert_allclose(depth.values, 210.0 + x)


# ---------------------------------------------------------------------------
# Stacked layers sharing one triangulation
# ---------------------------------------------------------------------------


def _stack_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """A synthetic DEM, basin outline and basement, plus a horizon halfway down."""
    import h5py
    from typer.testing import CliRunner

    from nzcvm.scripts.synthetic import app as synthetic_app

    runner = CliRunner()
    for args in (
        ["dem", str(tmp_path / "dem.h5"), "--samples", "24"],
        ["basin", "gully", str(tmp_path), "--samples", "24"],
    ):
        result = runner.invoke(synthetic_app, args)
        assert result.exit_code == 0, result.output
    dem_path = tmp_path / "dem.h5"
    basement_path = tmp_path / "gully_basement.h5"
    with h5py.File(basement_path) as f:
        longitude, latitude = f["longitude"][:], f["latitude"][:]
        basement = f["elevation"][:]
    dem = construct_mesh.read_surface_window(dem_path)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    x, y = construct_mesh.TRANSFORMER.transform(lon_grid, lat_grid)
    ground = -dem(x.ravel(), y.ravel(), clamp=True)[0].reshape(x.shape)
    middle_path = tmp_path / "middle.h5"
    with h5py.File(middle_path, "w") as f:
        f["longitude"], f["latitude"] = longitude, latitude
        f["elevation"] = (ground + basement) / 2
    return tmp_path / "gully_outline.geojson", dem_path, middle_path, basement_path


def _build(*args: str | Path) -> None:
    from typer.testing import CliRunner

    result = CliRunner().invoke(construct_mesh.app, [str(arg) for arg in args])
    assert result.exit_code == 0, result.output


def _columns(mesh: xr.Dataset, reduce) -> pd.Series:
    """The top or bottom (by *reduce*) depth of each vertex column, by (x, y)."""
    frame = pd.DataFrame({"x": mesh.x.values, "y": mesh.y.values, "z": mesh.z.values})
    return frame.groupby(["x", "y"]).z.agg(reduce)


def test_stacked_layers_share_their_interface(tmp_path: Path) -> None:
    _assert_stack_meets(tmp_path, *_stack_inputs(tmp_path))


def test_stacked_layers_meet_over_bases_at_different_spacings(
    tmp_path: Path,
) -> None:
    """Every layer averages a Zarr DEM to the stack's spacing, not its own base's.

    The basement here has half the horizon's resolution, so each layer's own
    basement would average the DEM differently and convert the shared horizon
    to different depths.
    """
    import h5py
    from typer.testing import CliRunner

    from nzcvm.scripts.surface_cli import app as surface_app

    outline, dem_h5, middle, basement = _stack_inputs(tmp_path)
    coarse = tmp_path / "coarse_basement.h5"
    with h5py.File(basement) as source, h5py.File(coarse, "w") as f:
        f["longitude"] = source["longitude"][::2]
        f["latitude"] = source["latitude"][::2]
        f["elevation"] = source["elevation"][::2, ::2]
    fine = construct_mesh.median_spacing(construct_mesh.read_surface_window(middle))
    dem = tmp_path / "dem.zarr"
    result = CliRunner().invoke(
        surface_app, [str(dem_h5), str(dem), "--spacing", str(fine / 4)]
    )
    assert result.exit_code == 0, result.output
    _assert_stack_meets(tmp_path, outline, dem, middle, coarse)


def _assert_stack_meets(
    tmp_path: Path, outline: Path, dem: Path, middle: Path, basement: Path
) -> None:
    """Build the two layers of a stack in one run and check they meet."""
    uniform = "rho=2000,vp=2000,vs=800,qp=80,qs=40"
    upper, lower = tmp_path / "upper.zarr", tmp_path / "lower.zarr"
    _build(
        "main",
        outline,
        dem,
        dem,
        middle,
        basement,
        *("--output", upper, "--output", lower),
        *("--priority", "1", "--priority", "2"),
        *("--model", uniform, "--model", uniform),
    )

    upper_bottom = _columns(xr.open_zarr(upper), "max")
    lower_top = _columns(xr.open_zarr(lower), "min")
    shared = upper_bottom.index.intersection(lower_top.index)
    assert len(shared) > 10
    # A padded lower layer would start PAD_DEPTH above the upper one's bottom.
    np.testing.assert_array_equal(upper_bottom[shared], lower_top[shared])


def test_each_layer_needs_its_own_options(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    outline, dem, middle, basement = _stack_inputs(tmp_path)
    result = CliRunner().invoke(
        construct_mesh.app,
        [
            "main",
            *map(str, (outline, dem, dem, middle, basement)),
            *("--output", str(tmp_path / "only.zarr")),
            *("--priority", "1"),
            *("--model", "rho=2000,vp=2000,vs=800,qp=80,qs=40"),
        ],
    )
    assert result.exit_code != 0
    assert "2 layer(s)" in result.output


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("rho=1,vp=2,vs=3,qp=4", "exactly"),
        ("no/such/model.fd_modfile", "No 1-D model file"),
    ],
)
def test_read_model_rejects_bad_specs(spec: str, message: str) -> None:
    with pytest.raises(typer.BadParameter, match=message):
        construct_mesh.read_model(spec)


def test_read_model_reads_uniform_qualities() -> None:
    model = construct_mesh.read_model("rho=1,vp=2,vs=3,qp=4,qs=5")
    assert model[["rho", "vp", "vs", "qp", "qs"]].iloc[0].tolist() == [1, 2, 3, 4, 5]


def test_depth_mesh_bounds_span_its_depths_over_the_topography() -> None:
    """A depth mesh's bounds hold the elevation range it can occupy."""
    mesh = profile_mesh(
        "square",
        shapely.box(0.0, 0.0, SIDE, SIDE),
        _square(),
        TOPOGRAPHY.copy(),
        TOPOGRAPHY + THICKNESS,
        PROFILE,
        culling_volume=1e-3,
        pad_top=False,
        priority=3,
        topography_range=(-300.0, 20.0),
    )
    min_x, min_y, min_z, max_x, max_y, max_z = mesh.attrs["aabb"]
    assert (min_x, min_y, max_x, max_y) == pytest.approx((0.0, 0.0, SIDE, SIDE))
    margin = construct_mesh.ELEVATION_MARGIN
    assert min_z == pytest.approx(-300.0 + float(mesh.z.min()) - margin)
    assert max_z == pytest.approx(20.0 + float(mesh.z.max()) + margin)


def test_layer_bottoms_only_split_a_uniform_model(tmp_path: Path) -> None:
    model_file = tmp_path / "model.fd_modfile"
    model_file.write_text("header\n1.8 0.5 1.9 50 25 0.1\n")
    with pytest.raises(typer.BadParameter, match="uniform"):
        construct_mesh.read_model(str(model_file), layer_bottoms=[50.0, 100.0])
    layers = construct_mesh.read_model(
        "rho=1810,vp=1800,vs=2000,qp=38,qs=19", layer_bottoms=[50.0, 100.0]
    )
    assert layers["bottom_depth"].tolist() == [50.0, 100.0]
