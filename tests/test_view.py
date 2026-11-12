"""Tests for the ``nzcvm view`` commands and their drawing helpers.

The tests mock the plotter, so these tests don't need a display.
"""

import gzip
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import shapely
import xarray as xr
from typer.testing import CliRunner

from nzcvm.config.metadata import ModelMetadata
from nzcvm.formats import datatree
from nzcvm.models.mesh import surface_mesh
from nzcvm.models.surface import Surface
from nzcvm.scripts import view
from nzcvm.velocity_model import VelocityModel
from tests.test_sfile import _grid, _qualities

pv = pytest.importorskip("pyvista")

runner = CliRunner()


@pytest.fixture
def zarr_model(tmp_path: Path) -> Path:
    path = tmp_path / "model.zarr"
    model = VelocityModel(
        grids={"g0": _grid("g0", 100.0, 0.0)},
        qualities={"g0": _qualities()},
        metadata=ModelMetadata(),
    )
    datatree.to_zarr(model, path)
    return path


@pytest.fixture
def plotter(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    plotter = MagicMock()
    monkeypatch.setattr(pv, "Plotter", MagicMock(return_value=plotter))
    return plotter


def _invoke(*args: str) -> None:
    result = runner.invoke(
        view.app,
        ["model", *args, "--no-axes", "--slice-mode", "none", "--stride", "1"],
    )
    assert result.exit_code == 0, result.output


def test_missing_pyvista_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch):
    # A None entry in sys.modules makes ``import pyvista`` raise ImportError.
    monkeypatch.setitem(sys.modules, "pyvista", None)
    with pytest.raises(ImportError, match=r"pip install nzcvm\[visualization\]"):
        view._require_pyvista()


def test_zero_colour_limits_are_respected(zarr_model: Path, plotter: MagicMock):
    # The model's vs is well above 0, so a dropped --min-val 0 would show.
    _invoke(str(zarr_model), "--min-val", "0", "--max-val", "0")
    assert plotter.add_mesh.call_args.kwargs["clim"] == [0.0, 0.0]


def test_colour_limits_default_to_data_range(zarr_model: Path, plotter: MagicMock):
    _invoke(str(zarr_model))
    lo, hi = plotter.add_mesh.call_args.kwargs["clim"]
    assert lo > 0
    assert hi > lo


def test_log_diff_on_dask_grids():
    # The test grids are dask-backed, like grids read from disk.
    grid, qualities = _grid("g0", 100.0, 0.0), _qualities()
    other = _qualities(offset=100.0)
    mesh = view._build_structured_grid(
        0, grid, qualities, "vs", 1, grid, other, view.DiffMode.LOG
    )
    expected = np.log10(qualities.vs.values) - np.log10(other.vs.values)
    np.testing.assert_allclose(mesh.point_data["vs"], expected.ravel(order="F"))


def test_stride_averages_blocks_of_nodes():
    # A 2 x 3 x 4 grid coarsened by 2 keeps one (i, j) block, trimming j = 2,
    # and leaves k alone.
    grid, qualities = _grid("g0", 100.0, 0.0), _qualities()
    mesh = view._build_structured_grid(3, grid, qualities, "vp", 2)

    k = np.arange(4.0)
    assert mesh.dimensions == (1, 1, 4)
    # x = 1000 i + 10 j + k averages to 500 + 5 + k over i, j in {0, 1}.
    np.testing.assert_allclose(mesh.points[:, 0], 505.0 + k)
    np.testing.assert_allclose(mesh.point_data["vp"], 3000 + 50 + 5 + k)
    np.testing.assert_array_equal(mesh.point_data["layer"], 3)
    assert mesh.active_scalars_name == "vp"


def test_log_diff_from_cli(zarr_model: Path, plotter: MagicMock):
    _invoke(str(zarr_model), "--compare-to", str(zarr_model), "--diff-mode", "log")
    lo, hi = plotter.add_mesh.call_args.kwargs["clim"]
    assert lo == hi == 0.0


# ---------------------------------------------------------------------------
# add_logical_axes
# ---------------------------------------------------------------------------


def _points(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> xr.Dataset:
    """A dataset of node coordinates indexed ``(i, j, k)``."""
    dims = ("i", "j", "k")
    return xr.Dataset({"x": (dims, x), "y": (dims, y), "z": (dims, z)})


@pytest.fixture
def arrows(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Record the arrows drawn, in place of :class:`pyvista.Arrow`."""
    arrow = MagicMock(side_effect=lambda **kwargs: kwargs)
    monkeypatch.setattr(pv, "Arrow", arrow)
    return arrow


def test_logical_axes_point_along_i_j_k_from_the_origin(arrows: MagicMock):
    # Nodes: x = 1000 i + 10 j + k, y = 2000 i + 20 j + k, z = 50 + 100 i + 10 j + k
    grid = _grid("g0", 100.0, 50.0)
    pl = MagicMock()
    # The bounds' diagonal is |(30, 40, 120)| = 130, so the scale is 19.5.
    view.add_logical_axes(pl, grid, (10.0, 40.0, 20.0, 60.0, 5.0, 125.0))

    p0 = np.array([0.0, 0.0, 50.0])
    expected = [
        ("i", np.array([1000.0, 2000.0, 0.0]), "red"),  # i and j stay horizontal
        ("j", np.array([10.0, 20.0, 0.0]), "green"),
        ("k", np.array([1.0, 1.0, 1.0]), "blue"),
    ]
    assert arrows.call_count == 3
    assert pl.add_mesh.call_count == pl.add_point_labels.call_count == 3
    for arrow, mesh_call, label_call, (label, vec, color) in zip(
        arrows.call_args_list,
        pl.add_mesh.call_args_list,
        pl.add_point_labels.call_args_list,
        expected,
        strict=True,
    ):
        direction = vec / np.linalg.norm(vec)
        np.testing.assert_allclose(arrow.kwargs["start"], p0)
        np.testing.assert_allclose(arrow.kwargs["direction"], direction, rtol=1e-6)
        assert arrow.kwargs["scale"] == pytest.approx(19.5)
        assert mesh_call.args[0] == arrow.kwargs
        assert mesh_call.kwargs["color"] == color

        (position,), (text,) = label_call.args
        assert text == label
        assert label_call.kwargs["text_color"] == color
        np.testing.assert_allclose(position, p0 + direction * 19.5 * 1.1, rtol=1e-6)


def test_logical_axes_skip_a_direction_with_no_horizontal_extent(arrows: MagicMock):
    # i only moves z, which the i and j arrows drop, so i has no direction.
    i, j, k = np.meshgrid(np.arange(2.0), np.arange(2.0), np.arange(2.0), indexing="ij")
    grid = _points(100 * j, 100 * k, 10 * i + 5 * k)
    pl = MagicMock()
    view.add_logical_axes(pl, grid, (0.0, 3.0, 0.0, 4.0, 0.0, 0.0))

    labels = [c.args[1] for c in pl.add_point_labels.call_args_list]
    assert labels == [["j"], ["k"]]
    j_arrow, k_arrow = arrows.call_args_list
    np.testing.assert_allclose(j_arrow.kwargs["direction"], [1, 0, 0])
    np.testing.assert_allclose(
        k_arrow.kwargs["direction"], np.array([0, 100, 5]) / np.hypot(100, 5)
    )
    assert j_arrow.kwargs["scale"] == pytest.approx(0.75)


def test_logical_axes_warn_without_drawing_on_a_single_node_grid(
    arrows: MagicMock, capsys: pytest.CaptureFixture[str]
):
    one = np.zeros((1, 1, 1))
    pl = MagicMock()
    view.add_logical_axes(pl, _points(one, one, one), (0.0, 1.0, 0.0, 1.0, 0.0, 1.0))

    assert "Could not calculate logical axes" in capsys.readouterr().out
    arrows.assert_not_called()
    pl.add_mesh.assert_not_called()
    pl.add_point_labels.assert_not_called()


# ---------------------------------------------------------------------------
# add_coastline_underlay
# ---------------------------------------------------------------------------


def _write_coastline(path: Path, geometry: shapely.Geometry) -> Path:
    with gzip.open(path, "wb") as handle:
        handle.write(shapely.to_wkb(geometry))
    return path


def test_coastline_rings_and_lines_become_polylines_above_the_meshes(
    tmp_path: Path,
):
    island = shapely.Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(2, 2), (4, 2), (4, 4), (2, 4)]]
    )
    river = shapely.LineString([(20, 0), (21, 1), (22, 0)])
    streams = shapely.MultiLineString([[(30, 0), (31, 0)], [(40, 0), (41, 1)]])
    collection = shapely.GeometryCollection(
        [island, shapely.Point(5, 5), shapely.LineString(), river, streams]
    )
    path = _write_coastline(tmp_path / "coast.wkb.gz", collection)
    pl = MagicMock()
    view.add_coastline_underlay(pl, (0.0, 50.0, 0.0, 50.0, 200.0, 900.0), path)

    pl.add_mesh.assert_called_once()
    poly = pl.add_mesh.call_args.args[0]
    assert pl.add_mesh.call_args.kwargs["name"] == "coastline"
    expected_xy = np.vstack(
        [
            np.array(island.exterior.coords),
            np.array(island.interiors[0].coords),
            np.array(river.coords),
            np.array(streams.geoms[0].coords),
            np.array(streams.geoms[1].coords),
        ]
    )
    np.testing.assert_allclose(poly.points[:, :2], expected_xy)
    # Drawn 500 m above the shallowest mesh point.
    np.testing.assert_allclose(poly.points[:, 2], -300.0)
    np.testing.assert_array_equal(
        poly.lines,
        [5, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 3, 10, 11, 12, 2, 13, 14, 2, 15, 16],
    )


def test_coastline_with_no_lines_warns_and_draws_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    points = shapely.MultiPoint([(0, 0), (1, 1)])
    path = _write_coastline(tmp_path / "coast.wkb.gz", points)
    pl = MagicMock()
    view.add_coastline_underlay(pl, (0.0, 1.0, 0.0, 1.0, 0.0, 1.0), path)

    pl.add_mesh.assert_not_called()
    assert "No valid geometries" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# dem_window
# ---------------------------------------------------------------------------


def _dem() -> Surface:
    """z = x' + 2 y' on a 10 m grid, with frame (x', y') = NZTM - (1000, 2000)."""
    nodes = 10.0 * np.arange(11)
    xx, yy = np.meshgrid(nodes, nodes)
    return Surface.from_dataset(
        surface_mesh(
            "dem", (xx + 2 * yy).astype(np.float32), 1000.0 + nodes, 2000.0 + nodes
        )
    )


def test_dem_window_reads_the_padded_footprint():
    # Footprint x in [1020, 1060], y in [2030, 2070]; a 10% margin widens it
    # by 4 m, which takes in no extra 10 m nodes.
    surface = view.dem_window(_dem(), (1020.0, 1060.0, 2030.0, 2070.0, 0.0, 1.0))

    xs, ys = 1000.0 + 10 * np.arange(2, 7), 2000.0 + 10 * np.arange(3, 8)
    np.testing.assert_allclose(np.unique(surface.points[:, 0]), xs)
    np.testing.assert_allclose(np.unique(surface.points[:, 1]), ys)
    assert surface.n_points == 25
    # Each node keeps the DEM's z at that node.
    z = (surface.points[:, 0] - 1000) + 2 * (surface.points[:, 1] - 2000)
    np.testing.assert_allclose(surface.points[:, 2], z)
    np.testing.assert_allclose(surface.point_data["elevation"], -z)


def test_dem_window_margin_widens_the_window():
    # A 50% margin pads by 20 m on each side.
    surface = view.dem_window(_dem(), (1020.0, 1060.0, 2030.0, 2070.0), margin=0.5)
    np.testing.assert_allclose(
        np.unique(surface.points[:, 0]), 1000 + 10.0 * np.arange(0, 9)
    )
    np.testing.assert_allclose(
        np.unique(surface.points[:, 1]), 2000 + 10.0 * np.arange(1, 10)
    )


def test_dem_window_strides_down_to_the_node_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(view, "DEM_MAX_NODES", 2)
    surface = view.dem_window(_dem(), (1020.0, 1060.0, 2030.0, 2070.0), margin=0.0)
    # Five nodes along each axis need a stride of ceil(5 / 2) = 3.
    np.testing.assert_allclose(np.unique(surface.points[:, 0]), [1020.0, 1050.0])
    np.testing.assert_allclose(np.unique(surface.points[:, 1]), [2030.0, 2060.0])


# ---------------------------------------------------------------------------
# model command
# ---------------------------------------------------------------------------


def _run_model(*args: str):
    return runner.invoke(view.app, ["model", *args, "--no-axes", "--stride", "1"])


@pytest.mark.parametrize(
    ("axis", "column", "hi"),
    # g0 spans x in [0, 1023], y in [0, 2043] and z in [0, 123].
    [("x", 0, 1023.0), ("y", 1, 2043.0), ("Z", 2, 123.0)],
)
def test_static_slice_spans_the_grids_along_the_axis(
    zarr_model: Path, plotter: MagicMock, axis: str, column: int, hi: float
):
    result = _run_model(str(zarr_model), "--slice-axis", axis)
    assert result.exit_code == 0, result.output

    slider = plotter.add_slider_widget.call_args.kwargs
    assert slider["rng"] == [0.0, pytest.approx(hi)]
    assert slider["value"] == pytest.approx(hi / 2)
    assert slider["title"] == f"{axis.upper()} Slice Location"

    plotter.add_mesh.assert_called_once()
    sliced = plotter.add_mesh.call_args.args[0]
    assert plotter.add_mesh.call_args.kwargs["name"] == "slice_g0"
    assert sliced.n_points > 0
    np.testing.assert_allclose(sliced.points[:, column], hi / 2, rtol=1e-5)

    # The slider moves the slice.
    slider["callback"](hi / 4)
    moved = plotter.add_mesh.call_args.args[0]
    assert moved.n_points > 0
    np.testing.assert_allclose(moved.points[:, column], hi / 4, rtol=1e-5)


def test_invalid_slice_axis_exits(zarr_model: Path, plotter: MagicMock):
    result = _run_model(str(zarr_model), "--slice-axis", "w")
    assert result.exit_code == 1
    assert "Invalid axis 'w'" in result.output
    plotter.show.assert_not_called()


def test_orthogonal_slices_are_the_default(zarr_model: Path, plotter: MagicMock):
    result = _run_model(str(zarr_model))
    assert result.exit_code == 0, result.output
    plotter.add_mesh_slice_orthogonal.assert_called_once()
    assert plotter.add_mesh_slice_orthogonal.call_args.kwargs["scalars"] == "vs"
    plotter.add_mesh.assert_not_called()


def test_abs_diff_uses_a_diverging_colormap(zarr_model: Path, plotter: MagicMock):
    _invoke(str(zarr_model), "--compare-to", str(zarr_model), "--diff-mode", "abs")
    kwargs = plotter.add_mesh.call_args.kwargs
    assert kwargs["cmap"] == "bwr"
    assert kwargs["clim"] == [0.0, 0.0]
    assert plotter.add_scalar_bar.call_args.kwargs["title"] == "▲ vs"


def test_plain_view_keeps_the_chosen_colormap(zarr_model: Path, plotter: MagicMock):
    _invoke(str(zarr_model), "--cmap", "magma")
    assert plotter.add_mesh.call_args.kwargs["cmap"] == "magma"
    assert plotter.add_mesh.call_args.kwargs["label"] == "g0"
    assert plotter.add_scalar_bar.call_args.kwargs["title"] == "vs"


def test_model_draws_axes_and_coastline(
    zarr_model: Path, plotter: MagicMock, tmp_path: Path
):
    coast = _write_coastline(
        tmp_path / "coast.wkb.gz", shapely.MultiLineString([[(0, 0), (1000, 2000)]])
    )
    args = ["model", str(zarr_model), "--slice-mode", "none", "--stride", "1"]
    result = runner.invoke(view.app, [*args, "--coastline", str(coast)])
    assert result.exit_code == 0, result.output

    names = [c.kwargs.get("name") for c in plotter.add_mesh.call_args_list]
    assert names.count("coastline") == 1
    labels = [c.args[1] for c in plotter.add_point_labels.call_args_list]
    assert labels == [["i"], ["j"], ["k"], ["g0"]]


def test_model_saves_a_screenshot(zarr_model: Path, plotter: MagicMock, tmp_path: Path):
    shot = tmp_path / "shot.png"
    _invoke(str(zarr_model), "--screenshot", str(shot))
    plotter.show.assert_called_once_with(auto_close=False)
    plotter.screenshot.assert_called_once_with(str(shot))


def test_model_exits_when_no_grid_builds(
    zarr_model: Path, plotter: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    def fail(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(view, "_build_structured_grid", fail)
    result = _run_model(str(zarr_model))
    assert result.exit_code == 1
    assert "Failed to build grid for 'g0': boom" in result.output
    assert "No grids could be built" in result.output
    plotter.show.assert_not_called()


def test_point_picking_reports_values_inside_a_grid(
    zarr_model: Path, plotter: MagicMock
):
    _invoke(str(zarr_model))
    pick = plotter.enable_point_picking.call_args.kwargs["callback"]

    pick(None)
    pick(np.array([]))
    plotter.add_text.assert_not_called()

    # The node at i = j = k = 0 of g0, where vs = 1500.
    pick(np.array([0.0, 0.0, 0.0]))
    text = plotter.add_text.call_args.args[0]
    assert text.startswith("Picked Point:\nX: 0.000\nY: 0.000\nZ: 0.000")
    assert "vs: 1.5e+03" in text
    assert "vtkValidPointMask" not in text

    pick(np.array([1e7, 1e7, 1e7]))
    assert plotter.add_text.call_args.args[0].endswith(
        "Point clicked outside of all grids."
    )


# ---------------------------------------------------------------------------
# basin command
# ---------------------------------------------------------------------------


def _tetra_mesh(path: Path, offset: float, vertical: str = "elevation") -> Path:
    """A single tetrahedron with every coordinate shifted by *offset*."""
    corners = (
        np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], dtype=np.float32)
        + offset
    )
    values = np.arange(4, dtype=np.float32)
    properties = ("rho", "vp", "vs", "qp", "qs")
    dataset = xr.Dataset(
        {
            "x": ("i", corners[:, 0]),
            "y": ("i", corners[:, 1]),
            "z": ("i", corners[:, 2]),
            "connectivity": (("j", "k"), np.array([[0, 1, 2, 3]], dtype=np.uint64)),
            "priority": ("j", np.zeros(1, dtype=np.uint8)),
            "model_type": ("j", np.zeros(1, dtype=np.uint8)),
            **{name: ("i", values + n) for n, name in enumerate(properties)},
            "alpha": ("i", np.ones(4, dtype=np.float32)),
        },
        coords={
            "i": np.arange(4, dtype=np.uint64),
            "j": np.arange(1, dtype=np.uint64),
            "k": np.arange(4, dtype=np.uint64),
        },
        attrs={
            "name": path.stem,
            "vertical": vertical,
            "aabb": [offset] * 3 + [offset + 100] * 3,
        },
    )
    dataset.to_netcdf(path)
    return path


def _points_of(call) -> np.ndarray:
    return np.asarray(call.args[0].points)


def test_basin_colours_meshes_and_spans_the_pole(tmp_path: Path, plotter: MagicMock):
    a = _tetra_mesh(tmp_path / "a.nc", 0.0)
    b = _tetra_mesh(tmp_path / "b.nc", 1000.0)
    args = ["basin", str(a), str(b), "--x-pos", "5", "--y-pos", "6"]
    result = runner.invoke(view.app, args)
    assert result.exit_code == 0, result.output

    calls = plotter.add_mesh.call_args_list
    assert [c.kwargs["color"] for c in calls] == ["red", "blue", "cyan"]
    np.testing.assert_allclose(_points_of(calls[1]).min(axis=0), [1000.0] * 3)
    # The pole runs 5 km past the meshes' combined z range, [0, 1100].
    np.testing.assert_allclose(
        _points_of(calls[2]), [[5.0, 6.0, -5000.0], [5.0, 6.0, 6100.0]]
    )
    plotter.enable_depth_peeling.assert_not_called()
    assert plotter.camera.up == (0.0, 0.0, -1.0)
    plotter.show.assert_called_once()


def test_basin_needs_both_pole_coordinates(tmp_path: Path, plotter: MagicMock):
    a = _tetra_mesh(tmp_path / "a.nc", 0.0)
    result = runner.invoke(view.app, ["basin", str(a), "--x-pos", "5"])
    assert result.exit_code == 0, result.output
    assert [c.kwargs.get("color") for c in plotter.add_mesh.call_args_list] == ["red"]


def test_basin_scalar_colours_by_property(tmp_path: Path, plotter: MagicMock):
    a = _tetra_mesh(tmp_path / "a.nc", 0.0)
    b = _tetra_mesh(tmp_path / "b.nc", 1000.0)
    result = runner.invoke(view.app, ["basin", str(a), str(b), "--scalar", "vp"])
    assert result.exit_code == 0, result.output

    plotter.enable_depth_peeling.assert_called_once()
    calls = plotter.add_mesh.call_args_list
    # Each mesh, plus one invisible copy of the first for the scalar bar.
    assert [(c.kwargs["opacity"], c.kwargs["show_scalar_bar"]) for c in calls] == [
        ("alpha", False),
        (0.0, True),
        ("alpha", False),
    ]
    mesh = calls[0].args[0]
    assert mesh.point_data.active_scalars_name == "vp"
    np.testing.assert_allclose(mesh.point_data["vp"], [1, 2, 3, 4])
    np.testing.assert_allclose(mesh.point_data["alpha"], 1.0)


def test_basin_hangs_depth_meshes_from_the_dem(tmp_path: Path, plotter: MagicMock):
    # A flat DEM 300 m up: z = -300, positive down.
    nodes = -500.0 + 200.0 * np.arange(11)
    dem_path = tmp_path / "dem.zarr"
    flat = np.full((11, 11), -300.0, dtype=np.float32)
    surface_mesh("dem", flat, nodes, nodes).to_zarr(dem_path)
    depth = _tetra_mesh(tmp_path / "depth.nc", 0.0, vertical="depth")
    elevation = _tetra_mesh(tmp_path / "elevation.nc", 0.0)
    coast = _write_coastline(
        tmp_path / "coast.wkb.gz", shapely.MultiLineString([[(0, 0), (50, 50)]])
    )
    topo = tmp_path / "topo.vtk"
    pv.Sphere().save(topo)

    result = runner.invoke(
        view.app,
        [
            "basin",
            str(depth),
            str(elevation),
            "--dem",
            str(dem_path),
            "--dem-opacity",
            "0.25",
            "--coastline",
            str(coast),
            "--topography",
            str(topo),
        ],
    )
    assert result.exit_code == 0, result.output

    calls = plotter.add_mesh.call_args_list
    np.testing.assert_allclose(_points_of(calls[0])[:, 2], [-300, -300, -300, -200])
    np.testing.assert_allclose(_points_of(calls[1])[:, 2], [0, 0, 0, 100])
    assert calls[2].kwargs["style"] == "wireframe"
    assert calls[2].args[0].n_points == pv.Sphere().n_points
    dem_call = calls[3]
    assert dem_call.kwargs["name"] == "dem"
    assert dem_call.kwargs["opacity"] == 0.25
    np.testing.assert_allclose(dem_call.args[0].point_data["elevation"], 300.0)
    coast_call = calls[4]
    assert coast_call.kwargs["name"] == "coastline"
    # 500 m above the shallowest point of either mesh (z = -300).
    np.testing.assert_allclose(_points_of(coast_call)[:, 2], -800.0)
