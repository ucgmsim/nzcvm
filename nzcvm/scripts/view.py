"""
3D visualisation of xarray DataTree grid data using PyVista + Typer.

Usage examples
--------------
# Show depth scalar, both grids, interactive orthogonal slicer
python visualise_grid.py model.nc --scalar depth

# Show absolute difference between two models for the Vs scalar
python visualise_grid.py model.nc --scalar vs --compare-to model2.nc --diff-mode abs
"""

from __future__ import annotations

import gzip
import itertools
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import shapely
import typer
import xarray as xr

# Adjust these imports according to your local package structure
from nzcvm.components import Component
from nzcvm.grids.grid import Grid
from nzcvm.models.mesh import TetrahedralMeshSchema
from nzcvm.qualities import Qualities
from nzcvm.velocity_model import VelocityModel

if TYPE_CHECKING:
    import pyvista as pv


def _require_pyvista():  # type: ignore[return]
    """Import pyvista or raise a helpful error."""
    try:
        import pyvista as pv

        return pv
    except ImportError as err:
        raise ImportError(
            "pyvista is required for the view command. "
            "Install it with: pip install nzcvm[visualization]"
        ) from err


# ---------------------------------------------------------------------------
# Types & Constants
# ---------------------------------------------------------------------------


class DiffMode(StrEnum):
    NONE = auto()
    ABS = auto()
    LOG = auto()


QUALITIES_COMPONENTS = ("rho", "vp", "vs", "qp", "qs", "alpha")
DEPTH_SCALAR = "depth"
LAYER_SCALAR = "layer"
ALL_SCALARS = (DEPTH_SCALAR, LAYER_SCALAR, *QUALITIES_COMPONENTS)

app = typer.Typer(
    name="visualise-grid",
    help="Interactive 3D PyVista viewer for xarray DataTree model grids.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def add_logical_axes(
    pl: pv.Plotter, grid: xr.Dataset, bounds: tuple[float, ...]
) -> None:
    """Draws i, j, k logical direction vectors starting from the logical origin."""
    pv = _require_pyvista()
    w, e, s, n, z_min, z_max = bounds
    scale = np.linalg.norm([e - w, n - s, z_max - z_min]) * 0.15

    def _get_pt(i: int, j: int, k: int) -> np.ndarray:
        pt = grid.isel(i=i, j=j, k=k).compute()
        return np.array([pt.x.item(), pt.y.item(), pt.z.item()])

    try:
        p0 = _get_pt(0, 0, 0)
        axes_config = [
            (_get_pt(1, 0, 0), "i", "red"),
            (_get_pt(0, 1, 0), "j", "green"),
            (_get_pt(0, 0, 1), "k", "blue"),
        ]
    except Exception as exc:  # noqa: BLE001 - optional viewer annotation, must not crash the viewer
        typer.secho(f"⚠️ Could not calculate logical axes: {exc}", fg="yellow")
        return

    for p_axis, label, color in axes_config:
        vec = p_axis - p0
        if label != "k":
            vec[-1] = 0.0

        if not (norm := np.linalg.norm(vec)):
            continue

        direction = vec / norm
        pl.add_mesh(
            pv.Arrow(
                start=p0,
                direction=direction,
                scale=scale,
                shaft_radius=0.02,
                tip_radius=0.06,
                tip_length=0.2,
            ),
            color=color,
            lighting=True,
        )

        pl.add_point_labels(
            [p0 + direction * (scale * 1.1)],
            [label],
            text_color=color,
            point_size=0,
            shape_opacity=0.0,
            font_size=24,
            always_visible=True,
            margin=0,
        )


def add_coastline_underlay(
    pl: pv.Plotter, bounds: tuple[float, ...], coastline_path: Path
) -> None:
    """Reads a vector geometry file in NZTM and plots it as a clean line underlay."""
    pv = _require_pyvista()
    z_level = bounds[4] - 500
    with gzip.open(coastline_path) as handle:
        coastline = shapely.from_wkb(handle.read())

    points, lines, offset = [], [], 0

    for geom in (g for g in coastline.geoms if g and not g.is_empty):
        sub_geoms = (
            geom.geoms
            if geom.geom_type in ("MultiPolygon", "MultiLineString")
            else [geom]
        )
        for sg in sub_geoms:
            rings = (
                [sg.exterior] + list(sg.interiors)
                if sg.geom_type == "Polygon"
                else ([sg] if sg.geom_type in ("LineString", "LinearRing") else [])
            )

            for ring in rings:
                coords = np.array(ring.coords)
                n_pts = len(coords)
                points.append(
                    np.column_stack(
                        (coords[:, 0], coords[:, 1], np.full(n_pts, z_level))
                    )
                )
                lines.append(np.hstack([[n_pts], np.arange(offset, offset + n_pts)]))
                offset += n_pts

    if not points:
        return typer.secho(
            "⚠️ No valid geometries found in the coastline file.", fg="yellow"
        )

    poly = pv.PolyData(np.vstack(points))
    poly.lines = np.hstack(lines)
    pl.add_mesh(poly, color="red", line_width=2.5, opacity=0.8, name="coastline")


def _build_structured_grid(
    layer: int,
    grid1: Grid,
    qualities1: Qualities,
    active_scalar: str,
    stride: int,
    grid2: Grid | None = None,
    qualities2: Qualities | None = None,
    diff_mode: DiffMode = DiffMode.NONE,
) -> pv.StructuredGrid:
    """Build a PyVista StructuredGrid from merged xarray Datasets, optionally applying a functional difference mapping."""
    pv = _require_pyvista()
    ds1 = xr.merge([grid1, qualities1])

    if diff_mode != DiffMode.NONE and grid2 is not None and qualities2 is not None:
        ds2 = xr.merge([grid2, qualities2])

        ops = {
            DiffMode.ABS: lambda a, b: a - b,
            DiffMode.LOG: lambda a, b: (
                xr.apply_ufunc(np.log10, a) - xr.apply_ufunc(np.log10, b)
            ),
        }
        for var in ds1.data_vars:
            if var in ("x", "y", "z") or var not in ds2:
                continue

            ds1[var] = ops[diff_mode](ds1[var], ds2[var])

    if stride > 1:
        ds1 = ds1.coarsen(  # ty: ignore[unresolved-attribute]
            i=stride, j=stride, boundary="trim"
        ).mean()

    mesh = pv.StructuredGrid(ds1.x.values, ds1.y.values, ds1.z.values)

    for scalar in ALL_SCALARS:
        if scalar == LAYER_SCALAR:
            mesh.point_data[scalar] = np.full_like(ds1.z.values, layer).ravel(order="F")
        elif scalar in ds1:
            mesh.point_data[scalar] = ds1[scalar].values.ravel(order="F")

    if active_scalar in mesh.point_data:
        mesh.set_active_scalars(active_scalar)

    return mesh


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def basin(
    mesh: Annotated[
        list[Path], typer.Argument(help="Mesh files to read.", exists=True)
    ],
    scalar: Annotated[
        str | None, typer.Option(help="Material property to display.")
    ] = None,
    topography: Annotated[
        Path | None,
        typer.Option(help="Optional topography mesh to overlay.", exists=True),
    ] = None,
    coastline: Annotated[
        Path | None,
        typer.Option(help="Path to coastline vector file (NZTM).", exists=True),
    ] = None,
    x_pos: Annotated[
        float | None, typer.Option(help="Add a vertical pole at (x, y).")
    ] = None,
    y_pos: Annotated[
        float | None, typer.Option(help="Add a vertical pole at (x, y).")
    ] = None,
) -> None:
    """Entry point for the ``nzcvm view-basin`` command."""

    pv = _require_pyvista()
    pl = pv.Plotter()

    if scalar:
        # Needed so that PyVista renders opacity correctly
        pl.enable_depth_peeling(number_of_peels=10, occlusion_ratio=0.0)

    # A palette of visually distinct colors to cycle through for the meshes
    colors = itertools.cycle(
        ["red", "blue", "green", "yellow", "cyan", "magenta", "orange", "purple"]
    )

    # Track overall bounds [x_min, x_max, y_min, y_max, z_min, z_max] across all meshes
    global_bounds = [
        float("inf"),
        float("-inf"),
        float("inf"),
        float("-inf"),
        float("inf"),
        float("-inf"),
    ]

    for i, current_mesh in enumerate(mesh):
        mesh_dset = TetrahedralMeshSchema.from_dataset(xr.open_dataset(current_mesh))

        points = np.c_[mesh_dset.x.values, mesh_dset.y.values, mesh_dset.z.values]
        connectivity = mesh_dset.connectivity.values
        cell_length = connectivity.shape[1]
        lengths = np.full(
            (connectivity.shape[0], 1), cell_length, dtype=connectivity.dtype
        )
        cell_type = np.full(len(lengths), pv.CellType.TETRA)
        cells = np.hstack((lengths, connectivity)).ravel()
        mesh_data = pv.UnstructuredGrid(cells, cell_type, points)

        # Update global bounds
        b = mesh_data.bounds
        global_bounds[0] = min(global_bounds[0], b[0])
        global_bounds[1] = max(global_bounds[1], b[1])
        global_bounds[2] = min(global_bounds[2], b[2])
        global_bounds[3] = max(global_bounds[3], b[3])
        global_bounds[4] = min(global_bounds[4], b[4])
        global_bounds[5] = max(global_bounds[5], b[5])

        if scalar:
            mesh_data.point_data[scalar] = mesh_dset[scalar].values
            mesh_data.point_data["alpha"] = mesh_dset["alpha"].values
            mesh_data.point_data.active_scalars_name = scalar

            pl.add_mesh(mesh_data, cmap="hot", opacity="alpha", show_scalar_bar=False)

            # Only add the invisible mesh for the scalar bar once
            if i == 0:
                pl.add_mesh(mesh_data, cmap="hot", opacity=0.0, show_scalar_bar=True)
        else:
            # Assign the next distinct color from the cycle
            pl.add_mesh(mesh_data, color=next(colors))

    if topography:
        pl.add_mesh(
            pv.read(topography),
            style="wireframe",
            color="black",
            opacity=0.3,
            label="Surface",
        )

    if x_pos and y_pos:
        # Use global mesh bounds to define the vertical extent across all models
        z_min, z_max = global_bounds[4], global_bounds[5]
        # Create a line running from deep under the meshes to high over them
        start = (x_pos, y_pos, z_min - 5000)
        end = (x_pos, y_pos, z_max + 5000)

        pl.add_mesh(
            pv.Line(start, end), color="cyan", line_width=5, label="Location Marker"
        )

    if coastline:
        # Coastline should map to the combined extents of all loaded meshes
        add_coastline_underlay(pl, tuple(global_bounds), coastline)

    pl.camera.up = (0.0, 0.0, -1.0)
    pl.show()


@app.command()
def model(
    filepath: Annotated[
        Path, typer.Argument(help="Path to the primary NetCDF/Zarr file.", exists=True)
    ],
    scalar: Annotated[Component, typer.Option("--scalar", "-s")] = Component.VS,
    compare_to: Annotated[
        Path | None,
        typer.Option(
            "--compare-to",
            "-b",
            help="Optional second file path to evaluate diff against.",
            exists=True,
        ),
    ] = None,
    diff_mode: Annotated[
        DiffMode,
        typer.Option(
            "--diff-mode",
            "-d",
            help="Algebraic strategy used if computing a comparison.",
        ),
    ] = DiffMode.NONE,
    coastline: Annotated[
        Path | None,
        typer.Option(
            "--coastline", "-c", help="Path to coastline vector file.", exists=True
        ),
    ] = None,
    stride: Annotated[
        int, typer.Option("--stride", "-S", help="Downsampling stride factor.")
    ] = 4,
    show_axes: Annotated[
        bool,
        typer.Option("--show-axes/--no-axes", help="Show the i, j, k logical axes"),
    ] = True,
    slice_mode: Annotated[
        str, typer.Option(help="'orthogonal' or 'none'.")
    ] = "orthogonal",
    slice_axis: Annotated[
        str | None, typer.Option(help="Axis to static slice ('x', 'y', 'z').")
    ] = None,
    cmap: Annotated[
        str, typer.Option(help="Matplotlib / PyVista colormap name.")
    ] = "viridis",
    opacity: Annotated[float, typer.Option(help="Mesh opacity [0, 1].")] = 1.0,
    show_edges: Annotated[bool, typer.Option("--show-edges/--no-edges")] = False,
    off_screen: Annotated[bool, typer.Option(help="Render off-screen.")] = False,
    screenshot: Annotated[
        Path | None, typer.Option(help="Save a PNG screenshot.")
    ] = None,
    min_val: float | None = None,
    max_val: float | None = None,
) -> None:
    """Load an xarray DataTree pipeline and visualise model grids cleanly."""
    pv = _require_pyvista()
    scalar_str = str(scalar.value if isinstance(scalar, Component) else scalar).lower()

    if scalar_str not in ALL_SCALARS:
        typer.secho(
            f"[error] Unknown scalar '{scalar_str}'. Choose from: {', '.join(ALL_SCALARS)}",
            fg="red",
        )
        raise typer.Exit(1)

    typer.echo(f"Loading DataTree A from {filepath} …")
    vmod1 = VelocityModel.from_datatree(xr.open_datatree(str(filepath), chunks="auto"))

    vmod2 = None
    if compare_to:
        typer.echo(
            f"Loading DataTree B from {compare_to} for comparison [{diff_mode.upper()}] …"
        )
        vmod2 = VelocityModel.from_datatree(
            xr.open_datatree(str(compare_to), chunks="auto")
        )

    grids = {}
    for i, (name, (grid1, qualities1)) in enumerate(vmod1.pairwise.items()):
        typer.echo(f"  Building '{name}' (stride={stride}) …")
        try:
            # Safely query structural counterpart from model B if comparing models
            pair2 = vmod2.pairwise.get(name) if vmod2 else None
            grid2, qualities2 = pair2 if pair2 else (None, None)

            grids[name] = (
                g := _build_structured_grid(
                    i,
                    grid1,
                    qualities1,
                    scalar_str,
                    stride,
                    grid2,
                    qualities2,
                    diff_mode,
                )
            )
            typer.echo(f"    → {g.n_points:,} points")
        except Exception as exc:  # noqa: BLE001 - one bad grid must not stop the whole viewer
            typer.secho(f"  [error] Failed to build grid for '{name}': {exc}", fg="red")

    if not grids:
        typer.secho("No grids could be built. Exiting.", fg="red")
        raise typer.Exit(1)

    global_bounds = pv.MultiBlock(list(grids.values())).bounds
    all_vals = np.concatenate([g.point_data[scalar_str] for g in grids.values()])
    clim = [
        min_val or float(np.nanmin(all_vals)),
        max_val or float(np.nanmax(all_vals)),
    ]

    # Choose a smart diverging colormap automatically when running comparison transformations
    final_cmap = cmap if diff_mode == DiffMode.NONE else "bwr"
    pl = pv.Plotter(
        title=f"Grid viewer — {scalar_str} ({diff_mode})", off_screen=off_screen
    )

    if coastline:
        add_coastline_underlay(pl, global_bounds, coastline)
    if show_axes:
        for grid_ds, _ in vmod1.pairwise.values():
            add_logical_axes(pl, grid_ds, global_bounds)

    mesh_kwargs = {
        "scalars": scalar_str,
        "cmap": final_cmap,
        "clim": clim,
        "opacity": opacity,
        "show_edges": show_edges,
        "show_scalar_bar": False,
    }

    if slice_axis:
        all_bounds = [g.bounds for g in grids.values()]

        match slice_axis.lower():
            case "x":
                min_val, max_val, normal = (
                    min(b[0] for b in all_bounds),
                    max(b[1] for b in all_bounds),
                    [1, 0, 0],
                )
            case "y":
                min_val, max_val, normal = (
                    min(b[2] for b in all_bounds),
                    max(b[3] for b in all_bounds),
                    [0, 1, 0],
                )
            case "z":
                min_val, max_val, normal = (
                    min(b[4] for b in all_bounds),
                    max(b[5] for b in all_bounds),
                    [0, 0, 1],
                )
            case invalid:
                typer.secho(f"[error] Invalid axis '{invalid}'.", fg="red")
                raise typer.Exit(1)

        mid_val = (min_val + max_val) / 2

        def create_slice(value: float) -> None:
            origin = [
                value if normal[0] else 0,
                value if normal[1] else 0,
                value if normal[2] else 0,
            ]
            for name, g in grids.items():
                pl.add_mesh(
                    g.slice(normal=normal, origin=origin),
                    name=f"slice_{name}",
                    **mesh_kwargs,
                )

        pl.add_slider_widget(
            callback=create_slice,
            rng=[min_val, max_val],
            value=mid_val,
            title=f"{slice_axis.upper()} Slice Location",
            pointa=(0.025, 0.1),
            pointb=(0.31, 0.1),
            interaction_event="always",
        )
        create_slice(mid_val)

    elif slice_mode.lower() == "orthogonal":
        for g in grids.values():
            pl.add_mesh_slice_orthogonal(g, **mesh_kwargs)
    else:
        for name, g in grids.items():
            pl.add_mesh(g, label=name, **mesh_kwargs)

    def pick_point_callback(point_data: np.ndarray) -> None:
        if point_data is None or point_data.size == 0:
            return

        x, y, z = point_data[:3]
        poly = pv.PolyData([[x, y, z]])

        sample_result = next(
            (
                out
                for g in grids.values()
                if (out := poly.sample(g)).point_data.get("vtkValidPointMask", [0])[0]
            ),
            None,
        )

        msg = f"Picked Point:\nX: {x:.3f}\nY: {y:.3f}\nZ: {z:.3f}\n\n"

        if sample_result:
            for name, arr in sample_result.point_data.items():
                if name != "vtkValidPointMask":
                    msg += f"{name}: {arr[0]:.3g}\n"
        else:
            msg += "Point clicked outside of all grids."

        pl.add_text(
            msg.strip(),
            position="upper_left",
            font_size=10,
            color="black",
            name="picker_text",
        )

    pl.enable_point_picking(
        callback=pick_point_callback, show_message="", color="pink", point_size=10
    )
    pl.add_scalar_bar(
        title=f"▲ {scalar_str}" if diff_mode != DiffMode.NONE else scalar_str,
        fmt="%.3g",
        position_x=0.85,
        position_y=0.05,
        vertical=True,
    )
    pl.add_axes(xlabel="X (m)", ylabel="Y (m)", zlabel="m (Z)")

    for name, g in grids.items():
        pl.add_point_labels(
            [g.center],
            [name],
            point_size=0,
            font_size=10,
            text_color="yellow",
            shape_opacity=0.0,
        )

    pl.camera_position = "iso"
    pl.camera.up = (0.0, 0.0, -1.0)
    pl.reset_camera()

    if screenshot:
        pl.show(auto_close=False)
        pl.screenshot(str(screenshot))
        typer.echo(f"Screenshot saved to {screenshot}")
    else:
        pl.show()


if __name__ == "__main__":
    app()
