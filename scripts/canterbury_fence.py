"""Render the README fence diagram from a generated velocity model.

Draws west-east vertical sections of shear-wave velocity standing on shaded
relief, viewed obliquely from the south-west. Everything comes from the
model output. Each section is a row of fixed ``j``. The relief shades the
grid's top surface (``k = 0``) and sits on its bottom surface, where the
sections end.

``just canterbury_fence`` rebuilds the README figure. It runs::

    nzcvm generate examples/canterbury.toml canterbury.zarr
    python scripts/canterbury_fence.py canterbury.zarr docs/images/canterbury_fence.png

Needs the ``vis`` extra (pyvista and matplotlib).
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import typer
import xarray as xr
from matplotlib import patheffects
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize

#: Colour map and limits for Vs (m/s), shared by the sections and colour bar.
CMAP = plt.get_cmap("viridis_r")
VS_RANGE = (300.0, 3500.0)
#: Shaded-relief colours: land from lowland to peaks, and the sea.
LAND_CMAP = LinearSegmentedColormap.from_list(
    "relief", ["#9a9a7e", "#b8b39c", "#d6d0c2", "#f4f2ee"]
)
LAND_RANGE = (0.0, 3500.0)
SEA_COLOUR = np.array([0.72, 0.76, 0.81])
#: Render size of the 3D scene, in pixels.
WINDOW = (2400, 1500)

app = typer.Typer(add_completion=False)


def surface_mesh(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> pv.StructuredGrid:
    """A structured surface from 2D ``(n, m)`` coordinate arrays."""
    return pv.StructuredGrid(x[..., None], y[..., None], z[..., None])


def relief_colours(elevation: np.ndarray, spacing: float) -> np.ndarray:
    """Hill-shaded RGB colours for an ``(i, j)`` elevation grid.

    Parameters
    ----------
    elevation :
        Elevation in metres, positive up, ``i`` east and ``j`` north.
    spacing :
        Node spacing in metres.

    Returns
    -------
    numpy.ndarray
        ``(i, j, 3)`` RGB colours in ``[0, 1]``. Nodes at or below sea level
        take the flat sea colour.
    """
    # LightSource wants rows running north to south.
    rows = elevation.T[::-1]
    shaded = LightSource(azdeg=315, altdeg=45).shade(
        rows,
        cmap=LAND_CMAP,
        norm=Normalize(*LAND_RANGE),
        blend_mode="soft",
        vert_exag=2.0,
        dx=spacing,
        dy=spacing,
    )[..., :3]
    rgb = shaded[::-1].transpose(1, 0, 2)
    rgb[elevation <= 0.0] = SEA_COLOUR
    return rgb


def to_pixels(plotter: pv.Plotter, points: np.ndarray) -> np.ndarray:
    """Project world *points* to image pixels (origin top left)."""
    width, height = plotter.window_size
    projection = pv.array_from_vtkmatrix(
        plotter.camera.GetCompositeProjectionTransformMatrix(width / height, -1, 1)
    )
    points = np.atleast_2d(points)
    clip = np.column_stack([points, np.ones(len(points))]) @ projection.T
    ndc = clip[:, :2] / clip[:, 3:]
    return np.column_stack([(ndc[:, 0] + 1) / 2 * width, (1 - ndc[:, 1]) / 2 * height])


@app.command()
def main(
    model: Annotated[Path, typer.Argument(help="Output of `nzcvm generate`.")],
    output: Annotated[Path, typer.Argument(help="PNG to write.")],
    sections: Annotated[int, typer.Option(help="Number of sections.", min=1)] = 4,
    exaggeration: Annotated[
        float, typer.Option(help="Vertical exaggeration.", min=0.0)
    ] = 8.0,
    title: Annotated[str, typer.Option(help="Figure title.")] = (
        "Canterbury velocity model"
    ),
) -> None:
    """Render a fence diagram of Vs through MODEL to OUTPUT."""
    tree = xr.open_datatree(model, engine="zarr")
    (name,) = tree["grids"].children
    grid = tree[f"grids/{name}"].ds
    vs = tree[f"qualities/{name}"].ds.vs
    ni, nj, _ = vs.shape

    # Sections centred in equal strips across the domain, south to north.
    rows = np.round((np.arange(sections) + 0.5) / sections * (nj - 1)).astype(int)

    plotter = pv.Plotter(off_screen=True, window_size=WINDOW)
    plotter.set_background("white")

    # Relief: the top surface's shading, draped on the bottom surface.
    top = grid.isel(k=0)[["x", "y", "z"]].compute()
    bottom_z = grid.z.isel(k=-1).values
    spacing = float(np.hypot(top.x[1, 0] - top.x[0, 0], top.y[1, 0] - top.y[0, 0]))
    elevation = -top.z.values
    relief = surface_mesh(top.x.values, top.y.values, -bottom_z * exaggeration)
    relief.point_data["rgb"] = relief_colours(elevation, spacing).reshape(
        -1, 3, order="F"
    )
    plotter.add_mesh(relief, scalars="rgb", rgb=True, lighting=False)

    for j in rows:
        section = grid.isel(j=j)[["x", "y", "z"]].compute()
        fence = surface_mesh(
            section.x.values, section.y.values, -section.z.values * exaggeration
        )
        fence.point_data["vs"] = vs.isel(j=j).values.ravel(order="F")
        plotter.add_mesh(
            fence,
            scalars="vs",
            cmap=CMAP,
            clim=VS_RANGE,
            lighting=False,
            show_scalar_bar=False,
        )
        crest = np.column_stack(
            [section.x[:, 0], section.y[:, 0], -section.z[:, 0] * exaggeration]
        )
        plotter.add_mesh(
            pv.lines_from_points(crest), color="#1a1a1a", line_width=3, lighting=False
        )

    # Oblique parallel view from the south-south-west.
    centre = np.array(relief.center)
    plotter.enable_parallel_projection()
    plotter.camera_position = [
        centre + np.array([-0.45, -1.0, 1.2]) * 1e6,
        centre,
        (0.0, 0.0, 1.0),
    ]
    plotter.reset_camera()
    plotter.camera.zoom(1.25)
    image = plotter.screenshot(return_img=True)

    # Annotation anchors, projected while the scene is still live.
    # W and E sit beside the mid-height of the front section's two ends.
    front = grid.isel(j=rows[0], i=[0, ni - 1])[["x", "y", "z"]].compute()
    ends = np.column_stack(
        [
            front.x[:, 0],
            front.y[:, 0],
            -front.z.mean("k") * exaggeration,
        ]
    ).astype(float)
    west, east = to_pixels(plotter, ends)
    bar_top, bar_bottom = to_pixels(
        plotter, [centre, centre - np.array([0.0, 0.0, 5000.0 * exaggeration])]
    )
    bar_length = float(np.hypot(*(bar_bottom - bar_top)))
    plotter.close()

    # Compose the figure around the render.
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(13, 13 * (height + 260) / width), dpi=150)
    ax = fig.add_axes((0.0, 260 / (height + 260), 1.0, height / (height + 260)))
    ax.imshow(image)
    ax.set_axis_off()

    outline = [patheffects.withStroke(linewidth=4, foreground="white")]
    offset = 70.0
    ax.text(
        *west + (-offset, 0),
        "W",
        fontsize=18,
        weight="bold",
        ha="right",
        va="center",
        path_effects=outline,
    )
    ax.text(
        *east + (offset, 0),
        "E",
        fontsize=18,
        weight="bold",
        ha="left",
        va="center",
        path_effects=outline,
    )

    bar_x, bar_y = 0.06 * width, 0.62 * height
    ax.plot([bar_x, bar_x], [bar_y, bar_y + bar_length], color="#1a1a1a", lw=1.5)
    for y in (bar_y, bar_y + bar_length):
        ax.plot([bar_x - 12, bar_x + 12], [y, y], color="#1a1a1a", lw=1.5)
    ax.text(
        bar_x + 25,
        bar_y + bar_length / 2,
        "5 km\ndepth",
        fontsize=11,
        color="#555555",
        va="center",
    )

    fig.text(0.045, 0.95, title, fontsize=20, va="top")

    cax = fig.add_axes((0.045, 0.07, 0.38, 0.025))
    colourbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(*VS_RANGE), cmap=CMAP),
        cax=cax,
        orientation="horizontal",
    )
    colourbar.set_label("Shear-wave velocity  Vs  (m/s)", color="#555555")
    colourbar.ax.tick_params(colors="#555555", labelsize=10)
    colourbar.outline.set_edgecolor("#bbbbbb")

    extent_x = float(np.hypot(top.x[-1, 0] - top.x[0, 0], top.y[-1, 0] - top.y[0, 0]))
    extent_y = float(np.hypot(top.x[0, -1] - top.x[0, 0], top.y[0, -1] - top.y[0, 0]))
    resolution_z = float(np.diff(grid.nominal_depth.values).mean())
    fig.text(
        0.96,
        0.09,
        f"{extent_x / 1000:.0f} × {extent_y / 1000:.0f} km domain · "
        f"{spacing:.0f} m horizontal, {resolution_z:.0f} m vertical sampling\n"
        f"{exaggeration:g}× vertical exaggeration",
        fontsize=11,
        color="#555555",
        ha="right",
        linespacing=1.6,
    )

    fig.savefig(output, dpi=150, facecolor="white")
    typer.echo(f"Wrote {output}")


if __name__ == "__main__":
    app()
