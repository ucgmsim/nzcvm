"""Render a fence diagram of Vs from a generated velocity model.

Draws vertical sections of shear-wave velocity standing on shaded relief,
viewed obliquely. A TOML config places each fence (its centre, azimuth and
length) and sets the camera and figure styling. See
``scripts/canterbury_fence.toml`` for an annotated example.

Everything comes from the model output. Each fence samples the model by
bilinear interpolation along its line, so a fence may run at any angle. The
relief shades the grid's top surface (``k = 0``) and sits on its bottom
surface, where the fences end.

``just canterbury_fence`` rebuilds the README figure. It runs::

    nzcvm generate examples/canterbury.toml canterbury.zarr
    python scripts/fence_diagram.py scripts/canterbury_fence.toml \\
        canterbury.zarr docs/images/canterbury_fence.png

Needs the ``vis`` extra (pyvista and matplotlib).
"""

import dataclasses
import tomllib
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import typer
import xarray as xr
from matplotlib import patheffects
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from pyproj import Transformer

#: Shaded-relief colours: land from lowland to peaks, and the sea.
LAND_CMAP = LinearSegmentedColormap.from_list(
    "relief", ["#9a9a7e", "#b8b39c", "#d6d0c2", "#f4f2ee"]
)
LAND_RANGE = (0.0, 3500.0)
SEA_COLOUR = np.array([0.72, 0.76, 0.81])
#: Render size of the 3D scene, in pixels.
WINDOW = [2400, 1500]

app = typer.Typer(add_completion=False)


@dataclasses.dataclass
class Fence:
    """One vertical section.

    Attributes
    ----------
    lon, lat :
        Centre of the fence, in WGS84 degrees.
    azimuth :
        Direction from the fence's start to its end, in degrees clockwise
        from grid north of the model's projection.
    length :
        Length in metres. ``None`` runs the fence across the whole domain.
        Either way, the domain clips the fence.
    labels :
        Text drawn beside the fence's start and end, or ``None`` for none.
    """

    lon: float
    lat: float
    azimuth: float
    length: float | None = None
    labels: tuple[str, str] | None = None


@dataclasses.dataclass
class View:
    """The camera.

    Attributes
    ----------
    azimuth :
        Bearing from the scene centre to the camera, degrees clockwise from
        north.
    elevation :
        Angle of the camera above the horizontal, in degrees.
    zoom :
        Zoom applied after fitting the scene to the window.
    """

    azimuth: float = 204.0
    elevation: float = 48.0
    zoom: float = 1.25


@dataclasses.dataclass
class FigureStyle:
    """Figure styling.

    Attributes
    ----------
    title :
        Figure title.
    exaggeration :
        Vertical exaggeration of the fences and relief.
    vs_range :
        Colour limits for Vs, in m/s.
    cmap :
        Matplotlib colour map for Vs.
    scale_bar :
        Length of the depth scale bar, in metres.
    """

    title: str = "Velocity model"
    exaggeration: float = 3.0
    vs_range: tuple[float, float] = (300.0, 3500.0)
    cmap: str = "viridis_r"
    scale_bar: float = 5000.0


@dataclasses.dataclass
class FenceConfig:
    """The fences, camera and styling of a fence diagram.

    Attributes
    ----------
    fences :
        The fences, drawn in order. The first fence's labels sit at its ends.
    view :
        The camera.
    figure :
        Figure styling.
    crs :
        Projected CRS of the model's ``x`` and ``y``.
    """

    fences: list[Fence]
    view: View = dataclasses.field(default_factory=View)
    figure: FigureStyle = dataclasses.field(default_factory=FigureStyle)
    crs: str = "EPSG:2193"

    @classmethod
    def load(cls, path: Path) -> "FenceConfig":
        """Read a config from a TOML file."""
        raw = tomllib.loads(path.read_text())
        fences = [Fence(**fence) for fence in raw.pop("fences", [])]
        if not fences:
            raise ValueError(f"{path} has no [[fences]].")
        return cls(
            fences=fences,
            view=View(**raw.pop("view", {})),
            figure=FigureStyle(**raw.pop("figure", {})),
            **raw,
        )


@dataclasses.dataclass
class IndexFrame:
    """The affine map from projected ``(x, y)`` to fractional ``(i, j)``."""

    origin: np.ndarray
    inverse: np.ndarray
    shape: tuple[int, int]

    @classmethod
    def from_surface(cls, x: np.ndarray, y: np.ndarray) -> "IndexFrame":
        """Fit the map to the corners of ``(i, j)`` coordinate arrays."""
        ni, nj = x.shape
        origin = np.array([x[0, 0], y[0, 0]], dtype=float)
        step_i = (np.array([x[-1, 0], y[-1, 0]], dtype=float) - origin) / (ni - 1)
        step_j = (np.array([x[0, -1], y[0, -1]], dtype=float) - origin) / (nj - 1)
        inverse = np.linalg.inv(np.column_stack([step_i, step_j]))
        return cls(origin, inverse, (ni, nj))

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fractional ``(i, j)`` of each point, as an ``(n, 2)`` array."""
        return (np.column_stack([x, y]) - self.origin) @ self.inverse.T

    def inside(self, ij: np.ndarray) -> np.ndarray:
        """Whether each fractional index lies in the grid."""
        upper = np.array(self.shape) - 1
        return np.all((ij >= 0) & (ij <= upper), axis=1)


def sample_fence(
    fence: Fence,
    centre: np.ndarray,
    frame: IndexFrame,
    spacing: float,
    variables: list[xr.DataArray],
) -> list[np.ndarray]:
    """Sample *variables* along *fence* by bilinear interpolation.

    Parameters
    ----------
    fence :
        The fence to sample.
    centre :
        The fence's centre, in projected coordinates.
    frame :
        The map from projected coordinates to grid indices.
    spacing :
        Distance between samples along the fence, in metres.
    variables :
        ``(i, j, k)`` arrays to sample.

    Returns
    -------
    list of numpy.ndarray
        One ``(n, k)`` array per variable, for the *n* samples inside the
        domain, ordered from the fence's start to its end.
    """
    ni, nj = frame.shape
    half = (fence.length or spacing * (ni + nj)) / 2
    distance = np.arange(-half, half + spacing / 2, spacing)
    bearing = np.radians(fence.azimuth)
    x = centre[0] + distance * np.sin(bearing)
    y = centre[1] + distance * np.cos(bearing)
    ij = frame(x, y)
    ij = ij[frame.inside(ij)]
    if len(ij) < 2:
        raise ValueError(f"{fence} lies outside the model domain.")

    # Load the box around the fence, then interpolate its corners.
    base = np.minimum(np.floor(ij).astype(int), [ni - 2, nj - 2])
    weight = ij - base
    low, high = base.min(axis=0), base.max(axis=0) + 2
    i, j = (base - low).T
    wi, wj = weight[:, :1], weight[:, 1:]
    samples = []
    for variable in variables:
        box = variable.isel(i=slice(low[0], high[0]), j=slice(low[1], high[1]))
        box = box.values.astype(float)
        samples.append(
            (1 - wi) * (1 - wj) * box[i, j]
            + wi * (1 - wj) * box[i + 1, j]
            + (1 - wi) * wj * box[i, j + 1]
            + wi * wj * box[i + 1, j + 1]
        )
    return samples


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
    config_path: Annotated[
        Path, typer.Argument(help="Fence diagram config (TOML).", exists=True)
    ],
    model: Annotated[Path, typer.Argument(help="Output of `nzcvm generate`.")],
    output: Annotated[Path, typer.Argument(help="PNG to write.")],
) -> None:
    """Render a fence diagram of Vs through MODEL to OUTPUT."""
    config = FenceConfig.load(config_path)
    style = config.figure
    exaggeration = style.exaggeration
    cmap = plt.get_cmap(style.cmap)

    tree = xr.open_datatree(model, engine="zarr")
    (name,) = tree["grids"].children
    grid = tree[f"grids/{name}"].ds
    vs = tree[f"qualities/{name}"].ds.vs

    top = grid.isel(k=0)[["x", "y", "z"]].compute()
    frame = IndexFrame.from_surface(top.x.values, top.y.values)
    spacing = float(np.hypot(top.x[1, 0] - top.x[0, 0], top.y[1, 0] - top.y[0, 0]))
    to_projected = Transformer.from_crs("EPSG:4326", config.crs, always_xy=True)

    plotter = pv.Plotter(off_screen=True, window_size=WINDOW)
    plotter.renderer.set_background("white")

    # Relief: the top surface's shading, draped on the bottom surface.
    relief = surface_mesh(
        top.x.values, top.y.values, -grid.z.isel(k=-1).values * exaggeration
    )
    relief.point_data["rgb"] = relief_colours(-top.z.values, spacing).reshape(
        -1, 3, order="F"
    )
    plotter.add_mesh(relief, scalars="rgb", rgb=True, lighting=False)

    labels = []
    for fence in config.fences:
        centre = np.array(to_projected.transform(fence.lon, fence.lat))
        x, y, z, fence_vs = sample_fence(
            fence, centre, frame, spacing, [grid.x, grid.y, grid.z, vs]
        )
        mesh = surface_mesh(x, y, -z * exaggeration)
        mesh.point_data["vs"] = fence_vs.ravel(order="F")
        plotter.add_mesh(
            mesh,
            scalars="vs",
            cmap=cmap,
            clim=style.vs_range,
            lighting=False,
            show_scalar_bar=False,
        )
        crest = np.column_stack([x[:, 0], y[:, 0], -z[:, 0] * exaggeration])
        plotter.add_mesh(
            pv.lines_from_points(crest), color="#1a1a1a", line_width=3, lighting=False
        )
        if fence.labels:
            # Anchor each label at mid-height of the fence's end.
            ends = np.column_stack(
                [x[[0, -1], 0], y[[0, -1], 0], -z[[0, -1]].mean(axis=1) * exaggeration]
            )
            labels.append((fence.labels, ends))

    # Oblique parallel view.
    view = config.view
    bearing, tilt = np.radians(view.azimuth), np.radians(view.elevation)
    direction = np.array(
        [
            np.sin(bearing) * np.cos(tilt),
            np.cos(bearing) * np.cos(tilt),
            np.sin(tilt),
        ]
    )
    scene_centre = np.array(relief.center)
    plotter.camera.parallel_projection = True
    plotter.camera_position = [
        scene_centre + direction * 1e6,
        scene_centre,
        (0.0, 0.0, 1.0),
    ]
    plotter.renderer.reset_camera()
    plotter.camera.zoom(view.zoom)
    image = plotter.screenshot(return_img=True)
    assert image is not None

    # Annotation anchors, projected while the scene is still live.
    label_pixels = [(text, to_pixels(plotter, ends)) for text, ends in labels]
    bar_top, bar_bottom = to_pixels(
        plotter,
        np.array(
            [scene_centre, scene_centre - (0.0, 0.0, style.scale_bar * exaggeration)]
        ),
    )
    bar_length = float(np.hypot(*(bar_bottom - bar_top)))
    plotter.close()

    # Compose the figure around the render.
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(13, 13 * (height + 260) / width), dpi=150)
    ax = fig.add_axes((0.0, 260 / (height + 260), 1.0, height / (height + 260)))
    ax.imshow(image)
    ax.set_axis_off()

    # Each label sits just past its end of the fence, along the fence.
    outline = [patheffects.withStroke(linewidth=4, foreground="white")]
    for (start_text, end_text), (start, end) in label_pixels:
        along = (end - start) / np.linalg.norm(end - start)
        for text, anchor in (
            (start_text, start - 80 * along),
            (end_text, end + 80 * along),
        ):
            ax.text(
                anchor[0],
                anchor[1],
                text,
                fontsize=18,
                weight="bold",
                ha="center",
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
        f"{style.scale_bar / 1000:g} km\ndepth",
        fontsize=11,
        color="#555555",
        va="center",
    )

    fig.text(0.045, 0.95, style.title, fontsize=20, va="top")

    cax = fig.add_axes((0.045, 0.07, 0.38, 0.025))
    colourbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(*style.vs_range), cmap=cmap),
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
