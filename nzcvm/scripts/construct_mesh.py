"""Construct a tetrahedral volumetric mesh for a basin model from an error-bounded triangulation."""

import gzip
import itertools
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import pandas as pd
import pyproj
import scipy.ndimage
import shapely
import shapely.ops
import typer
import xarray as xr

from nzcvm.models.mesh import (
    ModelType,
    TetrahedralMesh,
    TetrahedralMeshSchema,
    make_mesh,
    mesh_encoding,
)
from nzcvm.models.surface import SurfaceGrid, read_surface_file
from nzcvm.nzcvm import adaptive_tin  # ty: ignore[unresolved-import]

TRANSFORMER = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)
INV_TRANSFORMER = pyproj.Transformer.from_crs(2193, 4326, always_xy=True)

PAD_DEPTH = 50.0
#: Default vertical tolerance (m) of the basin triangulation.
TOLERANCE = 5.0
#: Samples per grid cell edge at which the triangulation checks each surface.
SAMPLES_PER_CELL = 2
#: Longest triangle edge (m) when smoothing the basin boundary.
SMOOTHING_MAX_EDGE = 1000.0

app = typer.Typer(help="Construct a volumetric tetrahedral mesh for a basin model.")


def get_surface_bounds(surface_path: Path) -> shapely.Polygon:
    """Return the WGS84 coordinate bounds (min_lat, max_lat, min_lon, max_lon) of a surface."""
    with h5py.File(surface_path, "r") as f:
        lat = f["latitude"][:]
        lon = f["longitude"][:]

    x_lon, y_lat = np.meshgrid(lon, lat)

    x, y = TRANSFORMER.transform(x_lon, y_lat)
    return shapely.box(xmin=x.min(), xmax=x.max(), ymin=y.min(), ymax=y.max())


def preprocess_polygon(poly: shapely.Polygon) -> shapely.Polygon:
    poly_nztm = shapely.ops.transform(TRANSFORMER.transform, poly)
    return poly_nztm


@dataclass
class Triangulation:
    """A planar triangulation of the basin outline."""

    #: ``(V, 2)`` vertex positions (NZTM).
    vertices: np.ndarray
    #: ``(T, 3)`` vertex indices of each triangle.
    triangles: np.ndarray


def interleave_top_and_bottom(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    interleaved = np.zeros((2 * len(top), 3), dtype=np.float32)
    interleaved[::2] = top
    interleaved[1::2] = bottom
    return interleaved


@dataclass
class Layer:
    vertices: np.ndarray
    tetra: np.ndarray
    rho: float
    vp: float
    vs: float
    qp: float
    qs: float
    alpha: float = 1.0


def construct_volumetric_mesh(
    name: str, geometry: shapely.Geometry, layers: list[Layer], priority: int
) -> TetrahedralMesh:

    mesh_vertices = np.concatenate([layer.vertices for layer in layers])
    tetra = np.concatenate([layer.tetra for layer in layers])

    tetra_offset = 0
    vertex_offset = 0

    rho = np.concatenate(
        [
            np.full((len(layer.vertices),), layer.rho, dtype=np.float32)
            for layer in layers
        ]
    )
    vp = np.concatenate(
        [
            np.full((len(layer.vertices),), layer.vp, dtype=np.float32)
            for layer in layers
        ]
    )
    vs = np.concatenate(
        [
            np.full((len(layer.vertices),), layer.vs, dtype=np.float32)
            for layer in layers
        ]
    )
    qp = np.concatenate(
        [
            np.full((len(layer.vertices),), layer.qp, dtype=np.float32)
            for layer in layers
        ]
    )
    qs = np.concatenate(
        [
            np.full((len(layer.vertices),), layer.qs, dtype=np.float32)
            for layer in layers
        ]
    )
    alpha = np.concatenate(
        [
            np.full((len(layer.vertices),), layer.alpha, dtype=np.float32)
            for layer in layers
        ]
    )

    model_type = np.concatenate(
        [
            np.full(len(layer.tetra), ModelType.INTERPOLATE, dtype=np.uint8)
            for layer in layers
        ]
    )

    for layer in layers:
        tetra[tetra_offset : tetra_offset + len(layer.tetra)] += vertex_offset
        vertex_offset += len(layer.vertices)
        tetra_offset += len(layer.tetra)

    priority_data = np.full((len(tetra),), np.uint8(priority))

    mesh = make_mesh(
        name=name,
        points=mesh_vertices,
        connectivity=tetra,
        cell_data={
            "model_type": model_type,
            "priority": priority_data,
        },
        geometry=geometry,
        field_data={"rho": rho, "vp": vp, "vs": vs, "qp": qp, "qs": qs, "alpha": alpha},
    )
    return mesh


def surface_samples(
    grid: SurfaceGrid, poly: shapely.Polygon, per_cell: int
) -> tuple[np.ndarray, float]:
    """Points at which the triangulation must fit `grid`.

    The grid nodes alone miss what bilinear interpolation does between them
    (a cell with a twist bulges at its centre), so the samples are a lattice
    `per_cell` times finer than the grid.  The triangulation also fits the
    surface along the outline, at the lattice spacing.

    Parameters
    ----------
    grid : SurfaceGrid
        The surface to sample.
    poly : shapely.Polygon
        Basin outline (NZTM).
    per_cell : int
        Samples per grid cell edge.

    Returns
    -------
    points : np.ndarray
        ``(N, 2)`` lattice points inside `poly`.
    spacing : float
        The lattice spacing (m), for sampling the outline.
    """
    # Sample only the cells overlapping the outline's bounding box.
    xmin, ymin, xmax, ymax = poly.bounds
    near = (grid.x >= xmin) & (grid.x <= xmax) & (grid.y >= ymin) & (grid.y <= ymax)
    rows, cols = np.nonzero(near)
    m, n = grid.x.shape
    if len(rows):
        r0, r1 = max(rows.min() - 1, 0), min(rows.max() + 1, m - 1)
        c0, c1 = max(cols.min() - 1, 0), min(cols.max() + 1, n - 1)
    else:
        r0, r1, c0, c1 = 0, m - 1, 0, n - 1
    lattice = np.meshgrid(
        np.linspace(r0, r1, per_cell * (r1 - r0) + 1),
        np.linspace(c0, c1, per_cell * (c1 - c0) + 1),
        indexing="ij",
    )
    # Linear in the index along each axis is bilinear within each cell, the
    # same map the grid interpolates with.
    x = scipy.ndimage.map_coordinates(grid.x, lattice, order=1).ravel()
    y = scipy.ndimage.map_coordinates(grid.y, lattice, order=1).ravel()
    inside = shapely.contains_xy(poly, x, y)

    cell_edge = min(
        np.median(np.hypot(np.diff(grid.x, axis=axis), np.diff(grid.y, axis=axis)))
        for axis in (0, 1)
    )
    return np.c_[x[inside], y[inside]], float(cell_edge / per_cell)


def triangulate_polygon(
    poly: shapely.Polygon,
    surfaces: list[tuple[SurfaceGrid, float]],
    max_edge: float | None = None,
    per_cell: int = SAMPLES_PER_CELL,
) -> Triangulation:
    """Triangulate `poly` to fit each surface within its vertical tolerance.

    Builds an error-bounded TIN (see ``nzcvm.adaptive_tin``): the linear
    interpolant of every surface stays within its tolerance (m) at each of
    its :func:`surface_samples` and along the outline.  Vertices sit on
    samples, so triangles go only where a surface bends.

    Parameters
    ----------
    poly : shapely.Polygon
        Basin outline (NZTM). The triangulation keeps its vertices and splits
        its edges where a surface needs it.
    surfaces : list of (SurfaceGrid, float)
        Each surface to fit, with its vertical tolerance in metres.
    max_edge : float, optional
        Longest edge (m) to allow, however flat the surfaces are.
    per_cell : int
        Samples per grid cell edge (see :func:`surface_samples`).

    Returns
    -------
    Triangulation
        The triangles covering `poly`.
    """
    tin_surfaces = [
        (grid.inner, *surface_samples(grid, poly, per_cell), tolerance)
        for grid, tolerance in surfaces
    ]
    boundary = np.asarray(poly.exterior.coords, dtype=np.float64)[:-1]
    vertices, triangles = adaptive_tin(boundary, tin_surfaces, max_edge)
    print(
        f"Triangulated basin into {len(triangles)} triangles "
        f"on {len(vertices)} vertices"
    )
    return Triangulation(vertices, triangles)


def interpolate_surface(
    surface: SurfaceGrid,
    vertices: np.ndarray,
    label: str = "surface",
) -> np.ndarray:
    """Interpolate a surface bilinearly onto mesh vertices, warning on extrapolation."""
    result, outside = surface(vertices[:, 0], vertices[:, 1])
    num_extrapolated = int(np.count_nonzero(outside))
    if num_extrapolated:
        print(
            f"Warning: {num_extrapolated} of {len(result)} vertices fell "
            f"outside the extent of the {label} and were filled by "
            "extending its edge."
        )
    return result


def enforce_mesh_constraints(mesh_top, mesh_bottom):
    bottom_nan_mask = np.isnan(mesh_bottom)
    top_nan_mask = np.isnan(mesh_top)
    invalid_mask = bottom_nan_mask & top_nan_mask
    if invalid_mask.any():
        idx = np.arange(len(mesh_top))
        nan_top_values = idx[invalid_mask]
        nan_bottom_values = idx[invalid_mask]
        e = ValueError(
            "Invalid interpolation result, nan values in both top and bottom surfaces"
        )
        e.add_note(f"{nan_top_values=}\n{nan_bottom_values=}")
        raise e
    if bottom_nan_mask.any():
        print("Warning: bottom surface has nan values. Will crimp to top-level.")
        mesh_bottom[bottom_nan_mask] = mesh_top[bottom_nan_mask]
    if top_nan_mask.any():
        print("Warning: top surface has nan values. Will crimp to bottom-level.")
        mesh_top[top_nan_mask] = mesh_bottom[top_nan_mask]
    overlap_mask = mesh_top > mesh_bottom
    if overlap_mask.any():
        print("Warning: top surface dips below bottom (basement) surface")
        mesh_top[overlap_mask] = mesh_bottom[overlap_mask]
    return mesh_top, mesh_bottom


def uniform_model(
    rho: float,
    vp: float,
    vs: float,
    qp: float,
    qs: float,
    layer_bottoms: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Return a profile with the same qualities at every depth.

    Without *layer_bottoms* the profile is one layer that covers any depth.
    With them, it's one layer per bottom depth, from the topography down,
    laid out as :func:`read_layered_model` would read the same layers from a
    file. ``--layered`` slices a mesh at each layer, so the layers give the
    mesh vertical resolution for a later pass (``banks.py``) to rewrite.
    """
    if layer_bottoms is not None:
        bottom_depth = np.asarray(layer_bottoms, dtype=float)
        if bottom_depth.size == 0 or np.any(np.diff(bottom_depth) <= 0):
            raise ValueError("Layer bottoms must be non-empty and increasing.")
        top_depth = np.concatenate(([0.0], bottom_depth[:-1]))
        return pd.DataFrame(
            {
                "vp": vp,
                "vs": vs,
                "rho": rho,
                "qp": qp,
                "qs": qs,
                "bottom_depth": bottom_depth,
                "z": top_depth,
                "thickness": bottom_depth - top_depth,
            }
        )
    return pd.DataFrame(
        {
            "z": [-1e6],
            "thickness": [1e7],
            "rho": rho,
            "vs": vs,
            "vp": vp,
            "qp": qp,
            "qs": qs,
        }
    )


def read_layered_model(layered_model_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        layered_model_path,
        header=None,
        skiprows=1,
        sep=r"\s+",
        comment="#",
        names=["vp", "vs", "rho", "qp", "qs", "bottom_depth"],
    )
    for col in ["vp", "vs", "rho", "bottom_depth"]:
        df[col] *= 1000.0
    bottom_depth = df["bottom_depth"].to_numpy()
    df["z"] = np.concatenate(([0.0], bottom_depth[:-1]))
    df["thickness"] = bottom_depth - df["z"]

    return df


def volumes(vertices: np.ndarray, tetra: np.ndarray) -> np.ndarray:
    v0 = vertices[tetra[:, 0]]
    v = vertices[tetra[:, 1:]]
    return 1 / 6 * np.abs(np.linalg.det(v - v0[:, np.newaxis, :]))


def cull_mesh(
    vertices: np.ndarray, tetra: np.ndarray, culling_volume: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Drop tetrahedra at or below `culling_volume`, and the vertices left unused.

    Returns the kept vertices, the re-indexed kept tetrahedra, a mask of the
    kept vertices, and the number of tetrahedra dropped.
    """
    volume = volumes(vertices, tetra)
    has_non_zero_volume = volume > culling_volume
    non_zero_tetra = tetra[has_non_zero_volume]
    has_vertex_neighbours = np.zeros(vertices.shape[0], dtype=np.bool_)
    has_vertex_neighbours[non_zero_tetra.ravel()] = 1
    vertex_map = np.cumulative_sum(
        has_vertex_neighbours.astype(np.uint64), include_initial=True
    )
    culled_vertices = vertices[has_vertex_neighbours]
    culled_tetra = vertex_map[non_zero_tetra]
    assert (
        len(culled_vertices) == 0
        or volumes(culled_vertices, culled_tetra).min() >= culling_volume
    )
    return (
        culled_vertices,
        culled_tetra,
        has_vertex_neighbours,
        int(len(tetra) - np.sum(has_non_zero_volume)),
    )


def prism_tetra(triangles: np.ndarray) -> np.ndarray:
    """Split the prism under each triangle into three tetrahedra.

    Vertex ``2 n`` is the top of column ``n`` and ``2 n + 1`` its bottom (see
    :func:`interleave_top_and_bottom`).  Sorting each triangle's vertices
    makes neighbouring prisms split their shared face along the same
    diagonal, so the tetrahedra conform.
    """
    i, j, k = 2 * np.sort(triangles, axis=1).T
    return np.concatenate(
        (
            np.stack((i, i + 1, j + 1, k + 1), axis=1),
            np.stack((i, j, j + 1, k), axis=1),
            np.stack((i, j + 1, k, k + 1), axis=1),
        )
    )


def profile_mesh(
    name: str,
    geometry: shapely.Geometry,
    triangulation: Triangulation,
    topography: np.ndarray,
    top_surface: np.ndarray,
    bottom_surface: np.ndarray,
    model: pd.DataFrame,
    culling_volume: float,
    pad_top: bool,
    priority: int,
) -> TetrahedralMesh:
    """Fill the space between two surfaces with one layer of depth-profile tetrahedra.

    Every tetrahedron evaluates `model` at the query point's depth below the
    topography, so the profile's layer interfaces don't need tetrahedra of
    their own: one prism column per triangle spans the whole basin.  As in
    the layered meshes, the basin stops at the bottom of the model's last
    layer.

    Raises
    ------
    ValueError
        If the model's layer tops decrease with depth.
    """
    tops = model["z"].to_numpy(dtype=np.float32)
    # A layer whose top equals the next one's (in float32) spans no depth.
    keep = np.r_[tops[:-1] != tops[1:], True]
    model, tops = model[keep], tops[keep]
    if np.any(np.diff(tops[1:]) <= 0):
        raise ValueError("1-D model layer tops must increase with depth")

    model_bottom = topography + (model["z"].iloc[-1] + model["thickness"].iloc[-1])
    bottom_surface = np.maximum(np.minimum(bottom_surface, model_bottom), top_surface)
    if pad_top:
        # Raise the top where the basin has thickness so topography slightly
        # above the top surface still lands inside the mesh.  The first
        # profile layer extends upwards to cover it.
        top_surface = np.where(
            bottom_surface > top_surface, top_surface - PAD_DEPTH, top_surface
        )
    columns = triangulation.vertices
    vertices = interleave_top_and_bottom(
        np.c_[columns, top_surface], np.c_[columns, bottom_surface]
    )
    vertices, tetra, kept, zero_volume = cull_mesh(
        vertices, prism_tetra(triangulation.triangles), culling_volume
    )
    print(
        f"Culled {zero_volume} degenerate tetrahedra, "
        f"leaving {len(tetra)} tetrahedra on {len(vertices)} vertices"
    )

    depth = vertices[:, 2] - np.repeat(topography, 2)[kept]
    # The layer each vertex lies in, matching the Rust `DepthProfile::at`: a
    # depth on a boundary belongs to the deeper layer.
    vertex_layer = np.searchsorted(tops[1:], depth, side="right")
    fields = ("rho", "vp", "vs", "qp", "qs")
    profile = {f: model[f].to_numpy(dtype=np.float32) for f in fields}
    return make_mesh(
        name=name,
        points=vertices,
        connectivity=tetra,
        cell_data={
            "model_type": np.full(len(tetra), ModelType.DEPTH_PROFILE, dtype=np.uint8),
            "priority": np.full(len(tetra), priority, dtype=np.uint8),
        },
        geometry=geometry,
        # The vertex qualities are only for display: depth-profile
        # tetrahedra read just the vertex alpha.
        field_data={f: profile[f][vertex_layer] for f in fields}
        | {"alpha": np.ones(len(vertices), dtype=np.float32)},
        depth_profile={"depth": depth.astype(np.float32), "profile_top": tops}
        | {f"profile_{f}": profile[f] for f in fields},
    )


def slice_with_model(
    triangulation: Triangulation,
    topography: np.ndarray,
    top_surface: np.ndarray,
    bottom_surface: np.ndarray,
    model: pd.DataFrame,
    culling_volume: float,
    pad_top: bool,
) -> list[Layer]:
    layers = []
    tetra = prism_tetra(triangulation.triangles)
    emitted_layer = False
    for _, row in model.iterrows():
        z = row["z"]
        print(f"Layer starting at z={z:3f}m")
        if np.all(top_surface >= bottom_surface):
            print("Stopping because the basin is filled to the basement everywhere")
            break
        bottom_depth = topography + (row["z"] + row["thickness"])
        if np.all(bottom_depth < top_surface):
            print("Skipping layer because bottom depth does not cut the top surface")
            continue
        bottom_of_layer = np.maximum(
            np.minimum(bottom_depth, bottom_surface), top_surface
        )

        top_of_layer = top_surface
        if pad_top and not emitted_layer:
            has_thickness = bottom_of_layer > top_surface
            top_of_layer = np.where(has_thickness, top_surface - PAD_DEPTH, top_surface)

        mesh_top = np.c_[triangulation.vertices, top_of_layer]
        mesh_bottom = np.c_[triangulation.vertices, bottom_of_layer]

        mesh_vertices = interleave_top_and_bottom(mesh_top, mesh_bottom)
        mesh_vertices, mesh_tetra, kept, zero_volume = cull_mesh(
            mesh_vertices, tetra, culling_volume
        )
        no_neighbours = int(np.count_nonzero(~kept))
        if zero_volume or no_neighbours:
            print(
                f"Found {zero_volume} degenerate tetra in this layer and {no_neighbours} vertices to remove"
            )
            print(f"{len(mesh_vertices)=}, {len(mesh_tetra)=}")

        if len(mesh_vertices) and len(mesh_tetra):
            layers.append(
                Layer(
                    vertices=mesh_vertices,
                    tetra=mesh_tetra,
                    rho=row["rho"],
                    vp=row["vp"],
                    vs=row["vs"],
                    qp=row["qp"],
                    qs=row["qs"],
                )
            )
            emitted_layer = True
        else:
            print("Skipping layer as no vertices met culling criteria")
            continue
        top_surface = bottom_of_layer
    return layers


def _read_compressed_shapely_wkb(path: Path) -> shapely.Geometry:
    with gzip.open(path) as handle:
        return shapely.from_wkb(handle.read())


def retain_connected(
    poly: shapely.Geometry, internal_poly: shapely.Polygon
) -> shapely.Polygon:
    if isinstance(poly, shapely.Polygon):
        return poly
    elif isinstance(poly, shapely.MultiPolygon):
        return shapely.union_all(
            [geom for geom in poly.geoms if shapely.intersects(internal_poly, geom)]
        )
    else:
        raise TypeError(f"Invalid geometry: {poly!r}")


@app.command()
def validate(mesh_paths: list[Path]) -> None:
    total_size = 0
    for mesh_path in mesh_paths:
        with xr.open_dataset(mesh_path) as dset:
            mesh = TetrahedralMeshSchema.from_dataset(dset)
            points = np.c_[mesh.x, mesh.y, mesh.z]
            connectivity = mesh.connectivity.values
            tetra_volumes = volumes(points, connectivity)
            print(f"{mesh_path.stem} Minimum tetra volume: {tetra_volumes.min()}")
            dset_size = dset.nbytes
            total_size += dset_size
            print(f"dataset size = {dset_size / (1024**2):.1f}M")
            if (tetra_volumes < 1e-6).any():
                sys.exit(-1)
    print(f"Total dataset in-memory requirements: {total_size / (1024**3):.1f}G")


@app.command()
def main(
    bounds: Annotated[
        Path,
        typer.Argument(
            help="GeoJSON file defining the basin boundary.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    topography: Annotated[
        Path,
        typer.Argument(
            help="Topography surface file (HDF5).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    top_surface: Annotated[
        Path,
        typer.Argument(
            help="Top surface file (HDF5).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    bottom_surface: Annotated[
        Path,
        typer.Argument(
            help="Bottom surface file (HDF5).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[Path, typer.Argument(help="Output VTKHDF file path.")],
    simplification: Annotated[
        float,
        typer.Option(
            "-s",
            "--simplification",
            help="Polygon simplification parameter (higher = simpler).",
            min=0.0,
        ),
    ] = 10.0,
    culling_volume: Annotated[
        float,
        typer.Option(
            "-v",
            "--culling-volume",
            help="Tetrahedron volume cull threshold (lower = finer).",
            min=0.0,
        ),
    ] = 1e-3,
    tolerance: Annotated[
        float,
        typer.Option(
            "-t",
            "--tolerance",
            help="Largest vertical error (m) the triangulation may leave in "
            "each surface, checked on a lattice twice as fine as the surface's "
            "grid and along the outline (lower = finer).",
            min=0.0,
        ),
    ] = TOLERANCE,
    smoothing: Annotated[
        float,
        typer.Option(
            "-S",
            "--smoothing",
            help="Smoothing boundary distance for model.",
            min=0.0,
        ),
    ] = 0.0,
    coastline: Annotated[
        Path | None,
        typer.Option(
            "--coastline",
            help="Coastline to clip smoothing boundary",
        ),
    ] = None,
    vm_1d: Annotated[
        Path | None,
        typer.Option(
            help="Path to 1-D velocity model CSV.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    rho: Annotated[
        float | None, typer.Option(help="Constant density (kg/m³).", min=0.0)
    ] = None,
    vp: Annotated[
        float | None, typer.Option(help="Constant P-wave velocity (m/s).", min=0.0)
    ] = None,
    vs: Annotated[
        float | None, typer.Option(help="Constant S-wave velocity (m/s).", min=0.0)
    ] = None,
    qp: Annotated[
        float | None, typer.Option(help="Constant P-wave attenuation factor.", min=0.0)
    ] = None,
    qs: Annotated[
        float | None, typer.Option(help="Constant S-wave attenuation factor.", min=0.0)
    ] = None,
    layer_bottoms: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated layer bottom depths (m) that split the constant "
            "--rho/--vp/--vs/--qp/--qs model into layers, for --layered."
        ),
    ] = None,
    priority: Annotated[
        int,
        typer.Option(
            help="Basin mesh priority (lower = higher priority).", min=0, max=255
        ),
    ] = 0,
    pad_top: Annotated[
        bool,
        typer.Option(
            help="If true, pad the top layer (useful to prevent topography peek through)."
        ),
    ] = True,
    layered: Annotated[
        bool,
        typer.Option(
            help="Slice the basin into one tetrahedral layer per 1-D model layer "
            "and interpolate vertex qualities, instead of evaluating the model "
            "as a depth profile.  Many times larger; only needed when the "
            "vertex qualities are rewritten afterwards (see banks.py)."
        ),
    ] = False,
) -> None:
    """Entry point for the ``nzcvm construct-mesh`` command."""

    if vm_1d is not None and layer_bottoms is not None:
        raise typer.BadParameter("--layer-bottoms only applies to a constant model.")
    if vm_1d is not None:
        model = read_layered_model(vm_1d)
    elif (
        rho is not None
        and vp is not None
        and vs is not None
        and qp is not None
        and qs is not None
    ):
        model = uniform_model(
            rho,
            vp,
            vs,
            qp,
            qs,
            layer_bottoms=None
            if layer_bottoms is None
            else [float(depth) for depth in layer_bottoms.split(",")],
        )
    else:
        raise typer.BadParameter(
            "Provide either --vm-1d, or all of --rho, --vp, --vs, --qp and --qs."
        )

    collection = shapely.from_geojson(bounds.read_text())
    internal_poly = preprocess_polygon(collection.geoms[0]).simplify(simplification)
    shapely.prepare(internal_poly)
    will_smooth = smoothing > 0 and coastline
    if will_smooth:
        coastline_poly = _read_compressed_shapely_wkb(coastline)
        buffer = shapely.buffer(
            shapely.difference(internal_poly, coastline_poly), smoothing
        )
        offshore_smoothing = shapely.difference(buffer, coastline_poly)
        poly = shapely.union(internal_poly, offshore_smoothing)
        poly = retain_connected(poly, internal_poly)
    else:
        poly = internal_poly

    shapely.prepare(poly)

    top_bbox = get_surface_bounds(top_surface)
    bottom_bbox = get_surface_bounds(bottom_surface)
    (min_x, min_y, max_x, max_y) = shapely.bounds(
        shapely.buffer(shapely.intersection(top_bbox, bottom_bbox), 10000.0)
    )
    corners = np.array(list(itertools.product((min_x, max_x), (min_y, max_y)))).T
    corners_x = corners[0]
    corners_y = corners[1]
    corners_lon, corners_lat = INV_TRANSFORMER.transform(corners_x, corners_y)

    buffered_bbox = (
        corners_lat.min(),
        corners_lat.max(),
        corners_lon.min(),
        corners_lon.max(),
    )

    topo_grid = read_surface_file(topography, bbox=buffered_bbox)
    top_grid = read_surface_file(top_surface, bbox=buffered_bbox)
    bot_grid = read_surface_file(bottom_surface, bbox=buffered_bbox)
    surfaces = [(bot_grid, tolerance)]
    if top_surface != bottom_surface:
        surfaces.append((top_grid, tolerance))
    # Depth below the topography picks the profile layer, so the topography
    # matters only when the model has more than one layer.
    if topography not in (top_surface, bottom_surface) and (layered or len(model) > 1):
        surfaces.append((topo_grid, tolerance))
    # When smoothing, cap the edge length to keep a decent smoothing
    # resolution.  Otherwise, only the tolerance limits it.
    max_edge = SMOOTHING_MAX_EDGE if will_smooth else None
    triangulation = triangulate_polygon(poly, surfaces, max_edge)
    mesh_top = interpolate_surface(
        top_grid, triangulation.vertices, label="top surface"
    )
    mesh_bottom = interpolate_surface(
        bot_grid, triangulation.vertices, label="bottom surface"
    )
    mesh_top, mesh_bottom = enforce_mesh_constraints(mesh_top, mesh_bottom)
    mesh_topography = interpolate_surface(
        topo_grid, triangulation.vertices, label="topography"
    )

    name = output.stem
    if layered:
        print("Slicing model into volumetric mesh")
        layers = slice_with_model(
            triangulation,
            mesh_topography,
            mesh_top,
            mesh_bottom,
            model,
            culling_volume,
            pad_top=pad_top,
        )
        mesh = construct_volumetric_mesh(name, poly, layers, priority)
    else:
        print("Filling basin with depth-profile tetrahedra")
        mesh = profile_mesh(
            name,
            poly,
            triangulation,
            mesh_topography,
            mesh_top,
            mesh_bottom,
            model,
            culling_volume,
            pad_top=pad_top,
            priority=priority,
        )

    if smoothing > 0:
        print("Applying smoothing boundary")

        points_2d = shapely.points(mesh.x, mesh.y)
        distances = shapely.distance(internal_poly, points_2d)
        alpha = np.interp(
            distances,
            np.array([0.0, smoothing], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        )
        mesh["alpha"].loc[:] = alpha

    print(mesh)
    mesh.to_zarr(output, mode="w", encoding=mesh_encoding(mesh))
    nbytes = sum(f.stat().st_size for f in output.rglob("*") if f.is_file())
    print(f"Saved model with size {nbytes / (1024**2):.1f} MB")
