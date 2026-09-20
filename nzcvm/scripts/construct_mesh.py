"""Construct a tetrahedral volumetric mesh for a basin model using adaptive topography gradient logic."""

import gzip
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TextIO

import h5py
import numpy as np
import pandas as pd
import pyproj
import scipy as sp
import shapely
import shapely.ops
import typer
import xarray as xr

from nzcvm.models.mesh import (
    DEFAULT_ENCODING_SETTINGS,
    TetrahedralMesh,
    TetrahedralMeshSchema,
    make_mesh,
)

TRANSFORMER = pyproj.Transformer.from_crs(4326, 2193, always_xy=True)
INV_TRANSFORMER = pyproj.Transformer.from_crs(2193, 4326, always_xy=True)

PAD_DEPTH = 50.0

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
class PolyData:
    vertices: np.ndarray
    segments: np.ndarray


@dataclass
class Triangulation:
    vertices: np.ndarray
    triangles: np.ndarray


def extract_poly_data(poly: shapely.Polygon) -> PolyData:
    vertices = np.array(poly.exterior.coords)[:-1]
    idx = np.arange(len(vertices), dtype=np.int64)
    segments = np.stack((idx, (idx + 1) % len(vertices)), axis=1) + 1
    return PolyData(vertices, segments)


def write_poly_file(poly: PolyData, buffer: TextIO) -> None:
    buffer.write(f"{len(poly.vertices)} 2 0 0\n")
    vertex_columns = np.column_stack((np.arange(len(poly.vertices)) + 1, poly.vertices))
    np.savetxt(buffer, vertex_columns, ["%d", "%10.5f", "%10.5f"])
    buffer.write(f"{len(poly.segments)} 0\n")
    segment_columns = np.column_stack(
        (np.arange(len(poly.segments)) + 1, poly.segments)
    )
    np.savetxt(buffer, segment_columns, "%d")
    buffer.write("0")


def read_vertices(handle: TextIO | Path) -> np.ndarray:
    return np.genfromtxt(
        handle,
        skip_header=1,
        dtype=[
            ("vertex", np.int64),
            ("x", np.float64),
            ("y", np.float64),
            ("boundary", np.int64),
        ],
    )


def read_triangles(handle: TextIO | Path) -> np.ndarray:
    return np.genfromtxt(
        handle,
        skip_header=1,
        dtype=[
            ("vertex", np.int64),
            ("i", np.int64),
            ("j", np.int64),
            ("k", np.int64),
        ],
    )


def construct_mesh_tetra(triangles: np.ndarray) -> np.ndarray:
    i = 2 * (triangles["i"] - 1)
    i1 = i + 1
    j = 2 * (triangles["j"] - 1)
    j1 = j + 1
    k = 2 * (triangles["k"] - 1)
    k1 = k + 1
    tets_1 = np.stack((i, i1, j1, k1), axis=1)
    tets_2 = np.stack((i, j, j1, k), axis=1)
    tets_3 = np.stack((i, j1, k, k1), axis=1)
    return np.concatenate((tets_1, tets_2, tets_3))


def interleave_top_and_bottom(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    interleaved = np.zeros((2 * len(top), 3), dtype=np.float32)
    interleaved[::2] = top
    interleaved[1::2] = bottom
    return interleaved


def read_surface_file(
    surface_path: Path, bbox: tuple[float, float, float, float] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(surface_path, "r") as f:
        latitude = f["latitude"][:]
        longitude = f["longitude"][:]

        if bbox is not None:
            min_lat, max_lat, min_lon, max_lon = bbox
            lat_idxs = np.where((latitude >= min_lat) & (latitude <= max_lat))[0]
            lon_idxs = np.where((longitude >= min_lon) & (longitude <= max_lon))[0]

            if len(lat_idxs) > 0 and len(lon_idxs) > 0:
                lat_slice = slice(lat_idxs.min(), lat_idxs.max() + 1)
                lon_slice = slice(lon_idxs.min(), lon_idxs.max() + 1)
                latitude = latitude[lat_slice]
                longitude = longitude[lon_slice]
                elevation = f["elevation"][lat_slice, lon_slice]
            else:
                elevation = f["elevation"][:]
        else:
            elevation = f["elevation"][:]

    elevation *= -1
    x_lon, x_lat = np.meshgrid(longitude, latitude)
    x, y = TRANSFORMER.transform(x_lon, x_lat)
    return x, y, elevation


class LinearNDInterpolatorExt:
    """Linear interpolator with nearest-neighbour fallback outside the hull.

    Construction drops NaN input values before building the interpolators, so
    that an undefined source node doesn't contaminate the linear interpolation
    of its valid neighbours. :attr:`num_extrapolated` counts the query points
    that fell outside the convex hull of the valid input points on the most
    recent call and so took a nearest-neighbour extrapolation instead, which
    lets callers report silent extrapolation.
    """

    def __init__(self, points, values):
        points = np.asarray(points)
        values = np.asarray(values)
        valid = ~np.isnan(values)
        if not valid.all():
            points = points[valid]
            values = values[valid]
        self.funcinterp = sp.interpolate.LinearNDInterpolator(points, values)
        self.funcnearest = sp.interpolate.NearestNDInterpolator(points, values)
        self.num_extrapolated = 0

    def __call__(self, *args):
        t = self.funcinterp(*args)
        nan_mask = np.isnan(t)
        self.num_extrapolated = int(np.count_nonzero(nan_mask))
        if self.num_extrapolated:
            t_n = self.funcnearest(*args)
            t[nan_mask] = t_n[nan_mask]
        return t


def gradient_field(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, error_target: float, max_h: float
) -> np.ndarray:
    """Compute an unstructured mesh sizing field from topography spatial gradients."""
    del_y, del_x = np.gradient(z)

    dx = np.gradient(x, axis=1)
    dy = np.gradient(y, axis=0)
    dx[dx == 0] = 1.0
    dy[dy == 0] = 1.0

    grad_x = del_x / dx
    grad_y = del_y / dy
    grad_mag = np.hypot(grad_x, grad_y)

    # Characteristic local grid spacing (metres)
    cell_size = np.sqrt(np.abs(dx * dy))  # geometric mean of dx, dy

    with np.errstate(divide="ignore", invalid="ignore"):
        h_field = (error_target * cell_size) / (grad_mag + 1e-8)

    min_size = cell_size
    return np.clip(h_field, min_size, max_h)


def compute_topography_sizing_field(
    *fields: tuple[np.ndarray, np.ndarray, np.ndarray],
    bin_size: float = 100.0,
) -> LinearNDInterpolatorExt:
    """Build a sizing field interpolator from one or more gradient fields.

    Where fields overlap, the minimum h (finest resolution) wins.
    Binning the points at `bin_size` metres resolution before the reduction
    identifies near-coincident points across grids.

    Example:
        coarse = gradient_field(x1, y1, z1, error_target=0.01, max_h=50000)
        fine   = gradient_field(x2, y2, z2, error_target=0.01, max_h=1000)
        interp = compute_topography_sizing_field(coarse, fine, bin_size=250.0)
    """
    all_x = np.concatenate([f[0] for f in fields])
    all_y = np.concatenate([f[1] for f in fields])
    all_h = np.concatenate([f[2] for f in fields])

    # Bin coordinates to identify near-coincident points across grids
    binned_x = np.round(all_x / bin_size).astype(np.int64)
    binned_y = np.round(all_y / bin_size).astype(np.int64)

    # Take minimum h per bin using a dict reduction
    bin_map: dict[tuple[int, int], float] = {}
    for bx, by, h in zip(binned_x, binned_y, all_h):
        key = (bx, by)
        if key not in bin_map or h < bin_map[key]:
            bin_map[key] = h

    # Reconstruct arrays from reduced bins (convert keys back to metres)
    keys = np.array(list(bin_map.keys()), dtype=np.float64)
    reduced_x = keys[:, 0] * bin_size
    reduced_y = keys[:, 1] * bin_size
    reduced_h = np.array(list(bin_map.values()))

    points = np.stack((reduced_x, reduced_y), axis=1)
    q0, q50, q100 = np.quantile(reduced_h, [0, 0.5, 1.0])
    print(f"Refinement field (min, median, max) = ({q0:.1f}, {q50:.1f}, {q100:.1f})")
    return LinearNDInterpolatorExt(points, reduced_h)


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
        [np.full((len(layer.tetra),), 1, dtype=np.uint8) for layer in layers]
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


def triangulate_polygon(
    poly: shapely.Polygon, sizing_field: LinearNDInterpolatorExt
) -> Triangulation:
    import gmsh

    gmsh.initialize()
    # gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("surface_triangulation")

    coords = np.array(poly.exterior.coords)[:-1]

    # Map adaptive local sizing to boundary vertices
    point_tags = []
    for x, y in coords:
        r_local = float(sizing_field([[x, y]])[0])
        tag = gmsh.model.geo.addPoint(x, y, 0.0, meshSize=r_local)
        point_tags.append(tag)

    line_tags = []
    num_points = len(point_tags)
    for i in range(num_points):
        p1 = point_tags[i]
        p2 = point_tags[(i + 1) % num_points]
        tag = gmsh.model.geo.addLine(p1, p2)
        line_tags.append(tag)

    cl = gmsh.model.geo.addCurveLoop(line_tags)
    surface_tag = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()

    # Callback evaluation loop providing Gmsh with local terrain constraints during triangulation
    def mesh_size_callback(dim, tag, x, y, z, lc):
        return float(sizing_field([[x, y]])[0])

    gmsh.model.mesh.setSizeCallback(mesh_size_callback)
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(2)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes(
        2, surface_tag, includeBoundary=True
    )
    node_coords = node_coords.reshape(-1, 3)

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, surface_tag)

    tri_type_idx = list(elem_types).index(2)
    tri_node_tags = elem_node_tags[tri_type_idx].reshape(-1, 3)
    tri_tags = elem_tags[tri_type_idx]

    gmsh.finalize()

    num_nodes = len(node_tags)
    sort_idx = np.argsort(node_tags)
    sorted_tags = node_tags[sort_idx]
    sorted_coords = node_coords[sort_idx]

    vertices = np.empty(
        num_nodes,
        dtype=[
            ("vertex", np.int64),
            ("x", np.float64),
            ("y", np.float64),
            ("boundary", np.int64),
        ],
    )
    vertices["vertex"] = np.arange(1, num_nodes + 1)
    vertices["x"] = sorted_coords[:, 0]
    vertices["y"] = sorted_coords[:, 1]
    vertices["boundary"] = 0

    local_tri_nodes = np.searchsorted(sorted_tags, tri_node_tags) + 1

    num_triangles = len(tri_tags)
    triangles = np.empty(
        num_triangles,
        dtype=[
            ("vertex", np.int64),
            ("i", np.int64),
            ("j", np.int64),
            ("k", np.int64),
        ],
    )
    triangles["vertex"] = np.arange(1, num_triangles + 1)
    triangles["i"] = local_tri_nodes[:, 0]
    triangles["j"] = local_tri_nodes[:, 1]
    triangles["k"] = local_tri_nodes[:, 2]

    return Triangulation(vertices, triangles)


def interpolate_surface(
    surface: np.ndarray,
    vertices: np.ndarray,
    label: str = "surface",
) -> np.ndarray:
    interp = LinearNDInterpolatorExt(surface[:, :-1], surface[:, -1])
    result = interp(np.c_[vertices["x"], vertices["y"]])
    if interp.num_extrapolated:
        print(
            f"Warning: {interp.num_extrapolated} of {len(result)} vertices fell "
            f"outside the extent of the {label} and were filled by "
            "nearest-neighbour extrapolation."
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
    rho: float, vp: float, vs: float, qp: float, qs: float
) -> pd.DataFrame:
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
) -> tuple[np.ndarray, np.ndarray, int, int]:
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
        int(len(vertices) - np.sum(has_vertex_neighbours)),
        int(len(tetra) - np.sum(has_non_zero_volume)),
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
    tri_verts = np.stack(
        [
            triangulation.triangles["i"],
            triangulation.triangles["j"],
            triangulation.triangles["k"],
        ],
        axis=1,
    )
    tri_verts.sort(axis=1)

    sorted_triangles = np.empty(len(tri_verts), dtype=triangulation.triangles.dtype)
    sorted_triangles["i"] = tri_verts[:, 0]
    sorted_triangles["j"] = tri_verts[:, 1]
    sorted_triangles["k"] = tri_verts[:, 2]

    tetra = construct_mesh_tetra(sorted_triangles)
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

        mesh_top = np.c_[
            triangulation.vertices["x"],
            triangulation.vertices["y"],
            top_of_layer,
        ]
        mesh_bottom = np.c_[
            triangulation.vertices["x"], triangulation.vertices["y"], bottom_of_layer
        ]

        mesh_vertices = interleave_top_and_bottom(mesh_top, mesh_bottom)
        mesh_vertices, mesh_tetra, no_neighbours, zero_volume = cull_mesh(
            mesh_vertices, tetra, culling_volume
        )
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


def mask_surface(
    bounds: shapely.Polygon, surface_points: np.ndarray, buffer: float
) -> np.ndarray:
    mask = shapely.contains_xy(
        bounds.buffer(buffer), surface_points[:, 0], surface_points[:, 1]
    )
    return surface_points[mask]


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
    error_target: Annotated[
        float,
        typer.Option(
            "-e",
            "--error-target",
            help="Geometric error target for adaptive topography triangulation (e.g. 0.01 for 1%).",
            min=0.0,
        ),
    ] = 0.01,
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
) -> None:
    """Entry point for the ``nzcvm construct-mesh`` command."""

    if vm_1d is not None:
        model = read_layered_model(vm_1d)
    elif (
        rho is not None
        and vp is not None
        and vs is not None
        and qp is not None
        and qs is not None
    ):
        model = uniform_model(rho, vp, vs, qp, qs)
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

    topo_x, topo_y, topo_z = read_surface_file(topography, bbox=buffered_bbox)
    top_x, top_y, top_z = read_surface_file(top_surface, bbox=buffered_bbox)
    bot_x, bot_y, bot_z = read_surface_file(bottom_surface, bbox=buffered_bbox)
    print("Computing topography gradient adaptive sizing field...")
    # When smoothing, cap the cell size to 1000.0 km to keep a decent smoothing resolution.
    # Otherwise, the only limit is geometry.
    max_h = 1000.0 if will_smooth else 50000.0
    fields = []
    h_topo = gradient_field(
        topo_x, topo_y, topo_z, max_h=max_h, error_target=error_target
    )
    fields.append((topo_x.ravel(), topo_y.ravel(), h_topo.ravel()))

    if topography != top_surface:
        h_top = gradient_field(
            top_x, top_y, top_z, max_h=max_h, error_target=error_target
        )
        fields.append((top_x.ravel(), top_y.ravel(), h_top.ravel()))
    h_bot = gradient_field(bot_x, bot_y, bot_z, max_h=max_h, error_target=error_target)
    fields.append((bot_x.ravel(), bot_y.ravel(), h_bot.ravel()))
    sizing_field = compute_topography_sizing_field(*fields)

    triangulation = triangulate_polygon(poly, sizing_field)
    top_surface_data = np.c_[top_x.ravel(), top_y.ravel(), top_z.ravel()]
    bottom_surface_data = np.c_[bot_x.ravel(), bot_y.ravel(), bot_z.ravel()]
    topography_data = np.c_[topo_x.ravel(), topo_y.ravel(), topo_z.ravel()]

    mesh_top = interpolate_surface(
        top_surface_data, triangulation.vertices, label="top surface"
    )
    mesh_bottom = interpolate_surface(
        bottom_surface_data, triangulation.vertices, label="bottom surface"
    )
    mesh_top, mesh_bottom = enforce_mesh_constraints(mesh_top, mesh_bottom)
    mesh_topography = interpolate_surface(
        topography_data, triangulation.vertices, label="topography"
    )

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
    name = output.stem
    mesh = construct_volumetric_mesh(name, poly, layers, priority)

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
    mesh.to_zarr(output, mode="w", encoding=DEFAULT_ENCODING_SETTINGS)
    nbytes = output.stat().st_size
    print(f"Saved model with size {nbytes / (1024**2):.1f} MB")
