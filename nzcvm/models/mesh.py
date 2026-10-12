"""Mesh I/O utilities for NZCVM.

Provides dataclass representations and VTKHDF-backed I/O for the two mesh
types used by NZCVM: tetrahedral unstructured grids (velocity model meshes)
and structured grids (topographic surfaces).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum, StrEnum
from typing import Any, Literal

import numpy as np
import shapely
import xarray as xr
from xarray_dataclasses import AsDataset, Attr, Coord, Data, DataOptions
from zarr.codecs import Blosc

from nzcvm.xarray import encode

I = Literal["i"]
J = Literal["j"]
K = Literal["k"]
L = Literal["layer"]
X = Literal["x"]
Y = Literal["y"]


class Vertical(StrEnum):
    """The coordinate a mesh's vertex ``z`` holds (``vertical`` attribute).

    Both run positive down. A query passes each point's elevation and its
    depth below the topography, and a mesh reads the one its ``z`` stores.
    """

    #: Negated elevation, fixed in space: tomography.
    ELEVATION = "elevation"
    #: Depth below the topography, following the terrain: basins.
    DEPTH = "depth"


class ModelType(IntEnum):
    """How a mesh turns a query point into a quality (``model_type``)."""

    #: Barycentric interpolation of the four vertex qualities.
    INTERPOLATE = 1
    #: The mesh's 1-D depth profile, evaluated at the interpolated depth.
    DEPTH_PROFILE = 2


def _encode_geometry_attr(value: Any) -> Any:
    """Turn a live ``shapely.Geometry`` attr into hex-WKB, a JSON-safe string zarr/netcdf can store."""
    if isinstance(value, shapely.Geometry):
        return shapely.to_wkb(value, hex=True)
    return value


def _decode_geometry_attr(value: Any) -> Any:
    """Turn a hex-WKB geometry attr back into a live ``shapely.Geometry``."""
    if isinstance(value, str):
        try:
            return shapely.from_wkb(value)
        except shapely.errors.ShapelyError:
            return value
    return value


class TetrahedralMesh(xr.Dataset):
    __slots__ = ()

    @property
    def has_depth_profile(self) -> bool:
        """Whether the mesh has a depth profile (see :class:`DepthProfileSchema`)."""
        return any(name in self.data_vars for name in DEPTH_PROFILE_VARIABLES)


@dataclass
class DepthProfileSchema(AsDataset):
    """The optional depth-profile variables of a :class:`TetrahedralMesh`.

    Tetrahedra of type :attr:`ModelType.DEPTH_PROFILE` evaluate a 1-D
    profile at the query point's depth below a reference surface (normally
    the topography).  The depth comes from barycentric interpolation of the
    per-vertex ``depth``; layer ``n`` spans ``[profile_top[n],
    profile_top[n + 1])``, the first layer extends upwards without limit and
    the last downwards without limit.
    """

    #: Depth of each vertex below the reference surface (m).
    depth: Data[tuple[I], np.float32]
    #: Depth of the top of each layer (m), strictly increasing.
    profile_top: Data[tuple[L], np.float32]
    profile_rho: Data[tuple[L], np.float32]
    profile_vp: Data[tuple[L], np.float32]
    profile_vs: Data[tuple[L], np.float32]
    profile_qp: Data[tuple[L], np.float32]
    profile_qs: Data[tuple[L], np.float32]

    i: Coord[I, np.uint64]
    layer: Coord[L, np.uint64]


#: Names of the depth-profile data variables.
DEPTH_PROFILE_VARIABLES = tuple(
    f.name for f in fields(DepthProfileSchema) if f.name not in ("i", "layer")
)


@dataclass
class TetrahedralMeshSchema(AsDataset):
    name: Attr[str]
    #: A :class:`Vertical`: what the vertex ``z`` holds.
    vertical: Attr[str]

    x: Data[tuple[I,], np.float32]
    y: Data[tuple[I,], np.float32]
    z: Data[tuple[I,], np.float32]

    connectivity: Data[tuple[J, K], np.uint64]
    priority: Data[tuple[J,], np.uint8]
    model_type: Data[tuple[J,], np.uint8]

    rho: Data[tuple[I,], np.float32]
    vp: Data[tuple[I,], np.float32]
    vs: Data[tuple[I,], np.float32]
    qp: Data[tuple[I,], np.float32]
    qs: Data[tuple[I,], np.float32]
    alpha: Data[tuple[I,], np.float32]

    i: Coord[I, np.uint64]
    j: Coord[J, np.uint64]
    k: Coord[K, np.uint64]

    transform: Attr[np.ndarray | None] = None
    geometry: Attr[shapely.Geometry | None] = None
    __dataoptions__ = DataOptions(TetrahedralMesh)

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> TetrahedralMesh:
        """Parses, validates, and builds a Grid from a standard xr.Dataset.

        The depth-profile variables are optional; when any is present they
        must all be present and :class:`DepthProfileSchema` checks them together.
        """
        dataset = encode(dataset, attr_hook=_decode_geometry_attr)
        profile_names = [name for name in DEPTH_PROFILE_VARIABLES if name in dataset]
        core = dataset.drop_vars([*profile_names, "layer"], errors="ignore")
        mesh = cls.new(**core.data_vars, **core.coords, **core.attrs)  # ty: ignore[invalid-argument-type]
        if not profile_names:
            return mesh
        profile = DepthProfileSchema.new(
            **dataset[profile_names].data_vars,  # ty: ignore[invalid-argument-type]
            i=dataset.i,
            layer=dataset.layer,
        )
        return mesh.merge(profile)


MEDIUM_COMPRESSOR = [Blosc(cname="zstd", clevel=5, shuffle=True)]
HIGH_COMPRESSOR = [Blosc(cname="zstd", clevel=7, shuffle=True)]

#: Compression for each tetrahedral mesh variable.  Use :func:`mesh_encoding`
#: to get the settings for a particular mesh.
ENCODING_SETTINGS = {
    "x": {"compressors": MEDIUM_COMPRESSOR},
    "y": {"compressors": MEDIUM_COMPRESSOR},
    "z": {"compressors": MEDIUM_COMPRESSOR},
    "connectivity": {"compressors": HIGH_COMPRESSOR},
    "priority": {"compressors": MEDIUM_COMPRESSOR},
    "model_type": {"compressors": MEDIUM_COMPRESSOR},
    "rho": {"compressors": HIGH_COMPRESSOR},
    "vp": {"compressors": HIGH_COMPRESSOR},
    "vs": {"compressors": HIGH_COMPRESSOR},
    "qp": {"compressors": HIGH_COMPRESSOR},
    "qs": {"compressors": HIGH_COMPRESSOR},
    "alpha": {"compressors": HIGH_COMPRESSOR},
    "depth": {"compressors": MEDIUM_COMPRESSOR},
    **{
        name: {"compressors": HIGH_COMPRESSOR}
        for name in DEPTH_PROFILE_VARIABLES
        if name.startswith("profile_")
    },
}


def mesh_encoding(mesh: xr.Dataset) -> dict[str, dict[str, Any]]:
    """Return the ``to_zarr`` encoding for the variables `mesh` has.

    Parameters
    ----------
    mesh : xr.Dataset
        The mesh to write.

    Returns
    -------
    dict
        :data:`ENCODING_SETTINGS` restricted to the variables in `mesh`, so
        optional variables a mesh lacks don't make the write fail.
    """
    return {k: v for k, v in ENCODING_SETTINGS.items() if k in mesh.data_vars}


class StructuredMesh(xr.Dataset):
    __slots__ = ()


#: The frame of an upright raster in NZTM: its ``x`` and ``y`` are NZTM's.
UPRIGHT_FRAME = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


@dataclass
class StructuredMeshSchema(AsDataset):
    """A surface sampled on a grid that's regular in its own frame.

    1-D coordinates in the surface's frame index ``z`` as ``(y, x)``.
    ``frame = [a, b, c, d, e, f]`` maps an NZTM point ``(X, Y)`` into it:
    ``x = a X + b Y + c`` and ``y = d X + e Y + f``. An upright raster in
    NZTM uses :data:`UPRIGHT_FRAME`; a rotated one rotates query points into
    its frame. Build one with :func:`surface_mesh`.
    """

    z: Data[tuple[Y, X], np.float32]

    y: Coord[Y, np.float64]
    x: Coord[X, np.float64]

    name: Attr[str]
    frame: Attr[list[float]]

    __dataoptions__ = DataOptions(StructuredMesh)

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> StructuredMesh:
        """Parses, validates, and builds a Grid from a standard xr.Dataset."""
        return cls.new(**dataset.data_vars, **dataset.coords, **dataset.attrs)  # ty: ignore[invalid-argument-type]


def surface_mesh(
    name: str,
    z: Any,
    x: Any,
    y: Any,
    frame: tuple[float, ...] = UPRIGHT_FRAME,
) -> StructuredMesh:
    """Build a :class:`StructuredMesh` from its values and frame coordinates.

    Parameters
    ----------
    name :
        Name of the surface.
    z :
        ``(len(y), len(x))`` values, as a NumPy or Dask array.
    x, y :
        1-D coordinates of the columns and rows, in the surface's frame.
    frame :
        The affine map from NZTM into the frame (see
        :class:`StructuredMeshSchema`).

    Returns
    -------
    StructuredMesh
        The surface.
    """
    return StructuredMeshSchema.new(
        z=z,
        y=np.asarray(y, dtype=np.float64),
        x=np.asarray(x, dtype=np.float64),
        name=name,
        frame=[float(v) for v in frame],
    )


#: Nodes per chunk along each axis of a stored surface. A query reads the
#: chunks around its points, and small chunks mean it reads little.
SURFACE_CHUNK_SIZE = 512

DEFAULT_STRUCTURED_ENCODING_SETTINGS = {"z": {"compressors": MEDIUM_COMPRESSOR}}


def make_mesh(
    name: str,
    vertical: Vertical,
    points: np.ndarray,
    connectivity: np.ndarray,
    cell_data: dict[str, np.ndarray],
    field_data: dict[str, np.ndarray],
    geometry: shapely.Geometry | None,
    depth_profile: dict[str, np.ndarray] | None = None,
) -> TetrahedralMesh:
    """Create a :class:`TetrahedralMesh`.

    Parameters
    ----------
    name :
        Optional human-readable name for the model.  Stored in
        ``field_data["name"]`` when provided so it survives VTKHDF
        round-trips and reaches :func:`~nzcvm.model.MeshModel`.
    vertical :
        Whether vertex ``z`` stores negated elevation or depth below the
        topography.
    points :
        ``(N, 3)`` float32 array of vertex coordinates.
    connectivity :
        ``(M, 4)`` integer array of tetrahedral cell vertex indices.
    cell_data :
        Per-cell arrays: ``model_type`` (one :class:`ModelType` for the
        whole mesh) and ``priority``.
    field_data :
        Per-model arrays such as ``rho``, ``vp``, ``vs``, ``qp``, ``qs``,
        ``alpha`` and ``priority``.
    geometry :
        Optional footprint of the mesh, stored alongside the mesh data.
    depth_profile :
        The :class:`DepthProfileSchema` variables (``depth`` and the
        ``profile_*`` arrays), required when any cell has type
        :attr:`ModelType.DEPTH_PROFILE`.

    Returns
    -------
    TetrahedralMesh
        The assembled mesh.
    """
    i = np.arange(len(points))
    nj, nk = connectivity.shape
    j = np.arange(nj)
    k = np.arange(nk)

    mesh = TetrahedralMeshSchema.new(
        name=name,
        vertical=str(vertical),
        x=points[..., 0],
        y=points[..., 1],
        z=points[..., 2],
        connectivity=connectivity,
        i=i,
        j=j,
        k=k,
        geometry=geometry,
        **cell_data,
        **field_data,
    )
    if depth_profile is not None:
        layer = np.arange(len(depth_profile["profile_top"]))
        mesh = mesh.merge(DepthProfileSchema.new(i=i, layer=layer, **depth_profile))
    return encode(mesh, attr_hook=_encode_geometry_attr)
