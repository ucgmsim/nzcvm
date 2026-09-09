"""Mesh I/O utilities for NZCVM.

Provides dataclass representations and VTKHDF-backed I/O for the two mesh
types used by NZCVM: tetrahedral unstructured grids (velocity model meshes)
and structured grids (topographic surfaces).
"""

from __future__ import annotations

from dataclasses import dataclass
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
    def models(self):
        return self.connectivity


@dataclass
class TetrahedralMeshSchema(AsDataset):
    name: Attr[str]

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
        """Parses, validates, and builds a Grid from a standard xr.Dataset."""
        dataset = encode(dataset, attr_hook=_decode_geometry_attr)
        return cls.new(**dataset.data_vars, **dataset.coords, **dataset.attrs)  # ty: ignore[invalid-argument-type]


MEDIUM_COMPRESSOR = [Blosc(cname="zstd", clevel=5, shuffle=True)]
HIGH_COMPRESSOR = [Blosc(cname="zstd", clevel=7, shuffle=True)]

DEFAULT_ENCODING_SETTINGS = {
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
}


class StructuredMesh(xr.Dataset):
    __slots__ = ()


@dataclass
class StructuredMeshSchema(AsDataset):
    """A structured-grid surface mesh."""

    x: Data[tuple[I, J], np.float32]
    y: Data[tuple[I, J], np.float32]
    z: Data[tuple[I, J], np.float32]

    i: Coord[I, np.uint64]
    j: Coord[J, np.uint64]

    name: Attr[str]

    __dataoptions__ = DataOptions(StructuredMesh)

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> StructuredMesh:
        """Parses, validates, and builds a Grid from a standard xr.Dataset."""
        return cls.new(**dataset.data_vars, **dataset.coords, **dataset.attrs)  # ty: ignore[invalid-argument-type]


DEFAULT_STRUCTURED_ENCODING_SETTINGS = {
    "x": {"compressors": MEDIUM_COMPRESSOR},
    "y": {"compressors": MEDIUM_COMPRESSOR},
    "z": {"compressors": MEDIUM_COMPRESSOR},
}


def triangulate(mesh: StructuredMesh) -> np.ndarray:
    nx = mesh.sizes["i"]
    ny = mesh.sizes["j"]  # This is the stride multiplier for row-major layout

    # Generate the 2D grids
    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")

    # Use ny (the size of j) instead of nx
    p00 = (ii * ny + jj).ravel()
    p10 = ((ii + 1) * ny + jj).ravel()
    p11 = ((ii + 1) * ny + (jj + 1)).ravel()
    p01 = (ii * ny + (jj + 1)).ravel()

    tri1 = np.stack((p00, p10, p11), axis=1)
    tri2 = np.stack((p00, p11, p01), axis=1)

    faces = np.vstack((tri1, tri2)).astype(np.uint64)
    return faces


def make_mesh(
    name: str,
    points: np.ndarray,
    connectivity: np.ndarray,
    cell_data: dict[str, np.ndarray],
    field_data: dict[str, np.ndarray],
    geometry: shapely.Geometry | None,
) -> TetrahedralMesh:
    """Create a :class:`TetrahedralMesh`.

    Parameters
    ----------
    name :
        Optional human-readable name for the model.  Stored in
        ``field_data["name"]`` when provided so it survives VTKHDF
        round-trips and reaches :func:`~nzcvm.model.MeshModel`.
    points :
        ``(N, 3)`` float32 array of vertex coordinates.
    connectivity :
        ``(M, 4)`` integer array of tetrahedral cell vertex indices.
    cell_data :
        Per-cell arrays such as ``model_type`` and ``models``.
    field_data :
        Per-model arrays such as ``rho``, ``vp``, ``vs``, ``qp``, ``qs``,
        ``alpha`` and ``priority``.
    geometry :
        Optional footprint of the mesh, stored alongside the mesh data.

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
    return encode(mesh, attr_hook=_encode_geometry_attr)
