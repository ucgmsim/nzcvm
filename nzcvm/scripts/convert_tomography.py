"""Convert a CSV tomography model to a VTKHDF tetrahedral mesh."""

from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from pyproj import CRS, Transformer
from scipy.spatial.transform import Rotation

from nzcvm.coordinates import Affine, affine, reflect_x, scale, translate
from nzcvm.models.mesh import DEFAULT_ENCODING_SETTINGS, TetrahedralMesh, make_mesh

CRS_NZTM = CRS.from_epsg(2193)
CRS_WGS = CRS.from_epsg(4326)
CRS_NZGD49 = CRS.from_epsg(4272)
CRS_UTM60S = CRS.from_epsg(32760)
CRS_NZGD2000 = CRS.from_epsg(4167)


def _project_origin(
    lon: float, lat: float, from_crs: CRS, to_crs: CRS
) -> tuple[float, float]:
    """Project a lon/lat origin into *to_crs* using ``always_xy=True``."""
    tr = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    return tr.transform(lon, lat)


def _ep_affine(origin_crs: CRS) -> Affine:
    ox, oy = _project_origin(172.9037, -41.7638, origin_crs, CRS_NZTM)

    inv = (
        affine(Rotation.from_rotvec([0, 0, 140.0], degrees=True).as_matrix())
        @ reflect_x(dims=3)
        @ scale(1 / 1000.0, 1 / 1000.0, 1 / 1000.0)
        @ translate(-ox, -oy, z=0.0)
    )
    return inv


_EP2010_INV = _ep_affine(CRS_NZGD49)
_EP2020_INV = _ep_affine(CRS_NZGD2000)

_db2025_ox, _db2025_oy = _project_origin(177.0, -39.7499, CRS_WGS, CRS_UTM60S)
DB2025_INV_AFFINE: Affine = (
    affine(Rotation.from_rotvec([0, 0, -35.0], degrees=True).as_matrix())
    @ scale(1 / 1000.0, 1 / 1000.0, 1 / 1000.0)
    @ translate(-_db2025_ox, -_db2025_oy, z=0.0)
)
DB2025_CRS_TRANSFORMER = Transformer.from_crs(CRS_UTM60S, CRS_NZTM, always_xy=True)


@dataclass
class TomographyModel:
    latitude: str
    longitude: str
    x: str
    y: str
    z: str
    rho: str
    vp: str
    vs: str
    #: Explicit analytical inverse of ``affine``; avoids numerical errors from
    #: ``np.linalg.inv``.  Used to populate the ``transform`` field of the mesh.
    affine_inverse: Affine
    # Some models don't contain a qp/qs column, so prefill where that makes sense.
    qp: str = "qp"
    qs: str = "qs"


# See https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/#%E2%80%9CMagic_Bits%E2%80%9D_method


def split_by_3(a: np.ndarray) -> np.ndarray:
    x = a & 0x1FFFFF
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


def morton_map(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    map = np.zeros_like(x)
    map |= split_by_3(x) | split_by_3(y) << 1 | split_by_3(z) << 2
    return map


def tet_connectivity(ni: int, nj: int, nk: int) -> np.ndarray:
    """Split each voxel of an ``(ni, nj, nk)`` grid into five tetrahedra.

    Neighbouring voxels have to agree on the diagonal they share, or the
    tetrahedra leave cracks instead of tiling the volume. Alternating the
    split on the parity of ``i + j + k`` is what makes them agree.

    Parameters
    ----------
    ni, nj, nk :
        Number of grid points along each axis.

    Returns
    -------
    numpy.ndarray
        ``(5 * (ni-1) * (nj-1) * (nk-1), 4)`` array of vertex indices into the
        flattened grid.
    """
    i, j, k = (a.ravel() for a in np.indices((ni - 1, nj - 1, nk - 1), dtype=np.int64))

    def vertex(di: int, dj: int, dk: int) -> np.ndarray:
        return (i + di) * (nj * nk) + (j + dj) * nk + (k + dk)

    # fmt: off
    v000, v100, v010, v110 = vertex(0,0,0), vertex(1,0,0), vertex(0,1,0), vertex(1,1,0)
    v001, v101, v011, v111 = vertex(0,0,1), vertex(1,0,1), vertex(0,1,1), vertex(1,1,1)

    even = [
        (v000, v100, v010, v001), (v110, v100, v010, v111),
        (v101, v100, v001, v111), (v011, v010, v001, v111),
        (v100, v010, v001, v111),
    ]
    odd = [
        (v100, v000, v110, v101), (v010, v000, v110, v011),
        (v001, v000, v101, v011), (v111, v110, v101, v011),
        (v000, v110, v101, v011),
    ]
    # fmt: on

    is_even = (i + j + k) % 2 == 0
    # Filled one tetrahedron at a time: stacking all ten patterns first needs
    # room for three copies of the output at once, and a country-sized
    # tomography grid runs to tens of millions of tetrahedra.
    connectivity = np.empty((len(i), 5, 4), dtype=np.int64)
    for slot, (even_tet, odd_tet) in enumerate(zip(even, odd, strict=True)):
        for corner, (e, o) in enumerate(zip(even_tet, odd_tet, strict=True)):
            np.copyto(connectivity[:, slot, corner], o)
            np.copyto(connectivity[:, slot, corner], e, where=is_even)
    return connectivity.reshape(-1, 4)


def data_frame_to_mesh(
    name: str, df: pd.DataFrame, tomography_model: TomographyModel
) -> TetrahedralMesh:
    rho = (df[tomography_model.rho] * 1000).to_numpy()
    vp = (df[tomography_model.vp] * 1000).to_numpy()
    vs = (df[tomography_model.vs] * 1000).to_numpy()

    if tomography_model.qs not in df.columns:
        qs = 0.05 * vs
    else:
        qs = df[tomography_model.qs].to_numpy()

    if tomography_model.qp not in df.columns:
        qp = 2.0 * qs
    else:
        qp = df[tomography_model.qp].to_numpy()

    model_x = df[tomography_model.x].to_numpy()
    model_y = df[tomography_model.y].to_numpy()
    model_z = df[tomography_model.z].to_numpy()

    nz = len(np.unique_values(model_z))
    ny = len(np.unique_values(model_y))
    nx = len(np.unique_values(model_x))
    n_points = len(model_x)
    if nz * nx * ny != n_points:
        raise ValueError(
            f"Model is not a curvilinear grid, dimensions ({nx}, {ny}, {nz}) inconsistent with number of points ({n_points})."
        )

    modeller_points = np.stack((model_x, model_y, model_z))
    sorter = np.lexsort(modeller_points)

    points = np.c_[model_x, model_y, model_z]
    points = points[sorter]
    rho = rho[sorter]
    vp = vp[sorter]
    vs = vs[sorter]
    qp = qp[sorter]
    qs = qs[sorter]

    naive_idx = tet_connectivity(nz, ny, nx)

    z_idx, y_idx, x_idx = np.indices((nz, ny, nx), dtype=np.int64)
    z_flat = z_idx.flatten()
    y_flat = y_idx.flatten()
    x_flat = x_idx.flatten()

    morton_codes = morton_map(z_flat, y_flat, x_flat)
    morton_sorter = np.argsort(morton_codes)

    inverse_map = np.empty_like(morton_sorter)
    inverse_map[morton_sorter] = np.arange(len(morton_sorter))

    points = points[morton_sorter]
    connectivity = inverse_map[naive_idx]
    transform = tomography_model.affine_inverse.astype(np.float32).T

    field_data = {
        "rho": rho[morton_sorter].astype(np.float32),
        "vp": vp[morton_sorter].astype(np.float32),
        "vs": vs[morton_sorter].astype(np.float32),
        "qp": qp[morton_sorter].astype(np.float32),
        "qs": qs[morton_sorter].astype(np.float32),
        "alpha": np.ones(len(points), dtype=np.float32),
        "transform": transform,
    }
    num_cells = len(connectivity)
    model_type = np.full(num_cells, 1, dtype=np.uint8)
    priority = np.full(num_cells, np.iinfo(np.uint8).max, dtype=np.uint8)

    return make_mesh(
        name=name,
        points=points,
        connectivity=connectivity,
        cell_data={
            "model_type": model_type,
            "priority": priority,
        },
        field_data=field_data,
        geometry=None,
    )


class ModelType(StrEnum):
    EP2020 = auto()


MODEL_KWARGS: dict[ModelType, dict[str, int | str]] = {
    ModelType.EP2020: {"header": 1, "sep": r"\s+"}
}

MODEL_COLUMNS = {
    ModelType.EP2020: TomographyModel(
        latitude="Latitude",
        longitude="Longitude",
        x="x(km)",
        y="y(km)",
        z="Depth(km_BSL)",
        rho="Density",
        vp="Vp",
        vs="Vs",
        affine_inverse=_EP2020_INV,
    )
}

app = typer.Typer(
    help="Convert a CSV-like tomography model to a VTKHDF tetrahedral mesh."
)


@app.command()
def convert(
    model: Annotated[
        Path,
        typer.Argument(
            help="CSV-like readable tomography model.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path, typer.Argument(help="Output path for the converted model.")
    ],
    model_type: Annotated[
        ModelType, typer.Argument(help="Model type to read.")
    ] = ModelType.EP2020,
) -> None:
    """Entry point for the ``nzcvm convert-tomography`` command."""
    df = pd.read_csv(model, **MODEL_KWARGS[model_type])  # ty: ignore[no-matching-overload]

    column_keys = MODEL_COLUMNS[model_type]

    mesh = data_frame_to_mesh(model.stem, df, column_keys)
    mesh.to_zarr(output, mode="w", encoding=DEFAULT_ENCODING_SETTINGS)
