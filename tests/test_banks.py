"""Tests for the Banks Peninsula Volcanics GTL taper command-line tool.

* :func:`nzcvm.scripts.banks.main` reads the basement as a :class:`SurfaceGrid`
  from a WGS84 lat/lon/elevation HDF5 file, negating elevation to get the
  ``z`` used by the rest of the pipeline.
* The ``main`` command applies the Ely et al. (2010) taper only inside the
  region within ``vs30_taper_depth`` of the DEM *and* within
  ``ely_taper_depth`` of the BPV basement, leaving points outside that region
  at the caller-supplied "full" reference qualities (``vs_full``/``vp_full``/
  ``rho_full``), and leaves ``qp``/``qs`` untouched outside the mask when
  those variables are present in the input mesh.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr
from typer.testing import CliRunner

from nzcvm import ely_taper
from nzcvm.models.mesh import surface_mesh
from nzcvm.models.surface import TRANSFORMER
from nzcvm.scripts import banks

runner = CliRunner()


def _write_basement_h5(
    path: Path, longitude: np.ndarray, latitude: np.ndarray, elevation: np.ndarray
) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("longitude", data=longitude)
        f.create_dataset("latitude", data=latitude)
        f.create_dataset("elevation", data=elevation)


# ---------------------------------------------------------------------------
# main: applying the taper end to end
# ---------------------------------------------------------------------------


def _flat_dem_surface(cx: float, cy: float, side: float, z: float) -> xr.Dataset:
    n = 5
    return surface_mesh(
        "dem",
        np.full((n, n), z, dtype=np.float32),
        x=np.linspace(cx - side / 2, cx + side / 2, n),
        y=np.linspace(cy - side / 2, cy + side / 2, n),
    )


@pytest.fixture()
def basement_h5(tmp_path: Path) -> Path:
    """Flat BPV basement at elevation 50 m (z == -50 everywhere)."""
    longitude = np.linspace(172.0, 172.3, 4)
    latitude = np.linspace(-43.6, -43.3, 4)
    elevation = np.full((len(latitude), len(longitude)), 50.0)
    path = tmp_path / "basement.h5"
    _write_basement_h5(path, longitude, latitude, elevation)
    return path


@pytest.fixture()
def query_point(basement_h5: Path) -> tuple[float, float]:
    """A projected (x, y) point well inside the basement grid's interior."""
    with h5py.File(basement_h5) as f:
        longitude = np.array(f["longitude"])
        latitude = np.array(f["latitude"])
    lon_c = float(longitude[len(longitude) // 2])
    lat_c = float(latitude[len(latitude) // 2])
    x, y = TRANSFORMER.transform(lon_c, lat_c)
    return x, y


@pytest.fixture()
def dem_zarr(tmp_path: Path, query_point: tuple[float, float]) -> Path:
    """Flat DEM at elevation 100 m (z == -100), covering the query point."""
    x, y = query_point
    dset = _flat_dem_surface(cx=x, cy=y, side=20_000.0, z=-100.0)
    path = tmp_path / "dem.zarr"
    dset.to_zarr(path, mode="w")
    return path


def _mesh_dataset(x: float, y: float, z_values: list[float], qp_qs: bool) -> xr.Dataset:
    n = len(z_values)
    data_vars = {
        "x": ("p", np.full(n, x, dtype=np.float32)),
        "y": ("p", np.full(n, y, dtype=np.float32)),
        "z": ("p", np.array(z_values, dtype=np.float32)),
        "vp": ("p", np.full(n, np.nan, dtype=np.float32)),
        "vs": ("p", np.full(n, np.nan, dtype=np.float32)),
        "rho": ("p", np.full(n, np.nan, dtype=np.float32)),
    }
    if qp_qs:
        data_vars["qp"] = ("p", np.full(n, 999.0, dtype=np.float32))
        data_vars["qs"] = ("p", np.full(n, 999.0, dtype=np.float32))
    return xr.Dataset(data_vars, coords={"p": np.arange(n)})


# The mesh is depth-queried, so its z is depth below the DEM. With the
# preceding fixtures the DEM is at elevation 100 and the BPV basement at
# elevation 50. The basement lies 50 m below the DEM.
# dem_depth = z.clip(min=0).
# bpv_depth = (z - 50).clip(min=0).
# masked point (z=200): dem_depth=200 (<1000), bpv_depth=150 (<350)      -> True
# unmasked, dem-only culprit (z=2000): dem_depth=2000 (>=1000)           -> False
# unmasked, bpv-only culprit (z=450): dem_depth=450 (<1000), bpv_depth=400 (>=350) -> False
_Z_MASKED = 200.0
_Z_FAR_FROM_DEM = 2000.0
_Z_FAR_FROM_BPV = 450.0
_BPV_DEPTH = 50.0


def _run_main(
    tmp_path: Path,
    mesh_dset: xr.Dataset,
    dem_zarr: Path,
    basement_h5: Path,
    vs_full: float = 4000.0,
    vp_full: float = 7000.0,
    rho_full: float = 2900.0,
):
    mesh_path = tmp_path / "mesh.zarr"
    mesh_dset.to_zarr(mesh_path, mode="w")
    output_path = tmp_path / "out.zarr"

    result = runner.invoke(
        banks.app,
        [
            str(mesh_path),
            str(dem_zarr),
            str(basement_h5),
            str(output_path),
            "--vs-full",
            str(vs_full),
            "--vp-full",
            str(vp_full),
            "--rho-full",
            str(rho_full),
        ],
    )
    assert result.exit_code == 0, result.output
    return output_path, result


def test_main_applies_taper_inside_mask_and_full_values_outside(
    tmp_path: Path,
    query_point: tuple[float, float],
    dem_zarr: Path,
    basement_h5: Path,
) -> None:
    x, y = query_point
    mesh = _mesh_dataset(
        x, y, [_Z_MASKED, _Z_FAR_FROM_DEM, _Z_FAR_FROM_BPV], qp_qs=True
    )
    vs_full, vp_full, rho_full = 4000.0, 7000.0, 2900.0
    output_path, result = _run_main(
        tmp_path, mesh, dem_zarr, basement_h5, vs_full, vp_full, rho_full
    )
    assert "Saved tapered Banks Peninsula Volcanics model" in result.output

    out = xr.open_dataset(output_path)

    # Outside the mask, main pins points exactly to the "full" reference
    # values passed on the command line.
    for i in (1, 2):
        assert out["vs"].values[i] == pytest.approx(vs_full)
        assert out["vp"].values[i] == pytest.approx(vp_full)
        assert out["rho"].values[i] == pytest.approx(rho_full)
        # Outside the mask, main leaves qp/qs untouched rather than
        # overwriting them with a "full" constant.
        assert out["qp"].values[i] == pytest.approx(999.0)
        assert out["qs"].values[i] == pytest.approx(999.0)

    # The masked point gets the Ely GTL taper, computed independently here
    # from the same depth/vs30 relation the script uses.
    dem_depth = max(_Z_MASKED, 0.0)
    bpv_depth = max(_Z_MASKED - _BPV_DEPTH, 0.0)
    vs30_taper_depth = banks.VS30_TAPER_DEPTH
    vs_bpv_top = (
        banks.VS0 + (banks.VS_DEPTH - banks.VS0) * (dem_depth / vs30_taper_depth)
    ) * 1000.0
    expected = ely_taper._ely_vs_profile(
        depth=xr.DataArray(bpv_depth),
        vs30=xr.DataArray(vs_bpv_top),
        vp_at_z_t=xr.DataArray(vp_full),
        vs_at_z_t=xr.DataArray(vs_full),
        depth_t=banks.ELY_TAPER_DEPTH,
    )

    assert out["vs"].values[0] == pytest.approx(float(expected.vs), rel=1e-4)
    assert out["vp"].values[0] == pytest.approx(float(expected.vp), rel=1e-4)
    assert out["rho"].values[0] == pytest.approx(float(expected.rho), rel=1e-4)
    assert out["qp"].values[0] == pytest.approx(float(expected.qp), rel=1e-4)
    assert out["qs"].values[0] == pytest.approx(float(expected.qs), rel=1e-4)

    # The masked point's tapered vs/vp differ meaningfully from the "full"
    # constants that main would otherwise pin it to.
    assert out["vs"].values[0] != pytest.approx(vs_full)
    assert out["vp"].values[0] != pytest.approx(vp_full)


def test_main_without_qp_qs_leaves_them_absent(
    tmp_path: Path,
    query_point: tuple[float, float],
    dem_zarr: Path,
    basement_h5: Path,
) -> None:
    """When the input mesh has no qp/qs, main() must not try to add them."""
    x, y = query_point
    mesh = _mesh_dataset(x, y, [_Z_MASKED], qp_qs=False)
    output_path, _ = _run_main(tmp_path, mesh, dem_zarr, basement_h5)

    out = xr.open_dataset(output_path)
    assert "qp" not in out.data_vars
    assert "qs" not in out.data_vars
