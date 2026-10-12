"""Tests for the Banks Peninsula Volcanics GTL taper command-line tool.

* :func:`nzcvm.scripts.banks.main` reads the BPV top of each column from the
  layered, depth-queried mesh itself: the shallowest vertex at each (x, y).
* The ``main`` command applies the Ely et al. (2010) taper only inside the
  region within ``vs30_taper_depth`` of the topography *and* within
  ``ely_taper_depth`` of the BPV top, leaving points outside that region
  at the caller-supplied "full" reference qualities (``vs_full``/``vp_full``/
  ``rho_full``), and leaves ``qp``/``qs`` untouched outside the mask when
  those variables are present in the input mesh. The output stays float32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from typer.testing import CliRunner

from nzcvm import ely_taper
from nzcvm.scripts import banks

runner = CliRunner()

# The mesh is depth-queried, so its z is depth below the topography. Each
# test mesh is one column whose shallowest vertex, the BPV top, is 50 m deep.
# dem_depth = z.clip(min=0).
# bpv_depth = (z - 50).clip(min=0).
# masked point (z=200): dem_depth=200 (<1000), bpv_depth=150 (<350)      -> True
# unmasked, dem-only culprit (z=2000): dem_depth=2000 (>=1000)           -> False
# unmasked, bpv-only culprit (z=450): dem_depth=450 (<1000), bpv_depth=400 (>=350) -> False
_BPV_DEPTH = 50.0
_Z_MASKED = 200.0
_Z_FAR_FROM_DEM = 2000.0
_Z_FAR_FROM_BPV = 450.0


def _mesh_dataset(z_values: list[float], qp_qs: bool) -> xr.Dataset:
    """One column of vertices at (x, y) = (1.5e6, 5.2e6), topped by the BPV top."""
    z = np.array([_BPV_DEPTH, *z_values], dtype=np.float32)
    n = len(z)
    data_vars = {
        "x": ("p", np.full(n, 1.5e6, dtype=np.float32)),
        "y": ("p", np.full(n, 5.2e6, dtype=np.float32)),
        "z": ("p", z),
        "vp": ("p", np.full(n, np.nan, dtype=np.float32)),
        "vs": ("p", np.full(n, np.nan, dtype=np.float32)),
        "rho": ("p", np.full(n, np.nan, dtype=np.float32)),
    }
    if qp_qs:
        data_vars["qp"] = ("p", np.full(n, 999.0, dtype=np.float32))
        data_vars["qs"] = ("p", np.full(n, 999.0, dtype=np.float32))
    return xr.Dataset(data_vars, coords={"p": np.arange(n)})


def _run_main(
    tmp_path: Path,
    mesh_dset: xr.Dataset,
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
) -> None:
    mesh = _mesh_dataset([_Z_MASKED, _Z_FAR_FROM_DEM, _Z_FAR_FROM_BPV], qp_qs=True)
    vs_full, vp_full, rho_full = 4000.0, 7000.0, 2900.0
    output_path, result = _run_main(tmp_path, mesh, vs_full, vp_full, rho_full)
    assert "Saved tapered Banks Peninsula Volcanics model" in result.output

    out = xr.open_dataset(output_path)
    masked, far_from_dem, far_from_bpv = 1, 2, 3

    # Outside the mask, main pins points exactly to the "full" reference
    # values passed on the command line.
    for i in (far_from_dem, far_from_bpv):
        assert out["vs"].values[i] == pytest.approx(vs_full)
        assert out["vp"].values[i] == pytest.approx(vp_full)
        assert out["rho"].values[i] == pytest.approx(rho_full)
        # Outside the mask, main leaves qp/qs untouched rather than
        # overwriting them with a "full" constant.
        assert out["qp"].values[i] == pytest.approx(999.0)
        assert out["qs"].values[i] == pytest.approx(999.0)

    # The masked point gets the Ely GTL taper, computed independently here
    # from the same depth/vs30 relation the script uses.
    dem_depth = _Z_MASKED
    bpv_depth = _Z_MASKED - _BPV_DEPTH
    vs_bpv_top = (
        banks.VS0 + (banks.VS_DEPTH - banks.VS0) * (dem_depth / banks.VS30_TAPER_DEPTH)
    ) * 1000.0
    expected = ely_taper._ely_vs_profile(
        depth=xr.DataArray(bpv_depth),
        vs30=xr.DataArray(vs_bpv_top),
        vp_at_z_t=xr.DataArray(vp_full),
        vs_at_z_t=xr.DataArray(vs_full),
        depth_t=banks.ELY_TAPER_DEPTH,
    )

    for name in ("vs", "vp", "rho", "qp", "qs"):
        assert out[name].values[masked] == pytest.approx(
            float(expected[name]), rel=1e-4
        )
        assert out[name].dtype == np.float32

    # The masked point's tapered vs/vp differ meaningfully from the "full"
    # constants that main would otherwise pin it to.
    assert out["vs"].values[masked] != pytest.approx(vs_full)
    assert out["vp"].values[masked] != pytest.approx(vp_full)


def test_main_without_qp_qs_leaves_them_absent(tmp_path: Path) -> None:
    """When the input mesh has no qp/qs, main() must not try to add them."""
    output_path, _ = _run_main(tmp_path, _mesh_dataset([_Z_MASKED], qp_qs=False))

    out = xr.open_dataset(output_path)
    assert "qp" not in out.data_vars
    assert "qs" not in out.data_vars
