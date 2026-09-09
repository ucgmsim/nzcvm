from pathlib import Path

import dask.array as da
import h5py
import numpy as np
import pytest
import shapely

from nzcvm.config.metadata import ModelMetadata
from nzcvm.coordinates import Coordinate
from nzcvm.formats import sfile
from nzcvm.grids.grid import GridSchema
from nzcvm.qualities import QualitiesSchema
from nzcvm.velocity_model import VelocityModel


def _grid(name: str, resolution: float, base: float, shape=(2, 3, 4)):
    ni, nj, nk = shape
    i = np.arange(ni, dtype=np.float32)
    j = np.arange(nj, dtype=np.float32)
    k = np.arange(nk, dtype=np.float32)

    ii, jj, kk = da.meshgrid(i, j, k, indexing="ij")

    x = 1000 * ii + 10 * jj + kk
    y = 2000 * ii + 20 * jj + kk
    z = base + 100 * ii + 10 * jj + kk
    depth = z.copy()

    return GridSchema.new(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        z=z.astype(np.float32),
        depth=depth.astype(np.float32),
        name=name,
        resolution=resolution,
        geometry=shapely.box(170.0, -43.0, 172.0, -41.0),
        origin_lon=np.float32(170.0),
        origin_lat=np.float32(-43.0),
        azimuth=np.float32(25.0),
        grid_azimuth=np.float32(25.0),
        bottom_left_lon=np.float32(171.0),
        bottom_left_lat=np.float32(-42.0),
    )


def _qualities(shape=(2, 3, 4), offset=0.0):
    ni, nj, nk = shape
    ii, jj, kk = da.meshgrid(
        np.arange(ni, dtype=np.float32),
        np.arange(nj, dtype=np.float32),
        np.arange(nk, dtype=np.float32),
        indexing="ij",
    )
    base = ii * 100 + jj * 10 + kk + offset

    return QualitiesSchema.new(
        rho=(2000 + base).astype(np.float32),
        vp=(3000 + base).astype(np.float32),
        vs=(1500 + base).astype(np.float32),
        qp=(100 + base).astype(np.float32),
        qs=(50 + base).astype(np.float32),
        alpha=np.zeros(shape, dtype=np.float32),
    )


@pytest.fixture
def simple_velocity_model():
    g0 = _grid("g0", resolution=100.0, base=0.0, shape=(2, 3, 4))
    g1 = _grid("g1", resolution=200.0, base=1000.0, shape=(2, 3, 5))

    q0 = _qualities(shape=(2, 3, 4), offset=0.0)
    q1 = _qualities(shape=(2, 3, 5), offset=1000.0)

    return VelocityModel(
        grids={"g0": g0, "g1": g1},
        qualities={"g0": q0, "g1": q1},
        metadata=ModelMetadata(),
    )


def test_sfile_root_structure(tmp_path: Path, simple_velocity_model):
    out = tmp_path / "model.sfile"

    sfile.to_sfile(simple_velocity_model, out)

    with h5py.File(out, "r") as f:
        assert "Material_model" in f
        assert "Z_interfaces" in f

        assert "Origin longitude, latitude, azimuth" in f.attrs
        assert "Attenuation" in f.attrs
        assert "ngrids" in f.attrs
        assert "Min, max depth" in f.attrs

        np.testing.assert_array_equal(
            f.attrs["Origin longitude, latitude, azimuth"],
            np.array([171.0, -42.0, 25.0], dtype=np.float64),
        )
        assert f.attrs["Attenuation"] == np.int32(1)
        assert f.attrs["ngrids"] == np.int32(2)


def test_sfile_material_groups_and_datasets(
    tmp_path: Path, simple_velocity_model: VelocityModel
):
    out = tmp_path / "model.sfile"

    sfile.to_sfile(simple_velocity_model, out)

    with h5py.File(out, "r") as f:
        mat = f["Material_model"]

        for idx, expected_resolution in enumerate([100.0, 200.0]):
            group = mat[f"grid_{idx}"]
            assert "Horizontal grid size" in group.attrs
            assert "Number of components" in group.attrs

            assert group.attrs["Horizontal grid size"] == expected_resolution
            assert group.attrs["Number of components"] == np.int32(5)

            for name in ["Cp", "Cs", "Qp", "Qs", "Rho"]:
                assert name in group
                assert group[name].dtype == np.float32


def test_sfile_interface_dataset_count(
    tmp_path: Path, simple_velocity_model: VelocityModel
):
    out = tmp_path / "model.sfile"

    sfile.to_sfile(simple_velocity_model, out)

    with h5py.File(out, "r") as f:
        surf = f["Z_interfaces"]
        assert "z_values_0" in surf
        assert "z_values_1" in surf
        assert "z_values_2" in surf


def test_sfile_min_max_depth_attr(tmp_path: Path, simple_velocity_model: VelocityModel):
    out = tmp_path / "model.sfile"

    sfile.to_sfile(simple_velocity_model, out)

    # The writer orients to (j, i, k) first, which must not move min/max.
    top_grid = simple_velocity_model.pairwise["g0"][0].transpose(
        Coordinate.J, Coordinate.I, Coordinate.K
    )
    bottom_grid = simple_velocity_model.pairwise["g1"][0].transpose(
        Coordinate.J, Coordinate.I, Coordinate.K
    )

    expected = np.array(
        [top_grid.z.min().compute().item(), bottom_grid.z.max().compute().item()],
        dtype=np.float64,
    )

    with h5py.File(out, "r") as f:
        assert f.attrs["Min, max depth"] == pytest.approx(expected)


def test_sfile_dataset_shapes_follow_orientation(
    tmp_path: Path, simple_velocity_model: VelocityModel
):
    out = tmp_path / "model.sfile"

    sfile.to_sfile(simple_velocity_model, out)

    # Original shapes:
    # g0 = (i, j, k) = (2, 3, 4)
    # after orient(j, i, k) -> (3, 2, 4)
    with h5py.File(out, "r") as f:
        assert f["Material_model/grid_0/Cp"].shape == (3, 2, 4)
        assert f["Material_model/grid_0/Cs"].shape == (3, 2, 4)
        assert f["Material_model/grid_0/Qp"].shape == (3, 2, 4)
        assert f["Material_model/grid_0/Qs"].shape == (3, 2, 4)
        assert f["Material_model/grid_0/Rho"].shape == (3, 2, 4)

        assert f["Material_model/grid_1/Cp"].shape == (3, 2, 5)

        assert f["Z_interfaces/z_values_0"].shape == (3, 2)
        assert f["Z_interfaces/z_values_1"].shape == (3, 2)
        assert f["Z_interfaces/z_values_2"].shape == (3, 2)


def test_sfile_top_and_bottom_interfaces_match_grid_z(
    tmp_path: Path, simple_velocity_model: VelocityModel
):
    out = tmp_path / "model.sfile"

    sfile.to_sfile(simple_velocity_model, out)

    vm = simple_velocity_model.orient(Coordinate.J, Coordinate.I, Coordinate.K)
    g0 = vm.pairwise["g0"][0]
    g1 = vm.pairwise["g1"][0]

    expected_top = g0.z.isel(k=0).values
    expected_bottom_0 = g0.z.isel(k=-1).values
    expected_bottom_1 = g1.z.isel(k=-1).values

    with h5py.File(out, "r") as f:
        assert f["Z_interfaces/z_values_0"][...] == pytest.approx(expected_top)
        assert f["Z_interfaces/z_values_1"][...] == pytest.approx(expected_bottom_0)
        assert f["Z_interfaces/z_values_2"][...] == pytest.approx(expected_bottom_1)


def test_sfile_material_dataset_values_match_component_mapping(
    tmp_path: Path, simple_velocity_model: VelocityModel
):
    out = tmp_path / "model.sfile"

    sfile.to_sfile(simple_velocity_model, out)

    vm = simple_velocity_model.orient(Coordinate.J, Coordinate.I, Coordinate.K)
    q0 = vm.pairwise["g0"][1]

    with h5py.File(out, "r") as f:
        assert f["Material_model/grid_0/Cp"][...] == pytest.approx(q0["vp"].values)
        assert f["Material_model/grid_0/Cs"][...] == pytest.approx(q0["vs"].values)
        assert f["Material_model/grid_0/Qp"][...] == pytest.approx(q0["qp"].values)
        assert f["Material_model/grid_0/Qs"][...] == pytest.approx(q0["qs"].values)
        assert f["Material_model/grid_0/Rho"][...] == pytest.approx(q0["rho"].values)
