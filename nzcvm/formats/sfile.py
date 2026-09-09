"""SFILE (HDF5) velocity-model writer.

Writes a multi-grid velocity model in the NZCVM sfile HDF5 format used by
downstream seismic simulation tools.
"""

import queue
import threading
from contextlib import AbstractContextManager
from pathlib import Path

import dask
import dask.array as da
import h5py
import numpy as np

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate
from nzcvm.velocity_model import VelocityModel

# Global attributes
ATTENUATION_ATTR = "Attenuation"
ATTENUATION = np.int32(1)
MIN_RESOLUTION_ATTR = "Finest horizontal grid spacing"
MIN_MAX_DEPTH_ATTR = "Min, max depth"
ORIGIN_AZIM_ATTR = "Origin longitude, latitude, azimuth"
NGRIDS_ATTR = "ngrids"
MATERIAL_GROUP = "Material_model"

# Material model attributes
HORIZONTAL_ATTR = "Horizontal grid size"
NUMBER_OF_COMPONENTS_ATTR = "Number of components"
COMPONENT_MAP = {
    "Cp": Component.VP,
    "Cs": Component.VS,
    "Qp": Component.QP,
    "Qs": Component.QS,
    "Rho": Component.RHO,
}
SURFACE_GROUP = "Z_interfaces"


class AsyncHDF5Writer(AbstractContextManager):
    def __init__(self, filename, max_buffer=10):
        self.queue = queue.Queue(maxsize=max_buffer)
        self.filename = filename
        self.stop_event = threading.Event()

    def _write_loop(self):
        with h5py.File(self.filename, "r+") as f:
            while not (self.stop_event.is_set() and self.queue.empty()):
                try:
                    path, key, value = self.queue.get(timeout=1)
                    f[path][key] = value
                    self.queue.task_done()
                except queue.Empty:
                    continue

    def target(self, datapath: str):
        # A dummy object with the dask storage interface, which only defers
        # queueing.
        class Dummy:
            def __setitem__(_self, key, value):
                self.queue.put((datapath, key, value))

        return Dummy()

    def __enter__(self) -> None:
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.stop_event.set()
        self.thread.join()


def to_sfile(velocity_model: VelocityModel, filename: Path):

    # The SW4 file format imposes that outermost axis (the i-axis in this
    # codebase) of the is due north. This code base asserts that i, j, k = x, y,
    # z = east, north, down, so the writer changes the orientation to match.
    velocity_model = velocity_model.orient(Coordinate.J, Coordinate.I, Coordinate.K)

    writer = AsyncHDF5Writer(filename)
    with h5py.File(filename, "w") as f:
        models = list(
            velocity_model.pairwise.values(),
        )
        top_grid, _ = models[0]
        bottom_grid, _ = models[-1]

        global_min, global_max = dask.compute(top_grid.z.min(), bottom_grid.z.max())

        f.attrs.create(
            ORIGIN_AZIM_ATTR,
            data=[
                top_grid.bottom_left_lon,
                top_grid.bottom_left_lat,
                top_grid.grid_azimuth,
            ],
            dtype=np.float64,
        )

        f.attrs.create(ATTENUATION_ATTR, data=ATTENUATION, dtype=np.int32)
        f.attrs.create(
            NGRIDS_ATTR, data=np.int32(len(velocity_model.grids)), dtype=np.int32
        )
        f.attrs.create(
            MIN_RESOLUTION_ATTR, data=np.float64(top_grid.resolution), dtype=np.float64
        )

        mat_group = f.create_group(MATERIAL_GROUP)
        _surface_group = f.create_group(SURFACE_GROUP)

        sources = []
        targets = []

        f.attrs.create(
            MIN_MAX_DEPTH_ATTR, data=[global_min, global_max], dtype=np.float64
        )

        for i, (grid, qualities) in enumerate(models):
            grid_name = f"grid_{i}"
            grid_h5 = mat_group.create_group(grid_name)
            grid_h5.attrs.update(
                {
                    HORIZONTAL_ATTR: float(grid.resolution),
                    NUMBER_OF_COMPONENTS_ATTR: np.int32(len(COMPONENT_MAP)),
                }
            )

            # Setup Material Model Datasets
            for sfile_name, var_name in COMPONENT_MAP.items():
                data = qualities[var_name].data
                ds_path = f"{MATERIAL_GROUP}/{grid_name}/{sfile_name}"

                # Pre-allocate the dataset skeleton
                f.create_dataset(
                    ds_path, shape=data.shape, chunks=data.chunksize, dtype=np.float32
                )

                sources.append(data)
                targets.append(writer.target(ds_path))

            if i == 0:
                top = grid.z.isel({Coordinate.K: 0}).data
                ds_path = f"{SURFACE_GROUP}/z_values_0"
                f.create_dataset(
                    ds_path, shape=top.shape, chunks=top.chunksize, dtype=top.dtype
                )
                sources.append(top)
                targets.append(writer.target(ds_path))

            bottom = grid.z.isel({Coordinate.K: -1}).data
            ds_path = f"{SURFACE_GROUP}/z_values_{i + 1}"
            f.create_dataset(
                ds_path, shape=bottom.shape, chunks=bottom.chunksize, dtype=bottom.dtype
            )
            sources.append(bottom)
            targets.append(writer.target(ds_path))

    with writer:
        da.store(sources, targets, lock=False)
