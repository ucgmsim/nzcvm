"""Hybrid terrain-following grid construction shared by the volumetric grids.

Every block of every volumetric grid places its nodes with

.. math::

    z = A_k + B(A_k, S) \\, S, \\qquad \\text{depth} = z - S

where :math:`S(x, y)` is the surface (positive down), :math:`A_k` the nominal
depth of level :math:`k` and :math:`B` the decay weight chosen by a
:class:`~nzcvm.config.grids.terrain.Decay`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import dask
import numpy as np
import xarray as xr

from nzcvm import coordinates
from nzcvm.config.grids.model import Model
from nzcvm.config.grids.terrain import (
    Decay,
    LinearDecay,
    SleveDecay,
    Solver,
    SquashedDecay,
    TaperedDecay,
)
from nzcvm.coordinates import Affine, Coordinate
from nzcvm.grids import helpers
from nzcvm.grids.grid import Grid, GridSchema
from nzcvm.models.surface import Surface

#: Number of surface values between the lowest and highest point used to check
#: a decay profile.  Every profile is either linear in the surface value or has
#: a slope independent of it.  The checks only need the extremes, and
#: the interior samples guard against future profiles.
SURFACE_SAMPLES = 65


def decay_weight(decay: Decay, levels: np.ndarray, surface: np.ndarray) -> np.ndarray:
    """Evaluate the decay weight :math:`B` of *decay*.

    Parameters
    ----------
    decay :
        A resolved decay profile.
    levels :
        Nominal depths :math:`A`, broadcastable against *surface*.
    surface :
        Surface values :math:`S`, positive down.

    Returns
    -------
    numpy.ndarray
        Weights in ``[0, 1]``, broadcast from *levels* and, for profiles that
        depend on the surface, *surface*.
    """
    match decay:
        case SquashedDecay():
            return np.ones_like(levels)
        case LinearDecay(length=length):
            return np.clip(1 - levels / length, 0, 1)
        case TaperedDecay(ratio=ratio):
            elevation = -surface
            length = np.where(elevation > 0, ratio * elevation, np.inf)
            # The weight varies with both the column and the level, so it's
            # the full size of the chunk: work on it in place.
            weight = levels / length
            np.subtract(1, weight, out=weight)
            return np.clip(weight, 0, 1, out=weight)
        case SleveDecay(scale=scale, length=length):
            # sinh(a) / sinh(b) rewritten so neither term overflows for
            # lengths many scale heights deep.  Computing b in the dtype of a
            # makes a == b at the surface, so the weight there is exactly 1.
            a = np.maximum(length - levels, 0) / scale
            b = np.asarray(length, dtype=a.dtype) / scale
            return np.exp(a - b) * np.expm1(-2 * a) / np.expm1(-2 * b)
        case _:
            raise TypeError(f"Unsupported decay profile: {type(decay).__name__}")


def terrain_z_depth(
    surface: np.ndarray, levels: np.ndarray, decay: Decay
) -> tuple[np.ndarray, np.ndarray]:
    """Node elevation and depth below the surface for one chunk.

    Computes :math:`z = A + B S` and :math:`\\text{depth} = z - S`, as
    :math:`A + (BS - S)`.

    Parameters
    ----------
    surface :
        Surface values, positive down, broadcastable against *levels*.
    levels :
        Nominal depths.
    decay :
        A resolved decay profile.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``z`` and ``depth`` with the broadcast shape of the inputs.
    """
    weight = decay_weight(decay, levels, surface)
    shift = weight * surface
    np.copyto(shift, 0, where=weight == 0)
    lift = shift - surface
    shift += levels
    lift += levels
    return shift, lift


@dataclass(frozen=True)
class Frame:
    """The rotated horizontal footprint under a terrain grid.

    Attributes
    ----------
    ni, nj :
        Node counts at the finest resolution.
    resolution :
        Finest horizontal node spacing in metres.
    chunks :
        Dask chunk sizes along ``i`` and ``j``, kept at every resolution.
    transform :
        Affine transform from local to projected coordinates.
    attrs :
        Orientation metadata passed to every :class:`GridSchema`.
    """

    ni: int
    nj: int
    resolution: float
    chunks: dict[Coordinate, int]
    transform: Affine
    attrs: dict[str, Any]

    @classmethod
    def new(
        cls,
        orientation: Model,
        ni: int,
        nj: int,
        resolution: float,
        chunks: dict[Coordinate, int],
    ) -> Self:
        """Place an ``ni`` by ``nj`` grid at *orientation*.

        Parameters
        ----------
        orientation :
            Model origin and azimuth.
        ni, nj :
            Node counts at the finest resolution.
        resolution :
            Finest horizontal node spacing in metres.
        chunks :
            Dask chunk sizes along ``i`` and ``j``.

        Returns
        -------
        Self
            The frame.
        """
        transform = helpers.grid_transform(orientation)

        # The same expression helpers.raw_coordinates uses for node (0, 0).
        corner_x, corner_y = coordinates.apply_affine_transform(
            transform,
            np.array([-ni / 2 * resolution], dtype=np.float32),
            np.array([-nj / 2 * resolution], dtype=np.float32),
        )
        corner_lon, corner_lat = orientation.to_wgs84.transform(
            corner_x.item(), corner_y.item()
        )

        attrs = {
            "origin_lon": orientation.origin_lon,
            "origin_lat": orientation.origin_lat,
            "azimuth": orientation.azimuth,
            "grid_azimuth": orientation.grid_azimuth,
            "bottom_left_lon": corner_lon,
            "bottom_left_lat": corner_lat,
            "geometry": helpers.outline(transform, ni * resolution, nj * resolution),
        }
        return cls(ni, nj, resolution, chunks, transform, attrs)

    def coordinates(self, stride: int = 1) -> tuple[xr.DataArray, xr.DataArray]:
        """Projected ``x`` and ``y`` of every *stride*-th node."""
        ox, oy = helpers.raw_coordinates(
            self.ni, self.nj, self.resolution, 0.0, self.chunks, stride
        )
        return coordinates.apply_affine_transform(self.transform, ox, oy)


@dataclass(frozen=True)
class TerrainSurface:
    """The surface on the finest nodes of a frame, with its range.

    Attributes
    ----------
    values :
        Persisted surface values, positive down, on the finest nodes.
    minimum, maximum :
        Range of *values*: the highest peak and the deepest seafloor.
    """

    values: xr.DataArray
    minimum: float
    maximum: float

    @classmethod
    def load(cls, frame: Frame, path: Path) -> Self:
        """Evaluate the surface at *path* on *frame* and persist it.

        Parameters
        ----------
        frame :
            Horizontal footprint to evaluate on.
        path :
            Topographic surface file.

        Returns
        -------
        Self
            The persisted surface and its range.
        """
        x, y = frame.coordinates()
        values = helpers.compute_surface_elevation(Surface.load(path), x, y)
        # Persisting is a 2D cost, small next to the 3D model, and avoids
        # evaluating the surface again for every block and every output array.
        values = values.persist()
        minimum, maximum = dask.compute(values.min(), values.max())
        return cls(values, float(minimum), float(maximum))

    def samples(self) -> np.ndarray:
        """Surface values spanning the range, as a column vector."""
        return np.linspace(self.minimum, self.maximum, SURFACE_SAMPLES)[:, np.newaxis]

    def at_stride(self, stride: int, chunks: dict[Coordinate, int]) -> xr.DataArray:
        """The surface on every *stride*-th node, chunked as *chunks*."""
        if stride == 1:
            return self.values
        return self.values.isel(
            {
                Coordinate.I: slice(None, None, stride),
                Coordinate.J: slice(None, None, stride),
            }
        ).chunk(chunks)


def check_monotonic(
    decay: Decay, levels: np.ndarray, surface: TerrainSurface, name: str
) -> None:
    """Raise if any column of the block folds over itself.

    Parameters
    ----------
    decay :
        A resolved decay profile.
    levels :
        Nominal depths of the block.
    surface :
        The surface the block hangs off.
    name :
        Block name for the error message.

    Raises
    ------
    ValueError
        If ``z`` fails to increase strictly with ``k`` for some surface value.
    """
    samples = surface.samples()
    z, _ = terrain_z_depth(samples, levels[np.newaxis, :], decay)
    spacing = np.diff(z, axis=1)
    if np.any(spacing <= 0):
        worst = samples[np.argmin(spacing.min(axis=1)), 0]
        raise ValueError(
            f"Grid {name!r} folds over where the surface is at z={worst:.1f} m: "
            f"the {type(decay).__name__} shrinks the node spacing to "
            f"{spacing.min():.3g} m.  Use a longer decay length or scale."
        )


def terrain_grid(
    frame: Frame,
    surface: TerrainSurface,
    decay: Decay,
    levels: np.ndarray,
    *,
    name: str,
    solver: Solver,
    stride: int = 1,
) -> Grid:
    """Build one terrain-following block.

    Parameters
    ----------
    frame :
        Horizontal footprint of the finest block.
    surface :
        The surface on the finest nodes of *frame*.
    decay :
        A resolved decay profile.
    levels :
        Nominal depths of the block's levels.
    name :
        Block name.
    solver :
        Where a solver places the samples, stored on the grid.
    stride :
        Keep every *stride*-th node of *frame* horizontally.

    Returns
    -------
    Grid
        The block, with ``i``/``j`` chunked as *frame* and ``k`` one chunk.
    """
    check_monotonic(decay, levels, surface, name)

    x, y = frame.coordinates(stride)
    values = surface.at_stride(stride, frame.chunks)
    nominal = xr.DataArray(
        levels.astype(np.float32),
        dims=[Coordinate.K],
        coords={Coordinate.K: np.arange(len(levels))},
    ).chunk({Coordinate.K: -1})

    # One task per chunk computes both outputs.  Dims come out (i, j, k) from
    # the order the inputs introduce them.
    z, depth = xr.apply_ufunc(
        terrain_z_depth,
        values,
        nominal,
        kwargs={"decay": decay},
        output_core_dims=[[], []],
        dask="parallelized",
        output_dtypes=[np.float32, np.float32],
    )

    x, y, z, depth = xr.broadcast(x, y, z, depth)
    x, y, z, depth = helpers.ensure_chunks(x, y, z, depth)

    return GridSchema.new(
        x,
        y,
        z,
        depth,
        nominal_depth=levels.astype(np.float32),
        name=name,
        resolution=frame.resolution * stride,
        solver=solver.value,
        **frame.attrs,
    )
