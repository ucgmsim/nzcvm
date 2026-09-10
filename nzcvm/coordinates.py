"""Coordinate systems and spatial transformations for the velocity model.

The core building blocks are composable 3×3 affine matrices (type alias
:data:`Affine`) for 2-D transforms plus a :func:`crs_transform` helper for
pyproj CRS conversions.  The factory functions :func:`translate`,
:func:`scale` and :func:`reflect_x` build affine transforms, which compose
with standard NumPy matrix multiplication (``@``).

A typical pipeline maps local model coordinates to a projected CRS::

    from pyproj import Transformer
    from nzcvm.coordinates import translate, scale

    origin_tr = Transformer.from_crs(4326, 2193, always_xy=True)
    ox, oy = origin_tr.transform(172.0, -43.5)
    affine = translate(ox, oy)
    crs_transformer = Transformer.from_crs(2193, 4326, always_xy=True)

See Also
--------
nzcvm.velocity_model.ModelMetadata : Stores coordinate-system parameters alongside model metadata.
"""

from enum import StrEnum, auto

import numpy as np
import xarray as xr
from pyproj import Transformer

#: 4×4 homogeneous affine matrix operating on (x, y, z, 1) column vectors.
#: Compose transforms left-to-right with ``@``. The leftmost matrix applies
#: last, so ``A @ B`` applies *B* first, then *A*.
Affine = np.ndarray[tuple[int, int], np.dtype[np.float32]]


class Coordinate(StrEnum):
    """Grid axis label for projected spatial and logical index coordinates.

    Used directly as xarray dimension or variable names.

    Examples
    --------
    >>> Coordinate.X == "x"
    True
    """

    X = auto()
    Y = auto()
    Z = auto()
    DEPTH = auto()
    COASTLINE = auto()
    I = auto()
    J = auto()
    K = auto()


NO_ORIGIN = 0
WGS84_EPSG = 4326
NZGD2000_EPSG = 4167


# ---------------------------------------------------------------------------
# Affine factory functions
# ---------------------------------------------------------------------------


def affine(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    aff = np.eye(n + 1, dtype=matrix.dtype)
    aff[:n, :n] = matrix

    return aff


def translate(x: float = 0.0, y: float = 0.0, z: float | None = None) -> Affine:
    """Return a translation matrix.

    Parameters
    ----------
    x, y :
        Translation offsets in the x-y plane.
    z :
        If given, operate in 3D and translate by this amount along z.
        Passing ``z=0.0`` still selects the 4×4 form.

    Returns
    -------
    Affine
        3×3 when *z* is ``None``; 4×4 otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> T = translate(100.0, 200.0)
    >>> (T @ np.array([0.0, 0.0, 1.0]))[:2]
    array([100., 200.])
    >>> T3 = translate(1.0, 2.0, z=3.0)
    >>> (T3 @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    array([1., 2., 3.])
    """
    if z is None:
        m = np.eye(3, dtype=np.float32)
        m[0, 2] = x
        m[1, 2] = y
    else:
        m = np.eye(4, dtype=np.float32)
        m[0, 3] = x
        m[1, 3] = y
        m[2, 3] = z
    return m


def scale(sx: float = 1.0, sy: float = 1.0, sz: float | None = None) -> Affine:
    """Return an anisotropic scale matrix.

    Parameters
    ----------
    sx, sy :
        Scale factors along x and y.
    sz :
        If given, operate in 3D and scale by this amount along z.
        Passing ``sz=1.0`` still selects the 4×4 form.

    Returns
    -------
    Affine
        3×3 when *sz* is ``None``; 4×4 otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> S = scale(2.0, 3.0)
    >>> (S @ np.array([1.0, 1.0, 1.0]))[:2]
    array([2., 3.])
    >>> S3 = scale(2.0, 3.0, sz=4.0)
    >>> (S3 @ np.array([1.0, 1.0, 1.0, 1.0]))[:3]
    array([2., 3., 4.])
    """
    if sz is None:
        m = np.eye(3, dtype=np.float32)
        m[0, 0] = sx
        m[1, 1] = sy
    else:
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = sx
        m[1, 1] = sy
        m[2, 2] = sz
    return m


def reflect_x(dims: int = 2) -> Affine:
    """Return a matrix that negates the x axis.

    Parameters
    ----------
    dims :
        ``2`` → 3×3 (default), ``3`` → 4×4.
    """
    return scale(sx=-1.0) if dims == 2 else scale(sx=-1.0, sy=1.0, sz=1.0)


# ---------------------------------------------------------------------------
# CRS transform helper
# ---------------------------------------------------------------------------


def crs_transform(x, y, *, transformer: Transformer):
    """Apply a pyproj CRS transform to *x* and *y*.

    Parameters
    ----------
    x, y :
        Coordinates to transform.  May be scalars, NumPy arrays, or
        dask-backed :class:`xarray.DataArray` objects.
    transformer :
        A :class:`pyproj.Transformer` built with ``always_xy=True``.

    Returns
    -------
    tuple[array-like, array-like]
        ``(x_out, y_out)`` in the target CRS, matching the type of the inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from pyproj import Transformer
    >>> tr = Transformer.from_crs(4326, 2193, always_xy=True)
    >>> x_out, y_out = crs_transform(np.array([172.0]), np.array([-41.0]), transformer=tr)
    """
    if isinstance(x, xr.DataArray) or isinstance(y, xr.DataArray):

        def _transform_both(
            xi: np.ndarray, yi: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            xo, yo = transformer.transform(xi, yi)
            return np.asarray(xo).astype(np.float32, copy=False), np.asarray(yo).astype(
                np.float32, copy=False
            )

        x_out, y_out = xr.apply_ufunc(
            _transform_both,
            x,
            y,
            input_core_dims=[[], []],
            output_core_dims=[[], []],
            dask="parallelized",
            output_dtypes=[np.float32, np.float32],
        )
        return x_out, y_out
    return transformer.transform(np.asarray(x), np.asarray(y))


def apply_affine_transform[T: (np.ndarray | xr.DataArray | xr.Dataset)](
    transform: Affine, x: T, y: T
) -> tuple[T, T]:
    x_prime = transform[0, 0] * x + transform[0, 1] * y + transform[0, 2]
    y_prime = transform[1, 0] * x + transform[1, 1] * y + transform[1, 2]
    return x_prime, y_prime
