"""Interpolation helpers shared between pipeline layers."""

import numpy as np


def step_interpolator(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    """Evaluate the right-continuous step function through *xp* and *fp*.

    Parameters
    ----------
    x :
        Points to evaluate at.
    xp :
        Increasing step positions.
    fp :
        The value from each step position up to the next.

    Returns
    -------
    numpy.ndarray
        ``fp[i]`` for the last ``xp[i] <= x``, with the shape of *x*. Points
        before ``xp[0]`` take ``fp[0]``.
    """
    idx = np.searchsorted(xp, x, side="right") - 1
    idx = np.clip(idx, 0, len(xp) - 1)
    return fp[idx]
