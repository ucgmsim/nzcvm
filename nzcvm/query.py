from enum import Enum


class ModelRange(Enum):
    """Priority ranges for bounded velocity-model queries.

    Priority values are ``u8`` ordered so that ``0`` is the highest priority
    and ``255`` is the lowest.  The ranges reflect the NZCVM convention:

    * ``0–127``   — basin models (higher priority, evaluated first).
    * ``128–255`` — tomography models (lower priority, blended in afterwards).

    Attributes
    ----------
    value :
        A ``(priority_lo, priority_hi)`` tuple (both inclusive) passed to
        :meth:`~nzcvm.models.model.ModelTree.query_bounded`.

    Examples
    --------
    >>> ModelRange.BASINS.value
    (0, 127)
    >>> ModelRange.TOMOGRAPHY.value
    (128, 255)
    """

    BASINS = (0, 127)
    TOMOGRAPHY = (128, 255)
    ALL = (0, 255)
