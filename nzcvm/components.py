"""Physical-property components produced by the velocity model.

Examples
--------
>>> from nzcvm.components import Component
>>> str(Component.VP)
'vp'
"""

from enum import StrEnum, auto


class Component(StrEnum):
    """Seismic material property label used as a dataset variable name.

    Each member's string value is also the variable name in output
    datasets, so ``Component.VP == "vp"`` and a member works directly as
    an xarray coordinate or dimension name.

    Examples
    --------
    >>> Component.RHO == "rho"
    True
    >>> list(Component)  # doctest: +NORMALIZE_WHITESPACE
    [<Component.RHO: 'rho'>, <Component.VP: 'vp'>, <Component.VS: 'vs'>, <Component.QP: 'qp'>, <Component.QS: 'qs'>, <Component.ALPHA: 'alpha'>]
    """

    RHO = auto()
    VP = auto()
    VS = auto()
    QP = auto()
    QS = auto()
    ALPHA = auto()
