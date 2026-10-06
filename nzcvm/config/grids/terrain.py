"""Terrain decay profiles for hybrid terrain-following grids.

Every volumetric grid places its nodes with the hybrid terrain-following
coordinate

.. math::

    z(x, y, k) = A_k + B(A_k, S(x, y)) \\, S(x, y)

where :math:`S` is the topographic surface (positive down, so negative above
sea level), :math:`A_k` is the *nominal depth* of level :math:`k` (its depth on
a flat earth) and :math:`B` is the *decay* weight.  A level
with :math:`B = 1` follows the topography exactly and a level with
:math:`B = 0` is flat.  The classes here choose :math:`B`.

See Also
--------
nzcvm.grids.terrain : Evaluates these profiles on a grid.
"""

import dataclasses
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Annotated, Literal, Self

from mashumaro.types import Discriminator

from nzcvm.config.core import ConfigObject
from nzcvm.config.validation import PositiveFloat, validate_positive

#: ``PositiveFloat`` validation does not reach through ``| None``, so an
#: optional length annotates the whole union.
OptionalPositiveFloat = Annotated[float | None, validate_positive]

_NO_LENGTH = object()


class Solver(StrEnum):
    """Where the solver places each sample of a terrain-following grid.

    ``PHYSICAL`` places it at its true position, for a solver that meshes the
    topography itself, like SW4 with a ``topography`` command.  ``NOMINAL``
    places it at its nominal depth in a flat box, which is how EMOD3D works.
    The earth is then distorted into the box by the decay, and SW4 can run the
    same way without topography to compare with EMOD3D.
    """

    PHYSICAL = auto()
    NOMINAL = auto()


@dataclass
class Decay(ConfigObject):
    """How quickly grid levels stop following the topography with depth."""

    class Config(ConfigObject.Config):
        discriminator = Discriminator(field="type", include_subtypes=True)

    def resolve(self, default_length: float) -> Self:
        """Return a copy with any unset decay length filled in.

        Parameters
        ----------
        default_length :
            Nominal depth (metres) to use when the profile leaves its length
            unset.  Grid builders pass the bottom of the first block.

        Returns
        -------
        Self
            The resolved decay.  Profiles without a length return themselves.
        """
        if getattr(self, "length", _NO_LENGTH) is None:
            return dataclasses.replace(self, length=default_length)
        return self


@dataclass
class SquashedDecay(Decay):
    """No decay: every level is the surface shifted down (:math:`B = 1`).

    The bottom of the grid follows the topography.  This is EMOD3D's
    ``squashed`` topography type.
    """

    type: Literal["squashed"] = "squashed"


@dataclass
class LinearDecay(Decay):
    """Linear decay to a flat level at *length* (:math:`B = 1 - A/L`).

    With *length* equal to the bottom of the first block, the first block is
    stretched linearly between the surface and a flat bottom, which is the
    SW4 curvilinear grid.

    Attributes
    ----------
    length :
        Nominal depth (metres) at which levels become flat.  Defaults to the
        bottom of the first block.
    """

    length: OptionalPositiveFloat = None
    type: Literal["linear"] = "linear"


@dataclass
class TaperedDecay(Decay):
    """Linear decay over a length set by each column's own elevation.

    The decay length is :math:`L = \\tau E(x, y)` for elevation :math:`E`
    above sea level.  Columns at or below sea level do not decay at all.
    With :math:`\\tau = 1` this is EMOD3D's ``squashed_tapered`` topography
    type, which squeezes the earth between :math:`+E` and :math:`-E` into the
    top :math:`E` of the grid and keeps nodes at their true elevation below.

    Attributes
    ----------
    ratio :
        The taper ratio :math:`\\tau`.  Real node spacing in the taper zone is
        :math:`(1 + 1/\\tau)` times the nominal spacing.
    """

    ratio: PositiveFloat = 1.0
    type: Literal["tapered"] = "tapered"


@dataclass
class SleveDecay(Decay):
    """Smooth exponential decay (SLEVE, Schär et al. 2002).

    :math:`B = \\sinh((L - A)/s) / \\sinh(L/s)` above *length* and zero below
    it.  Small *scale* values flatten the levels quickly below the surface;
    large values approach :class:`LinearDecay`.  Under the sea the levels
    bunch up near the seafloor, and they fold over unless
    :math:`s \\tanh(L/s)` is deeper than the deepest seafloor.

    Attributes
    ----------
    scale :
        Decay scale height :math:`s` in metres.
    length :
        Nominal depth (metres) at which levels become flat.  Defaults to the
        bottom of the first block.
    """

    scale: PositiveFloat
    length: OptionalPositiveFloat = None
    type: Literal["sleve"] = "sleve"
