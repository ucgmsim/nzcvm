from dataclasses import dataclass, field
from typing import Literal

from mashumaro.exceptions import InvalidFieldValue

from nzcvm.components import Component
from nzcvm.config.core import ConfigObject

from .core import LayerConfig


def _validate_bounds(name: str, min_val: float | None, max_val: float | None):
    """Internal validator for checking range limits and bounds symmetry."""
    match (min_val, max_val):
        case (None, None):
            pass
        case (None, max_val) if max_val <= 0:
            raise ValueError(f"Maximum {name} must be > 0, have: {max_val}.")
        case (None, _max):
            pass
        case (min_val, None) if min_val <= 0:
            raise ValueError(f"Minimum {name} must be > 0, have: {min_val}.")
        case (_min, None):
            pass
        case (min_val, max_val) if not (0 < min_val <= max_val):
            raise ValueError(
                f"{name} bounds make no sense, must have bounds between (0, inf) with max >= min,"
                f" but read {name} min = {min_val} and {name} max = {max_val}."
            )


@dataclass
class Bound(ConfigObject):
    """A ``(min, max)`` bound applied to a component.

    Each side is either a constant, or a multiple of another component when the
    matching ``*_ref`` names one.  ``Bound(min=0.05, min_ref="vs")`` clamps the
    component to at least ``0.05 * Vs`` at every point, which reproduces the
    EMOD3D ``Qs = 50 * Vs`` relation (the pipeline holds Vs in m/s internally,
    so the factor is ``50 / 1000``).  ``None`` leaves that side unbounded.  The
    reference component keeps the pipeline's native units (velocities in m/s),
    evaluated pointwise at clamp time.
    """

    min: float | None = None
    max: float | None = None
    min_ref: Component | None = None
    max_ref: Component | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        for side, coeff, ref in (
            ("min", self.min, self.min_ref),
            ("max", self.max, self.max_ref),
        ):
            if ref is None:
                continue
            try:
                ref = Component(ref)
            except ValueError:
                self._raise(
                    f"{side}_ref",
                    f"{side}_ref must be one of {sorted(str(c) for c in Component)}, got {ref!r}.",
                )
            setattr(self, f"{side}_ref", ref)
            if coeff is None:
                self._raise(
                    side, f"{side}_ref is set but {side} multiplier is missing."
                )

        try:
            if self.min_ref is None and self.max_ref is None:
                _validate_bounds("Component", self.min, self.max)
            else:
                if self.min is not None and self.min <= 0:
                    raise ValueError(
                        f"Minimum multiplier must be > 0, have: {self.min}."
                    )
                if self.max is not None and self.max <= 0:
                    raise ValueError(
                        f"Maximum multiplier must be > 0, have: {self.max}."
                    )
                if (
                    self.min is not None
                    and self.max is not None
                    and self.min_ref == self.max_ref
                    and self.min > self.max
                ):
                    raise ValueError(
                        f"Bounds make no sense: min ({self.min}) must be <= max ({self.max})."
                    )
        except ValueError as e:
            self._raise(
                "max" if self.max and self.min and self.max < self.min else "min",
                str(e),
            )

    def _raise(self, field_name: str, msg: str):
        raise InvalidFieldValue(
            field_name=field_name,
            field_type=float,
            field_value={
                "min": self.min,
                "max": self.max,
                "min_ref": self.min_ref,
                "max_ref": self.max_ref,
            },
            holder_class=self.__class__,
            msg=msg,
        )

    def resolve(self, which: Literal["min", "max"], qualities):
        """Resolve the ``"min"`` or ``"max"`` side to a scalar or per-point array.

        Returns ``None`` for an unbounded side, the constant when no
        ``*_ref`` names a component, or ``coefficient * qualities[ref]`` (a
        DataArray broadcastable to the grid) for a relative bound.
        """
        coeff = getattr(self, which)
        ref = getattr(self, f"{which}_ref")
        if coeff is None:
            return None
        if ref is None:
            return coeff
        return coeff * qualities[ref]


@dataclass
class ClampLayerConfig(LayerConfig):
    """Configuration DTO for a :class:`~nzcvm.layers.clamp.ClampLayer`.

    The *clamps* mapping associates each velocity component with its
    ``(min, max)`` bounds.  ``None`` means unbounded on that side, and the
    layer leaves any component the mapping omits unclamped.

    Vs is the *master* property. A ``vs`` bound clamps Vs, then rebuilds Vp and
    density from the Brocher/Nafe-Drake relations at the affected points, so
    Vs, Vp and density stay mutually consistent.  ``min_vp_vs_ratio`` /
    ``max_vp_vs_ratio`` bound the Vp/Vs (Poisson) ratio as a physical guard
    (the isotropic hard floor is ``sqrt(2)``).  Bounds on any other component
    act as plain hard guards and don't trigger the coherence machinery, so
    reserve them for properties Vs doesn't drive (``qp``, ``qs``) or for
    hard-capping an output.  Such a bound may be a constant or a multiple of
    another component via ``min_ref``/``max_ref`` on the :class:`Bound`.
    Writing ``[layers.clamps.qs] min = 0.05, min_ref = "vs"`` floors Qs at
    ``50 * Vs`` (Vs in m/s), and ``qp`` bounds against ``vp`` the same way.
    """

    type: str = "clamp"
    clamps: dict[Component, Bound] = field(default_factory=dict)
    min_vp_vs_ratio: float | None = None
    max_vp_vs_ratio: float | None = None

    def __post_init__(self):
        super().__post_init__()

        try:
            _validate_bounds("Vp/Vs ratio", self.min_vp_vs_ratio, self.max_vp_vs_ratio)
        except ValueError as e:
            faulty_field = "max_vp_vs_ratio"
            if self.min_vp_vs_ratio is not None and self.min_vp_vs_ratio <= 0:
                faulty_field = "min_vp_vs_ratio"

            raise InvalidFieldValue(
                field_name=faulty_field,
                field_type=float,
                field_value=getattr(self, faulty_field),
                holder_class=self.__class__,
                msg=str(e),
            ) from e

        for component, bound in self.clamps.items():
            try:
                _validate_bounds(str(component), bound.min, bound.max)
            except ValueError as e:
                raise InvalidFieldValue(
                    field_name=f"clamps.{component}",
                    field_type=Bound,
                    field_value=bound,
                    holder_class=self.__class__,
                    msg=str(e),
                ) from e
