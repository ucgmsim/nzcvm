from dataclasses import dataclass, field

from nzcvm.config.core import ConfigObject
from nzcvm.config.validation import (
    NonNegativeFloat,
    PositiveFloat,
    UnitIntervalFloat,
)

from .core import LayerConfig


@dataclass
class VelocityModel1D(ConfigObject):
    bottom_depth: NonNegativeFloat
    rho: PositiveFloat
    vp: PositiveFloat
    vs: PositiveFloat
    qp: PositiveFloat
    qs: PositiveFloat
    alpha: UnitIntervalFloat

    def __post_init__(self):
        super().__post_init__()
        if self.vp <= self.vs:
            raise ValueError(
                f"Physical constraint violation (vp > vs): {self.vp=}, {self.vs=}"
            )


@dataclass
class DepthModel(ConfigObject):
    distance: NonNegativeFloat
    bottom_depth: NonNegativeFloat


@dataclass
class OffshoreBasinConfig(LayerConfig):
    """Configuration DTO for an :class:`~nzcvm.layers.offshore.OffshoreBasinLayer`.

    Attributes
    ----------
    basin_depth :
        List of :class:`DepthModel` entries mapping offshore distance (m) to
        basin bottom depth (m).  The taper depth at any given distance is
        interpolated from this table.
    model :
        Ordered list of :class:`VelocityModel1D` entries defining the
        velocity–depth profile used in the offshore region.

    Examples
    --------
    TOML::

        [[layers]]
        type = "offshore"
        model = [
            {"bottom_depth": 100.0, "rho": 1820, "vp": 1720, "vs": 500, "qp": 100.0, "qs": 50.0, "alpha": 1.0}
        ]
        basin_depth = [
            {"distance": 0.0, "bottom_depth": 50.0}
        ]
    """

    basin_depth: list[DepthModel]
    model: list[VelocityModel1D]
    type: str = "offshore"
    requires: list[str] = field(default_factory=lambda: ["coastline"], init=False)
