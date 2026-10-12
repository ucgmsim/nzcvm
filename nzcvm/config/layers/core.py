from dataclasses import dataclass, field

from mashumaro import field_options
from mashumaro.types import Discriminator

from nzcvm.config.core import ConfigObject, ConfigObjectConfig

DERIVED = field_options(serialize="omit")


@dataclass
class LayerConfig(ConfigObject):
    provides: list[str] = field(default_factory=list, init=False, metadata=DERIVED)
    requires: list[str] = field(default_factory=list, init=False, metadata=DERIVED)

    class Config(ConfigObjectConfig):
        discriminator = Discriminator(field="type", include_subtypes=True)
