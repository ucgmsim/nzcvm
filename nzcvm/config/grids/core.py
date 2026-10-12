from dataclasses import dataclass

from mashumaro.types import Discriminator

from nzcvm.config.core import ConfigObject, ConfigObjectConfig


@dataclass
class GridConfig(ConfigObject):
    class Config(ConfigObjectConfig):
        discriminator = Discriminator(field="type", include_subtypes=True)
