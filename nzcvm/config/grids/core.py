from dataclasses import dataclass

from mashumaro.types import Discriminator

from nzcvm.config.core import ConfigObject


@dataclass
class GridConfig(ConfigObject):
    class Config(ConfigObject.Config):
        discriminator = Discriminator(field="type", include_subtypes=True)
