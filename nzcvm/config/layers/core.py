from dataclasses import dataclass, field

from mashumaro import field_options
from mashumaro.types import Discriminator

from nzcvm.config.core import ConfigObject

#: Keeps a derived field out of serialised output.  A layer declares
#: ``provides`` and ``requires`` for itself, so both are ``init=False``, which
#: leaves mashumaro no way to feed them back in.  Serialising them would
#: produce ``to_dict`` output that ``from_dict`` then rejects as an unknown
#: key.
DERIVED = field_options(serialize="omit")


@dataclass
class LayerConfig(ConfigObject):
    provides: list[str] = field(default_factory=list, init=False, metadata=DERIVED)
    requires: list[str] = field(default_factory=list, init=False, metadata=DERIVED)

    class Config(ConfigObject.Config):
        discriminator = Discriminator(field="type", include_subtypes=True)
