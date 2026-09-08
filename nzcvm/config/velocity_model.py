"""Top-level configuration for NZCVM velocity models.

:class:`VelocityModelConfig` is loaded from a TOML, YAML, or JSON config file
and passed to :class:`~nzcvm.velocity_model.VelocityModel` to build the grid
and query pipeline.

Layers are configured as an ordered list of :class:`~nzcvm.config.layers.LayerConfig`
objects under the ``layers`` key.  The list defines the pipeline: the first
entry is the outermost layer and the last must be a
:class:`~nzcvm.config.layers.query.QueryLayerConfig` that performs the actual
velocity-model queries.  A minimal TOML example::

    [[layers]]
    type = "clamp"
    [layers.clamps.vs]
    min = 500.0

    [[layers]]
    type = "ely"
    vs30 = "path/to/vs30.h5"
    depth_t = 450.0

    [[layers]]
    type = "query"
    model_path = "path/to/models"
    model_globs = ["*.vtkhdf"]

See Also
--------
nzcvm.velocity_model.VelocityModel : Model object built from this config.
nzcvm.layers : Pipeline layers that query and transform the model.
"""

from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Self

from mashumaro.codecs.json import JSONDecoder
from mashumaro.codecs.toml import TOMLDecoder
from mashumaro.codecs.yaml import YAMLDecoder
from mashumaro.exceptions import InvalidFieldValue

# This import just registers all the layers, so while it isn't used directly its plugin architecture is.
from nzcvm.config.core import ConfigObject
from nzcvm.config.grids import GridConfig
from nzcvm.config.layers import LayerConfig
from nzcvm.config.metadata import ModelMetadata

DECODER_MAP = {"yaml": YAMLDecoder, "json": JSONDecoder, "toml": TOMLDecoder}


class VelocityModelConfigFormat(StrEnum):
    """Supported serialisation formats for :class:`VelocityModelConfig` files."""

    INFERRED = auto()
    YAML = auto()
    TOML = auto()
    JSON = auto()


@dataclass
class VelocityModelConfig(ConfigObject):
    """Top-level configuration for an NZCVM velocity model.

    Holds model metadata, a :class:`~nzcvm.config.grids.GridConfig` describing
    the mesh structure, and an ordered list of
    :class:`~nzcvm.config.layers.LayerConfig` objects that define the query
    pipeline.

    See Also
    --------
    VelocityModelConfig.read_config : Load from a TOML, YAML, or JSON file.
    """

    grid: GridConfig
    metadata: ModelMetadata = field(default_factory=ModelMetadata)
    layers: list[LayerConfig] = field(default_factory=list)

    def __post_init__(self):
        # Check layer dependencies.
        provided = set()
        for layer in self.layers:
            requirements = set(layer.requires)
            if not requirements <= provided:
                required_str = ", ".join(sorted(requirements - provided))
                layer_type = getattr(layer, "type", None)
                raise InvalidFieldValue(
                    field_name=f"layers.{layer_type}" if layer_type else "layers",
                    field_type=layer.__class__,
                    field_value=layer,
                    holder_class=self.__class__,
                    msg=f"Layer requires {required_str} coordinates but no configured layer provides them.",
                )
            provided.update(layer.provides)

    @classmethod
    def read_config(
        cls,
        config_path: Path | str,
        format: VelocityModelConfigFormat = VelocityModelConfigFormat.INFERRED,
    ) -> Self:
        """Load a :class:`VelocityModelConfig` from a TOML, YAML, or JSON file.

        Parameters
        ----------
        config_path :
            Path to the configuration file.
        format :
            Explicit file format, or ``VelocityModelConfigFormat.INFERRED`` to
            detect from the file extension.

        Returns
        -------
        VelocityModelConfig
            The parsed configuration.
        """
        config_path = Path(config_path)
        decoder = (
            DECODER_MAP[format]
            if format != VelocityModelConfigFormat.INFERRED
            else DECODER_MAP.get(config_path.suffix.lstrip("."), TOMLDecoder)
        )
        return decoder(cls).decode(config_path.read_text())
