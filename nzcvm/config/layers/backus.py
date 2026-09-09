from dataclasses import dataclass

from .core import LayerConfig


@dataclass
class BackusAveragedLayerConfig(LayerConfig):
    """Configuration DTO for an :class:`~nzcvm.layers.backus.BackusAveragedLayer`.

    Attributes
    ----------
    samples : int
        The number of super samples to consider.

    Examples
    --------
    TOML::

        [[layers]]
        type = "backus"
        samples = 5
    """

    samples: int
    type: str = "backus"
