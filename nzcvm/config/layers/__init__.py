import importlib
import pkgutil
from importlib.metadata import entry_points

from .core import LayerConfig


def register_layer_config():
    """Trigger layer configuration registration."""
    for _loader, module_name, _is_pkg in pkgutil.walk_packages(
        __path__, __name__ + "."
    ):
        importlib.import_module(module_name)


register_layer_config()

for entry_point in entry_points(group="nzcvm.layers"):
    entry_point.load()


__all__ = ["LayerConfig"]
