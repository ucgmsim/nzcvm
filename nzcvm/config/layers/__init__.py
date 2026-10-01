import importlib
import pkgutil

from nzcvm import plugins

from .core import LayerConfig


def register_layer_config():
    """Trigger layer configuration registration."""
    for _loader, module_name, _is_pkg in pkgutil.walk_packages(
        __path__, __name__ + "."
    ):
        importlib.import_module(module_name)


register_layer_config()

plugins.load_plugins(plugins.LAYER_CONFIGS)


__all__ = ["LayerConfig"]
