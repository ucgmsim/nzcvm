import importlib
import pkgutil

from nzcvm import plugins

from .core import GridConfig


def register_grid_config():
    """Trigger grid configuration registration."""
    for _loader, module_name, _is_pkg in pkgutil.walk_packages(
        __path__, __name__ + "."
    ):
        importlib.import_module(module_name)


register_grid_config()
plugins.load_plugins(plugins.GRID_CONFIGS)


__all__ = ["GridConfig"]
