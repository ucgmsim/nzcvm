"""Discovery of third-party layers and grids through package entry points.

A plugin package registers its modules under these entry-point groups:

``nzcvm.layer_configs``
    Modules defining :class:`~nzcvm.config.layers.LayerConfig` subclasses.
``nzcvm.layer_impls``
    Modules defining :class:`~nzcvm.layers.core.Layer` subclasses.
``nzcvm.grid_configs``
    Modules defining :class:`~nzcvm.config.grids.GridConfig` subclasses.
``nzcvm.grid_impls``
    Modules registering builders with
    :func:`~nzcvm.grids.builder.build_grids_from_config`.

Configs and implementations live in separate groups so that parsing a config
doesn't import the (heavier) runtime half of a plugin.

Examples
--------
In the plugin's ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."nzcvm.layer_configs"]
    mylayer = "nzcvm_mylayer.config"

    [project.entry-points."nzcvm.layer_impls"]
    mylayer = "nzcvm_mylayer.layer"
"""

import logging
from dataclasses import MISSING, dataclass
from importlib.metadata import (
    EntryPoint,
    entry_points,
    packages_distributions,
    version,
)

logger = logging.getLogger(__name__)

LAYER_CONFIGS = "nzcvm.layer_configs"
LAYER_IMPLS = "nzcvm.layer_impls"
GRID_CONFIGS = "nzcvm.grid_configs"
GRID_IMPLS = "nzcvm.grid_impls"

GROUPS = (LAYER_CONFIGS, LAYER_IMPLS, GRID_CONFIGS, GRID_IMPLS)


@dataclass
class PluginStatus:
    """Outcome of loading one entry point."""

    entry_point: EntryPoint
    error: Exception | None = None


#: Load outcomes keyed by entry-point group, filled by :func:`load_plugins`.
LOADED: dict[str, list[PluginStatus]] = {}


def load_plugins(group: str) -> None:
    """Import every entry point in *group*, logging (not raising) failures.

    A broken plugin shouldn't stop ``nzcvm`` itself from importing, so errors
    are recorded in :data:`LOADED` and logged instead.
    """
    statuses = LOADED.setdefault(group, [])
    for entry_point in entry_points(group=group):
        try:
            entry_point.load()
            statuses.append(PluginStatus(entry_point))
        except Exception as e:  # noqa: BLE001 (plugin code may raise anything)
            logger.warning(
                "Failed to load %s plugin %r: %r", group, entry_point.name, e
            )
            statuses.append(PluginStatus(entry_point, e))


@dataclass
class RegisteredType:
    """A layer or grid type nzcvm can build, built in or from a plugin."""

    kind: str
    name: str
    implementation: str
    provider: str


def _provider(obj: object) -> str:
    """Name the distribution providing *obj*, or ``"builtin"`` for nzcvm itself."""
    module = obj.__module__
    if module == "nzcvm" or module.startswith("nzcvm."):
        return "builtin"
    top_level = module.partition(".")[0]
    dists = packages_distributions().get(top_level)
    if not dists:
        return top_level
    return ", ".join(f"{dist} {version(dist)}" for dist in dists)


def _qualified_name(obj: object) -> str:
    """Return the dotted import path of a class or function."""
    return f"{obj.__module__}.{getattr(obj, '__qualname__', repr(obj))}"


def _type_name(config_cls: type) -> str:
    """Read the ``type`` discriminator a config class deserialises from."""
    type_field = getattr(config_cls, "__dataclass_fields__", {}).get("type")
    if type_field is None or isinstance(type_field.default, type(MISSING)):
        return config_cls.__name__
    return str(type_field.default)


def registered_types() -> list[RegisteredType]:
    """List every layer and grid type currently registered.

    Covers built-ins and successfully loaded plugins alike, since both
    register through the same mechanism.
    """
    # Imported here because nzcvm.config imports this module.
    from nzcvm.grids.builder import build_grids_from_config
    from nzcvm.layers.core import Layer

    types = [
        RegisteredType(
            "layer",
            _type_name(config_cls),
            _qualified_name(layer_cls),
            _provider(layer_cls),
        )
        for config_cls, layer_cls in Layer.registry.items()
    ]
    types.extend(
        RegisteredType(
            "grid",
            _type_name(config_cls),
            _qualified_name(builder),
            _provider(builder),
        )
        for config_cls, builder in build_grids_from_config.registry.items()
        if config_cls is not object
    )
    return types


def failed_plugins() -> list[PluginStatus]:
    """List the entry points that raised while loading."""
    return [
        status
        for statuses in LOADED.values()
        for status in statuses
        if status.error is not None
    ]
