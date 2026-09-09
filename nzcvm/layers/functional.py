"""Decorator for constructing :class:`~nzcvm.layers.core.Layer` subclasses
from plain functions.

A *functional layer* is a Layer whose whole body is one plain Python
function.  The decorator inspects the function's keyword parameters and
generates:

1. A :class:`~nzcvm.config.layers.core.LayerConfig` dataclass whose fields
   correspond to those keyword parameters.
2. A :class:`~nzcvm.layers.core.Layer` subclass whose ``__init__`` accepts
   ``(next_layer=None, **config_kwargs)`` and whose ``__call__`` delegates to
   the wrapped function.

The decorated function must accept the following positional arguments:

* ``grid``, the current :class:`~nzcvm.grids.Grid` chunk
* ``model_range``, the active :class:`~nzcvm.query.ModelRange`

and the following keyword argument:

* ``next_layer``, the downstream :class:`~nzcvm.layers.core.Layer` (or
  ``None`` for terminal layers)

All other keyword parameters become config fields on the generated
``LayerConfig`` subclass.

Example
-------
::

    from nzcvm.layers.functional import functional_layer
    from nzcvm.grids.grid import Grid
    from nzcvm.query import ModelRange

    @functional_layer
    def scale(
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
        *,
        next_layer,
        factor: float = 1.0,
    ):
        qualities = next_layer(grid, model_range=model_range)
        qualities["vp"] *= factor
        return qualities

    # Instantiate: keyword args populate the generated ScaleConfig.
    layer = scale(factor=2.0, next_layer=some_terminal)

"""

from __future__ import annotations

import inspect
from dataclasses import field, make_dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast, get_type_hints

from mashumaro.core.meta.mixin import compile_mixin_packer, compile_mixin_unpacker
from shapely import Geometry

from nzcvm.config.layers.core import LayerConfig
from nzcvm.layers.core import Layer
from nzcvm.query import ModelRange

if TYPE_CHECKING:
    from nzcvm.grids.grid import Grid
    from nzcvm.qualities import Qualities

_RUNTIME_PARAMS = frozenset({"grid", "model_range", "next_layer", "return"})


class _LayerFunc(Protocol):
    """Protocol for the functions :func:`functional_layer` accepts."""

    __name__: str

    def __call__(
        self,
        grid: Grid,
        model_range: ModelRange,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Qualities: ...


class GeneratedLayer(Protocol):
    """Protocol for the :class:`~nzcvm.layers.core.Layer` subclass returned by
    :func:`functional_layer`.

    Instances accept either ``(config, geometry, next_layer)`` or
    ``(next_layer=..., geometry=..., **config_kwargs)``, a signature too
    variable for a plain ``type[Layer]`` constructor signature to express.
    """

    config_cls: type[LayerConfig]

    def __call__(self, *args: Any, **kwargs: Any) -> Layer: ...


def _recompile_mashumaro_codecs(cls: type) -> None:
    """Recompile mashumaro packer/unpacker codecs for *cls*.

    On Python 3.14, :func:`dataclasses.make_dataclass` sets ``__annotate__``
    *after* :func:`types.new_class` returns.  Mashumaro's
    ``__init_subclass__`` hook runs during class creation, before the
    annotations are readable, so the generated codecs come out empty.

    This function replays the same ancestor walk that ``__init_subclass__``
    performs, picking up every mixin's builder params (dict, JSON, TOML,
    YAML, …) so that all serialisation formats work correctly.
    """
    # Walk from the most-base ancestor down to the direct parent, mirroring
    # the order mashumaro's own __init_subclass__ uses.
    for ancestor in reversed(cls.__mro__[1:]):
        for attr_name in vars(ancestor):
            if attr_name.endswith("__mashumaro_builder_params"):
                bp = getattr(ancestor, attr_name)
                compile_mixin_unpacker(cls, **bp["unpacker"])
                compile_mixin_packer(cls, **bp["packer"])


def functional_layer(func: _LayerFunc) -> GeneratedLayer:
    """Derive a :class:`~nzcvm.layers.core.Layer` subclass from *func*.

    Parameters
    ----------
    func :
        A callable with signature
        ``(grid, model_range, *, next_layer, **config_params) -> Qualities``.

    Returns
    -------
    GeneratedLayer
        A new, registered Layer subclass whose name matches *func.__name__*.
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    config_fields: list = []
    param_names: list[str] = []

    for name, param in sig.parameters.items():
        if name in _RUNTIME_PARAMS:
            continue
        ann = hints.get(name, Any)
        if param.default is inspect.Parameter.empty:
            config_fields.append((name, ann))
        else:
            config_fields.append((name, ann, field(default=param.default)))
        param_names.append(name)

    type_tag = func.__name__
    # Discriminator field, kept a str so mashumaro serialises it on every
    # Python version.
    config_fields.append(("type", str, field(default=type_tag)))

    config_name = func.__name__.title().replace("_", "") + "Config"
    ConfigCls: type[LayerConfig] = make_dataclass(  # ty: ignore[invalid-assignment]
        config_name,
        config_fields,
        bases=(LayerConfig,),
    )

    _recompile_mashumaro_codecs(ConfigCls)

    _captured_param_names = list(param_names)

    # Parameter names accepted by ConfigCls.__init__ (mashumaro may override).
    _config_init_params = frozenset(
        inspect.signature(ConfigCls.__init__).parameters
    ) - {"self"}

    def _init(  # type: ignore[return]
        self: Any,
        config: Any = None,
        geometry: Geometry | None = None,
        next_layer: Layer | None = None,
        **kwargs: Any,
    ) -> None:
        # Supported calling conventions:
        #   1. LayerCls(config_obj, geometry, next_layer)
        #   2. LayerCls(next_layer=..., geometry=..., **config_kwargs)
        if isinstance(config, ConfigCls):
            # Convention 1: first arg is already a config object.
            pass
        else:
            # Convention 2: kwargs are config field values; config may be a
            # Layer (acting as next_layer) or None.
            if isinstance(config, Layer):
                next_layer = config
            # Keep only kwargs that ConfigCls.__init__ actually accepts
            # (dropping inherited base fields like 'provides', say).
            config = ConfigCls(
                **{k: v for k, v in kwargs.items() if k in _config_init_params}
            )

        Layer.__init__(self, config, geometry, next_layer)  # ty: ignore[invalid-argument-type]

    def _call(
        self: Any,
        grid: Grid,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        params = {n: getattr(self.config, n) for n in _captured_param_names}
        return func(grid, model_range, next_layer=self.next_layer, **params)

    LayerCls: type[Layer] = type(  # type: ignore[assignment]
        func.__name__,
        (Layer,),
        {
            "__init__": _init,
            "__call__": _call,
            "__doc__": func.__doc__,
            "config_cls": ConfigCls,
        },
        config_cls=ConfigCls,
    )

    return cast(GeneratedLayer, LayerCls)
