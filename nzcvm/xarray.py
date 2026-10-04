from collections.abc import Callable
from typing import Any, TypeVar, overload

import xarray as xr

T = TypeVar("T", bound=xr.DataArray | xr.Dataset)
Encoder = Callable[[T], T]
AttrEncoder = Callable[[Any], Any]


# One overload per xarray type so each branch below narrows to a concrete
# class, while callers still get their own (sub)class back.
@overload
def encode[Tree: xr.DataTree](
    obj: Tree, attr_hook: AttrEncoder | None = None, **kwarg_hooks: Encoder
) -> Tree: ...
@overload
def encode[Data: xr.Dataset](
    obj: Data, attr_hook: AttrEncoder | None = None, **kwarg_hooks: Encoder
) -> Data: ...
@overload
def encode[Array: xr.DataArray](
    obj: Array, attr_hook: AttrEncoder | None = None, **kwarg_hooks: Encoder
) -> Array: ...
def encode(
    obj: xr.DataArray | xr.DataTree | xr.Dataset,
    attr_hook: AttrEncoder | None = None,
    **kwarg_hooks: Encoder,
) -> xr.DataArray | xr.DataTree | xr.Dataset:
    if isinstance(obj, xr.DataTree):

        def encode_dataset(dataset: xr.Dataset) -> xr.Dataset:
            return encode(dataset, attr_hook=attr_hook, **kwarg_hooks)

        return obj.map_over_datasets(encode_dataset)
    elif isinstance(obj, xr.Dataset):
        obj = obj.copy(deep=False)
        if attr_hook:
            obj.attrs.update(
                {name: attr_hook(attr) for name, attr in obj.attrs.items()}
            )

        for key, encoder in kwarg_hooks.items():
            if key not in obj.data_vars:
                continue
            obj[key] = encoder(obj[key])
        return obj
    elif isinstance(obj, xr.DataArray):
        obj = obj.copy(deep=False)
        if encoder := kwarg_hooks.get(obj.name):
            obj = encoder(obj)

        if attr_hook:
            obj.attrs.update(
                {name: attr_hook(attr) for name, attr in obj.attrs.items()}
            )
        return obj
