"""Pickling for objects that wrap a compiled Rust handle.

A :class:`~nzcvm.models.model.ModelTree` holds a BVH built in Rust, and a
:class:`~nzcvm.models.surface.Surface` holds a triangulated index.  Neither has
a Python representation to pickle, so neither could cross a process
boundary, and ``--distributed`` could only ever run thread workers inside the
one interpreter.

Replaying the construction gets across instead of copying the object.  A
classmethod builds each of these objects over arguments that *do* pickle (a
path, or an :class:`xarray.Dataset`), so recording which classmethod and which
arguments is enough for a worker to rebuild it.

Rebuilding costs a read and a BVH build, so :func:`cached` memoises the
path-based factories.  Dask hands the same layer to every chunk, so a worker
process that pays once rather than per task is what makes the arrangement
practical.

The cache uses the path alone as its key and never expires, so a file
edited during a run keeps serving its old contents.  That suits a batch job reading a
fixed data root, which is what this is for.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Self


class Reconstructable:
    """Mixin that pickles an object by replaying its construction.

    A subclass registers its factory with :meth:`built_by`, and pickling then
    stores that call in place of the object's state.

    Examples
    --------
    >>> thing = Reconstructable().built_by(dict, [("a", 1)])
    >>> factory, args = thing.__reduce__()
    >>> factory(*args)
    {'a': 1}
    """

    #: Factory and arguments that rebuild this object, set by :meth:`built_by`.
    #:
    #: Declared on a plain base rather than a dataclass one, so that
    #: :func:`dataclasses.dataclass` won't collect it as a field on the
    #: subclasses that are dataclasses.
    _source: tuple[Callable[..., Any], tuple[Any, ...]] | None = None

    def built_by(self, factory: Callable[..., Any], *args: Any) -> Self:
        """Record the call that rebuilds this object, and return it.

        Parameters
        ----------
        factory :
            A module-level function or classmethod, so that pickle can refer
            to it by name.
        args :
            Arguments to call *factory* with.  They have to pickle.
        """
        self._source = (factory, args)
        return self

    def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        if self._source is None:
            raise TypeError(
                f"Nothing built {type(self).__name__} through "
                "one of its factory methods, so there is no record of how to "
                "rebuild it in another process. Construct it with a factory "
                "such as `load` to make it picklable."
            )
        return self._source


def cached[**P, T](factory: Callable[P, T]) -> Callable[P, T]:
    """Memoise a factory so that unpickling in a worker reloads once.

    Wraps :func:`functools.cache`, and exists to give the call sites a name
    that says why the cache is there rather than looking like a micro
    optimisation.  Use it only for factories keyed on hashable arguments that
    name data on disk.
    """
    return functools.cache(factory)
