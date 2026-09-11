"""High-level Python wrappers around the compiled Rust velocity-model backend.

The primary public interfaces are :class:`MeshModel` (one tetrahedral mesh) and
:class:`ModelTree` (a priority-ordered collection of meshes with alpha-blended
queries).  Both satisfy the :class:`QueryableModel` protocol, which requires a
:meth:`~QueryableModel.query` method.

:class:`Quality` and the other dataclasses mirror their Rust counterparts, and
query methods return them.

See Also
--------
nzcvm.layers : Pipeline layers for coordinate transforms and model queries.
nzcvm.models.mesh : Mesh I/O utilities used by :meth:`ModelTree.load_models`.
"""

import hashlib
import logging
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import rich
import xarray as xr
from mashumaro.mixins.dict import DataClassDictMixin
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm import nzcvm, registry  # ty: ignore[unresolved-import]
from nzcvm.components import Component
from nzcvm.models.mesh import TetrahedralMesh, TetrahedralMeshSchema
from nzcvm.nzcvm import (  # ty: ignore[unresolved-import]
    PyModelTree,
    QueryCoordinates,
    QueryParams,
)
from nzcvm.qualities import Qualities, QualitiesSchema, Quality
from nzcvm.query import ModelRange

MB = 1 / (1024 * 1024)
logger = logging.getLogger(__name__)


@dataclass
class Point(DataClassDictMixin):
    """A 3D point returned by some query methods.

    Examples
    --------
    >>> p = Point(x=1.5, y=2.5, z=-100.0)
    >>> str(p)
    '(1.5, 2.5, -100)'
    """

    x: float
    y: float
    z: float

    def __str__(self) -> str:
        """Return ``(x, y, z)`` formatted to six decimal digits."""
        return f"({self.x:.6g}, {self.y:.6g}, {self.z:.6g})"


@dataclass
class QueryStats(DataClassDictMixin):
    """Diagnostic counters for one model query.

    Useful for profiling BVH traversal efficiency. Returned by
    :meth:`ModelTree.query_stats`.

    Attributes
    ----------
    aabb_tests :
        Number of axis-aligned bounding-box intersection tests performed.
    simplex_tests :
        Number of simplex (tetrahedron) containment tests performed.
    hit_count :
        Number of simplices that contained the query point.
    output :
        Final blended quality, or ``None`` when the point lies outside the
        model.
    elapsed :
        Wall-clock time for the query in nanoseconds.
    """

    aabb_tests: int
    simplex_tests: int
    hit_count: int
    output: Quality | None
    elapsed: int


@dataclass
class ModelContribution(DataClassDictMixin):
    """One model's contribution to a blended quality result.

    Attributes
    ----------
    priority :
        Integer priority of this model (lower number = higher priority).
    quality :
        Raw (un-blended) quality returned by this model for the query point.
    """

    priority: int
    quality: Quality

    def __str__(self) -> str:
        """Return ``priority=<n>, quality=<Quality>``."""
        return f"priority={self.priority}, quality={self.quality!s}"


@dataclass
class Explanation(DataClassDictMixin):
    """Full audit trail for how :class:`ModelTree` produced a query result.

    Returned by :meth:`ModelTree.get_explanation`. Each element in
    ``contributions`` shows the raw quality from one model; ``output`` is
    the final blended result.

    Attributes
    ----------
    contributions :
        Per-model contributions in priority order.
    output :
        Final blended quality, or ``None`` if no model covered the point.
    termination :
        Index into ``contributions`` at which alpha saturation occurred.
        The query ignored every contribution at or after this index.

    Notes
    -----
    If ``termination`` is ``None`` the blend used every contribution.
    """

    contributions: list[ModelContribution]
    output: Quality | None
    termination: int | None

    def __rich__(self) -> Tree:
        """Return a :class:`rich.tree.Tree` showing per-model contributions."""
        if not self.output:
            return Tree("[red]No model coverage for query point.[/red]")

        root = Tree(f"[bold white]{self.output}[/bold white]")

        for i, contribution in enumerate(self.contributions):
            is_active = self.termination is None or i < self.termination
            colour = "green" if is_active else "red"
            node = root.add(
                f"[{colour}]Model {i} (priority = {contribution.priority})[/{colour}]"
            )
            node.add(f"Quality: {contribution.quality}")

        return root


class MeshModel:
    """One tetrahedral mesh velocity model.

    Wraps a compiled Rust :class:`PyMeshModel` and exposes spatial quality
    queries together with a rich display interface.

    Notes
    -----
    A ``MeshModel`` becomes *consumed* once you pass it to
    :class:`ModelTree`.  Calling :meth:`query` or :attr:`aabb` on a consumed
    instance raises :class:`ValueError`.

    See Also
    --------
    ModelTree : Combines multiple ``MeshModel`` instances with priority blending.
    nzcvm.models.mesh.make_mesh : Build a compatible ``UnstructuredGrid``.
    """

    def __init__(self, raw: Any) -> None:
        """
        Parameters
        ----------
        raw :
            Compiled Rust ``PyMeshModel`` object returned by
            :func:`nzcvm.nzcvm.mesh_model`.
        """
        self._raw = raw

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Load the mesh at *path*, mapping its compiled index when there is one.

        A current index (see :func:`index_path` and :meth:`compile_index`)
        turns the load into a header read and an ``mmap``. Without one, or
        with one whose fingerprint no longer matches the mesh, the loader reads
        the mesh and builds its BVH in memory as before.
        """
        path = Path(path)
        index = current_index(path)
        if index is not None:
            return cls(nzcvm.mesh_model_open(index))
        return cls._build(path)

    @classmethod
    def _build(cls, path: Path) -> Self:
        mesh_dataset = TetrahedralMeshSchema.from_dataset(xr.load_dataset(path))
        return cls(_mesh_model_from_tetra(mesh_dataset))

    @classmethod
    def compile_index(cls, path: Path, force: bool = False) -> Path:
        """Build the mesh at *path* and write its index beside it.

        Parameters
        ----------
        path :
            The mesh to compile.
        force :
            Rewrite an index whose fingerprint still matches the mesh.

        Returns
        -------
        Path
            The index file, whether this call wrote it or found it current.
        """
        path = Path(path)
        if not force and (index := current_index(path)) is not None:
            return index
        index = index_path(path)
        cls._build(path)._raw.write_index(index, fingerprint(path))
        return index

    @property
    def mapped(self) -> bool:
        """Whether the model's arrays are memory-mapped from an index file."""
        return self._raw.is_mapped()

    @classmethod
    def from_mesh(
        cls,
        mesh: TetrahedralMesh,
        name: str | None = None,
    ) -> Self:
        """Build a :class:`MeshModel` from a :class:`~nzcvm.models.mesh.TetrahedralMesh`.

        Parameters
        ----------
        mesh :
            A :class:`~nzcvm.models.mesh.TetrahedralMesh` with the NZCVM cell and
            field data layout (see :func:`nzcvm.models.mesh.make_mesh`).
        name :
            Optional human-readable name.  Takes precedence over any name
            stored in ``mesh.field_data["name"]``.

        Returns
        -------
        MeshModel
            The model wrapping *mesh*.

        See Also
        --------
        nzcvm.models.mesh.make_mesh : Create a compatible :class:`~nzcvm.models.mesh.TetrahedralMesh`.
        ModelTree : Combine multiple ``MeshModel`` instances for priority-blended queries.
        """
        return cls(_mesh_model_from_tetra(mesh, name=name))

    @property
    def name(self) -> str:
        """Human-readable name assigned at construction time."""
        name = self._raw.name
        assert isinstance(name, str)
        return name

    @property
    def priority(self) -> int:
        """int: Model priority lower number = higher priority in a :class:`ModelTree`."""
        priority = self._raw.priority
        assert isinstance(priority, int)
        return priority

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box of this mesh.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A pair ``(min_xyz, max_xyz)`` of shape-``(3,)`` float32 arrays.
        """
        return self._raw.aabb()  # type: ignore[no-any-return]

    def query(self, x: Any, y: Any, z: Any) -> Quality | None:
        """Query material properties at one point.

        Parameters
        ----------
        x, y, z :
            Coordinates in the model's projected CRS (metres).

        Returns
        -------
        Quality or None
            Quality at ``(x, y, z)``, or ``None`` if outside this mesh.

        Raises
        ------
        ValueError
            If a :class:`ModelTree` has already consumed this ``MeshModel``.
        """
        quality_dict = self._raw.query(x, y, z)
        return Quality.from_dict(quality_dict) if quality_dict is not None else None

    def view(self) -> Tree:
        """Return a :class:`rich.tree.Tree` summary of this mesh model."""
        data = self._raw.view()
        label = data.get("name") or f"MeshModel {data['id']}"
        tree = Tree(f"[bold]MeshModel[/bold] [cyan]{label!r}[/cyan]")
        tree.add(f"Priority: {data['priority']}")
        size_mb = round(data["size"] * MB)
        if size_mb > 1024:
            tree.add(f"[red]Size: {size_mb:,} MB[/red]")
        else:
            tree.add(f"Size: {size_mb:,} MB")
        b = data["bounds"]
        tree.add(
            f"Bounds: [X: {b[0]:.0f}–{b[3]:.0f}, "
            f"Y: {b[1]:.0f}–{b[4]:.0f}, "
            f"Z: {b[2]:.0f}–{b[5]:.0f}]"
        )
        transform_str = "None" if data["transform"] is None else "Active"
        tree.add(f"Transform: {transform_str}")
        return tree

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render this mesh model as a rich tree for ``rich.print``.

        Yields
        ------
        rich.tree.Tree
            The metadata tree rich should display for this model.
        """
        yield self.view()


@dataclass
class ModelTree:
    """A velocity model backed by a Rust BVH tree of tetrahedral meshes.

    Wraps one or more :class:`MeshModel` instances (or VTKHDF mesh files)
    into a priority-ordered spatial index.  Queries return blended
    :class:`Quality` values at arbitrary 3D coordinates.

    Notes
    -----
    Lower priority numbers take precedence. When multiple models cover the
    same point their qualities are alpha-composited until the cumulative
    alpha reaches 1.0.

    See Also
    --------
    ModelTree.load_models : Build a model tree from a list of paths.
    ModelTree.from_mesh : Build from an in-memory :class:`~nzcvm.models.mesh.TetrahedralMesh`.
    ModelTree.query : Single-point quality query.
    ModelTree.query_many : Vectorised multi-point query returning an xarray Dataset.
    """

    inner: PyModelTree

    @classmethod
    def load_models(cls, models: Iterable[Path]) -> Self:
        """Load a velocity model from one or more VTKHDF files or a directory.

        Parameters
        ----------
        models : iterable of path
            Paths to individual models.

        Returns
        -------
        ModelTree
            The model tree built from the paths supplied.

        Examples
        --------
        Load all mesh files in a directory (requires data files to exist):

        >>> from pathlib import Path
        >>> ModelTree.load_models([Path("/path/to/models")])  # doctest: +SKIP
        """

        mesh_models = []
        for p in models:
            mesh_models.append(MeshModel.from_path(p)._raw)

        raw = nzcvm.model_tree(mesh_models)
        return cls(raw)

    @classmethod
    def from_mesh(cls, mesh_model: TetrahedralMesh) -> Self:
        """Build a :class:`ModelTree` from one in-memory :class:`~nzcvm.models.mesh.TetrahedralMesh`.

        Parameters
        ----------
        mesh_model :
            A :class:`~nzcvm.models.mesh.TetrahedralMesh` with the NZCVM cell and
            field data layout (see :func:`nzcvm.models.mesh.make_mesh`).

        Returns
        -------
        ModelTree
            A single-mesh tree containing *mesh_model*.

        See Also
        --------
        ModelTree.load_models : Load from VTKHDF files on disk.
        nzcvm.models.mesh.make_mesh : Create a compatible :class:`~nzcvm.models.mesh.TetrahedralMesh`.
        """
        raw_mesh_model = _mesh_model_from_tetra(mesh_model)
        raw_model_tree = nzcvm.model_tree([raw_mesh_model])
        return cls(raw_model_tree)

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box of all meshes in the model.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A pair ``(min_xyz, max_xyz)`` of shape-``(3,)`` float32 arrays
            in the model's coordinate system.
        """
        return self.inner.aabb()

    def query(
        self,
        x: Any,
        y: Any,
        z: Any,
        *,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Quality | None:
        """Query material properties at one point.

        Parameters
        ----------
        x, y, z :
            Coordinates in the model's projected CRS.
        model_range :
            Restricts the query to models whose priority falls within this
            range.  Defaults to :attr:`ModelRange.ALL`.

        Returns
        -------
        Quality or None
            Blended quality, or ``None`` if the point lies outside all
            mesh models in the requested priority range.

        See Also
        --------
        ModelTree.query_many : Vectorised query for arrays of coordinates.
        ModelTree.query_stats : Query with BVH traversal diagnostics.
        ModelTree.get_explanation : Query with per-model contribution details.
        """
        lo, hi = model_range.value
        quality_dict = self.inner.query(x, y, z, lo, hi)
        return Quality.from_dict(quality_dict) if quality_dict is not None else None

    def query_stats(self, x: Any, y: Any, z: Any) -> QueryStats:
        """Query one point and return traversal diagnostics.

        Parameters
        ----------
        x, y, z :
            Coordinates in the model's projected CRS (metres).

        Returns
        -------
        QueryStats
            Traversal counters and the blended result for the query point.

        See Also
        --------
        ModelTree.query : Query without diagnostics.
        """
        return QueryStats.from_dict(self.inner.query_stats(x, y, z))

    def get_explanation(self, x: Any, y: Any, z: Any) -> Explanation:
        """Return a full :class:`Explanation` for a single-point query.

        Parameters
        ----------
        x, y, z :
            Coordinates in the model's projected CRS (metres).

        Returns
        -------
        Explanation
            Per-model contributions and the blended result for the point.

        See Also
        --------
        ModelTree.explain : Pretty-print the explanation to the terminal.
        """
        return Explanation.from_dict(self.inner.explain(x, y, z))

    def explain(self, x: float, y: float, z: float) -> None:
        """Pretty-print the blending explanation for a query point.

        Prints a rich-formatted tree to stdout showing each model's
        contribution and whether the final blend used it.

        Parameters
        ----------
        x, y, z :
            Coordinates in the model's projected CRS (metres).

        See Also
        --------
        ModelTree.get_explanation : Return the explanation as a Python object.
        """
        explanation = self.get_explanation(x, y, z)
        rich.print(explanation)
        if len(explanation.contributions) > 1:
            rich.print(
                "Qualities alpha-blended together until exhaustion "
                "or combined quality alpha ~ 1.0"
            )

    def _query_many_raw(
        self,
        x: Any,
        y: Any,
        z: Any,
        model_range: ModelRange = ModelRange.ALL,
        out: np.ndarray | None = None,
        where: Any = None,
    ) -> np.ndarray:
        """Vectorised query returning a raw float32 array.

        Parameters
        ----------
        x, y, z :
            Arrays of coordinates.
        model_range :
            Restricts the query to models whose priority falls within this
            range.  Defaults to :attr:`ModelRange.ALL`.
        out :
            Optional pre-allocated float32 array of shape ``(*x.shape, 6)``.
            Python is responsible for allocation and zeroing.  When provided,
            the query fills it in place and returns it.
        where :
            Optional boolean array broadcastable to ``x.shape``.  When
            provided, the query visits only points where the mask is ``True``
            and leaves the other rows of ``out`` unchanged.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(*x.shape, 6)`` with columns ordered as
            ``[rho, vp, vs, qp, qs, alpha]``.  Rows for points outside every
            matching model remain zero, or keep their previous value when you
            pass *out* and *where* masks them out.

        See Also
        --------
        ModelTree.query_many : Same query returning a labelled xarray Dataset.
        """
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        if x.shape != y.shape or x.shape != z.shape:
            raise ValueError(
                f"x, y, z must have the same shape; "
                f"got x={x.shape}, y={y.shape}, z={z.shape}"
            )
        orig_shape = x.shape
        n = x.size
        lo, hi = model_range.value
        params = QueryParams(lo, hi)

        if out is None:
            out = np.zeros(orig_shape + (6,), dtype=np.float32)

        # Rust expects a C-contiguous (N, 6) buffer with exclusive ownership.
        out_flat = np.ascontiguousarray(out.reshape(n, 6), dtype=np.float32)

        where_flat: np.ndarray | None = None
        if where is not None:
            where_np = np.broadcast_to(np.asarray(where, dtype=bool), orig_shape)
            where_flat = np.ascontiguousarray(where_np.ravel())

        coords = QueryCoordinates(x.ravel(), y.ravel(), z.ravel(), where_flat)
        self.inner.query_many(out_flat, coords, params)
        return out_flat.reshape(orig_shape + (6,))

    def query_many(
        self,
        x: xr.DataArray,
        y: xr.DataArray,
        z: xr.DataArray,
        *,
        model_range: ModelRange = ModelRange.ALL,
    ) -> Qualities:
        """Vectorised query returning a labelled :class:`xarray.Dataset`.

        Parameters
        ----------
        x, y, z :
            Arrays of coordinates, broadcastable to a common shape.
        model_range :
            Restricts the query to models whose priority falls within this
            range.  Defaults to :attr:`ModelRange.ALL`.

        Returns
        -------
        Qualities
            Qualities dataset.

        See Also
        --------
        ModelTree.query : Same query logic outside of the pipeline
        """
        x, y, z = xr.broadcast(x, y, z)
        darr = xr.apply_ufunc(
            self._query_many_raw,
            x,
            y,
            z,
            input_core_dims=[[], [], []],
            output_core_dims=[["component"]],
            kwargs={"model_range": model_range},
        )
        dset = darr.assign_coords(component=list(Component)).to_dataset(dim="component")
        return QualitiesSchema.from_dataset(dset)

    def view(self) -> Tree:
        """Return a :class:`rich.tree.Tree` representation of the model tree."""
        data = self.inner.view()

        total_size_mb = round(data["size"] * MB)
        tree = Tree(f"Model Tree (Total Size: {total_size_mb:,} MB)")

        for m in data["models"]:
            m_id = m["id"]
            embedded_name = m.get("name") or ""
            name = embedded_name or f"Model {m_id}"

            branch = tree.add(f"{name} (ID: {m_id})")
            size_mb = round(m["size"] * MB)
            branch.add(f"Priority: {m['priority']}")

            if size_mb > 1024:
                branch.add(f"[red]Size: {size_mb:,} MB[/red]")
            else:
                branch.add(f"Size: {size_mb:,} MB")

            b = m["bounds"]
            branch.add(
                f"Bounds: [X: {b[0]:.0f}-{b[3]:.0f}, Y: {b[1]:.0f}-{b[4]:.0f}, Z: {b[2]:.0f}-{b[5]:.0f}]"
            )

            transform_str = "None" if m["transform"] is None else "Active"
            branch.add(f"Transform: {transform_str}")

        return tree

    def __getstate__(self):
        # When standard pickle hits this object, bypass pickling the Rust object
        state = self.__dict__.copy()

        state["inner"] = registry.pickle_pass(self.inner)
        return state

    def __setstate__(self, state):
        # When unpickling, swap the key back for the live object reference
        self.__dict__.update(state)
        key = state["inner"]
        self.inner = registry.REGISTRY[key]


def index_path(mesh_path: Path) -> Path:
    """Where the compiled index for the mesh at *mesh_path* lives.

    Beside the mesh, with the ``.nzidx`` suffix in place of ``.zarr``.

    Examples
    --------
    >>> index_path(Path("models/Wellington.zarr"))
    PosixPath('models/Wellington.nzidx')
    """
    return Path(mesh_path).with_suffix(".nzidx")


def current_index(mesh_path: Path) -> Path | None:
    """The index beside the mesh at *mesh_path*, if it matches the mesh.

    ``None`` when there is no index, when its fingerprint differs from the
    mesh's, or when the reader rejects its header. A foreign or truncated
    file counts as no index rather than an error, since the loader can
    always build instead.
    """
    index = index_path(mesh_path)
    if not index.exists():
        return None
    try:
        stored = nzcvm.index_fingerprint(index)
    except (OSError, ValueError):
        return None
    return index if stored == fingerprint(mesh_path) else None


def fingerprint(mesh_path: Path) -> bytes:
    """A 32-byte summary of the mesh store at *mesh_path*.

    Hashes the path, size and modification time of every file in the store,
    plus the contents of each ``zarr.json``. It's a change detector rather
    than a content hash: it reads metadata rather than the arrays, so checking
    an index against its mesh costs a directory walk and no array reads. A
    mesh rewritten with identical contents fails to match its old index, which
    is the cheap side to err on.
    """
    digest = hashlib.blake2b(digest_size=32)
    for relative, entry in sorted(_walk(Path(mesh_path))):
        stat = entry.stat()
        digest.update(relative.encode())
        digest.update(stat.st_size.to_bytes(8, "little"))
        digest.update(stat.st_mtime_ns.to_bytes(8, "little", signed=True))
        if entry.name == "zarr.json":
            digest.update(Path(entry.path).read_bytes())
    return digest.digest()


def _walk(root: Path, prefix: str = "") -> Iterator[tuple[str, os.DirEntry[str]]]:
    """Every file under *root* with its path relative to *root*.

    ``scandir`` gets the file type from the directory listing, so a store of a
    few hundred chunk files costs one ``stat`` per file rather than the two or
    three a ``rglob`` walk makes. On a network filesystem each of those is a
    round trip.
    """
    with os.scandir(root) as entries:
        for entry in entries:
            relative = f"{prefix}{entry.name}"
            if entry.is_dir(follow_symlinks=False):
                yield from _walk(Path(entry.path), f"{relative}/")
            elif entry.is_file(follow_symlinks=False):
                yield relative, entry


def _mesh_model_from_tetra(
    mesh_model: TetrahedralMesh,
    name: str | None = None,
) -> Any:
    """Build a PyMeshModel from a :class:`~nzcvm.models.mesh.TetrahedralMesh`."""
    connectivity = mesh_model.connectivity.values

    types = mesh_model.model_type.values

    model_idx = mesh_model.models.values.ravel()

    qualities = np.c_[
        mesh_model.rho.values,
        mesh_model.vp.values,
        mesh_model.vs.values,
        mesh_model.qp.values,
        mesh_model.qs.values,
        mesh_model.alpha.values,
    ]

    priority = mesh_model.priority.isel(j=0).item()
    name = name if name is not None else mesh_model.name

    transform = mesh_model.attrs.get("transform")

    if transform is not None:
        transform = np.array(transform, dtype=np.float32)

    points = np.c_[mesh_model.x.values, mesh_model.y.values, mesh_model.z.values]
    try:
        return nzcvm.mesh_model(
            points,
            connectivity,
            types,
            model_idx,
            qualities,
            priority,
            transform,
            name,
        )
    except ValueError as e:
        e.add_note(f"While building model: {mesh_model.name!r}")
        raise
