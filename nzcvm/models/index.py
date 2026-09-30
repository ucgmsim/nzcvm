"""Compiled mesh indexes: where they live and when one can be used.

An index is a mesh's BVH and records written out once (see
``docs/design/index-format.md``), so that a later load maps the file instead
of rebuilding the tree. It sits beside its mesh and carries a fingerprint of
the mesh store; the reader refuses an index whose fingerprint differs from the
store's current one, so a stale index is rebuilt rather than trusted.
"""

import hashlib
import os
from pathlib import Path
from typing import Any

from nzcvm import nzcvm  # ty: ignore[unresolved-import]


def index_path(mesh_path: Path) -> Path:
    """Where the compiled index for the mesh at *mesh_path* lives.

    Beside the mesh, with the ``.nzidx`` suffix in place of ``.zarr``.

    Examples
    --------
    >>> index_path(Path("models/Wellington.zarr"))
    PosixPath('models/Wellington.nzidx')
    """
    return Path(mesh_path).with_suffix(".nzidx")


def open_index(mesh_path: Path, digest: bytes | None = None) -> Any | None:
    """Map the index beside the mesh at *mesh_path*, if it matches the mesh.

    Parameters
    ----------
    mesh_path :
        The mesh whose index to open.
    digest :
        The mesh's :func:`fingerprint`, when the caller already has it.

    Returns
    -------
    Any | None
        The mapped Rust ``PyMeshModel``, or ``None`` when there is no index,
        when it was built from a different version of the mesh, or when the
        reader rejects it. A foreign or truncated file counts as no index
        rather than an error, since the caller can always build instead.
    """
    if digest is None:
        digest = fingerprint(mesh_path)
    try:
        return nzcvm.mesh_model_open(index_path(mesh_path), digest)
    except (OSError, ValueError):
        return None


def fingerprint(mesh_path: Path) -> bytes:
    """A 32-byte summary of the mesh store at *mesh_path*.

    Hashes the path, size and modification time of every file in the store,
    plus the contents of each ``zarr.json``. It's a change detector rather
    than a content hash: it reads metadata rather than the arrays, so checking
    an index against its mesh costs a directory walk and no array reads. A
    mesh rewritten with identical contents fails to match its old index, which
    is the cheap side to err on.
    """
    files = []
    for directory, _, names in os.walk(mesh_path):
        for name in names:
            path = os.path.join(directory, name)
            files.append((os.path.relpath(path, mesh_path), path))

    digest = hashlib.blake2b(digest_size=32)
    for relative, path in sorted(files):
        stat = os.stat(path)
        digest.update(relative.encode())
        digest.update(stat.st_size.to_bytes(8, "little"))
        digest.update(stat.st_mtime_ns.to_bytes(8, "little", signed=True))
        if os.path.basename(path) == "zarr.json":
            digest.update(Path(path).read_bytes())
    return digest.digest()
