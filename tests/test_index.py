"""Tests for the compiled mesh index.

An index is the mesh's BVH and records written out once so that every later
load maps the file instead of rebuilding the tree. The properties that matter
are that a mapped model answers exactly as the built one does, that a mesh
which has changed since its index came off the compiler gets rebuilt rather
than trusted, and that the command which writes indexes leaves them beside
their meshes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from typer.testing import CliRunner

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]
from nzcvm.models import index as index_module
from nzcvm.models.index import fingerprint, index_path, open_index
from nzcvm.models.model import MeshModel, ModelTree
from nzcvm.scripts.nzcvm_cli import app

runner = CliRunner()


@pytest.fixture()
def mesh_path(tomography_mesh_path: Path) -> Path:
    return tomography_mesh_path


def _mapped(mesh_path: Path, monkeypatch: pytest.MonkeyPatch) -> bool:
    """Whether loading *mesh_path* maps its index rather than building."""
    with monkeypatch.context() as m:
        m.setattr(MeshModel, "_build", classmethod(lambda cls, path: None))
        return MeshModel.from_path(mesh_path) is not None


def _probe(tree: ModelTree) -> xr.Dataset:
    """Query a tree over a fixed spread of points, for comparison.

    `query_many` answers on the `(i, j, k)` grid index. The points go in as
    one column.
    """
    lo, hi = tree.aabb
    rng = np.random.default_rng(0)
    n = 500
    coords = [
        xr.DataArray(
            rng.uniform(lo[i] - 100.0, hi[i] + 100.0, n)
            .astype(np.float32)
            .reshape(n, 1, 1),
            dims=["i", "j", "k"],
        )
        for i in range(3)
    ]
    return tree.query_many(*coords)


# ---------------------------------------------------------------------------
# Loader policy
# ---------------------------------------------------------------------------


def test_index_path_sits_beside_the_mesh() -> None:
    assert index_path(Path("models/Wellington.zarr")) == Path("models/Wellington.nzidx")


def test_without_an_index_the_mesh_is_built(
    mesh_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not _mapped(mesh_path, monkeypatch)


def test_compile_index_writes_beside_the_mesh(mesh_path: Path) -> None:
    assert MeshModel.compile_index(mesh_path)
    written = index_path(mesh_path)
    assert written.exists()
    assert not list(written.parent.glob("*.partial")), "temp file cleaned up"


def test_with_an_index_the_mesh_is_mapped(
    mesh_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    MeshModel.compile_index(mesh_path)
    assert _mapped(mesh_path, monkeypatch)


def test_a_mapped_model_answers_like_the_built_one(mesh_path: Path) -> None:
    built = _probe(ModelTree.load_models([mesh_path]))
    MeshModel.compile_index(mesh_path)
    mapped = _probe(ModelTree.load_models([mesh_path]))
    for name in built.data_vars:
        assert np.array_equal(built[name].values, mapped[name].values), name


def test_the_index_opens_only_with_its_fingerprint(mesh_path: Path) -> None:
    MeshModel.compile_index(mesh_path)
    index = index_path(mesh_path)
    _nzcvm.mesh_model_open(index, fingerprint(mesh_path))
    with pytest.raises(ValueError, match="different source"):
        _nzcvm.mesh_model_open(index, bytes(32))


def test_a_changed_mesh_is_rebuilt_not_trusted(
    mesh_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Touching any file in the store changes the fingerprint."""
    MeshModel.compile_index(mesh_path)
    assert _mapped(mesh_path, monkeypatch)

    some_file = next(p for p in mesh_path.rglob("*") if p.is_file())
    stat = some_file.stat()
    os.utime(some_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert not _mapped(mesh_path, monkeypatch)


def test_compile_index_is_idempotent_unless_forced(mesh_path: Path) -> None:
    assert MeshModel.compile_index(mesh_path)
    index = index_path(mesh_path)
    first = index.stat().st_mtime_ns
    assert not MeshModel.compile_index(mesh_path)
    assert index.stat().st_mtime_ns == first, "a current index is left alone"
    assert MeshModel.compile_index(mesh_path, force=True)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_index_build_command(mesh_path: Path) -> None:
    result = runner.invoke(app, ["index", "build", str(mesh_path)])
    assert result.exit_code == 0, result.output
    assert "wrote" in result.output
    assert open_index(mesh_path) is not None


def test_index_build_reports_a_current_index(mesh_path: Path) -> None:
    runner.invoke(app, ["index", "build", str(mesh_path)])
    result = runner.invoke(app, ["index", "build", str(mesh_path)])
    assert result.exit_code == 0
    assert "current" in result.output


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_foreign_file_is_not_opened_as_an_index(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.nzidx"
    bogus.write_bytes(b"\0" * 8192)
    with pytest.raises(ValueError, match="not an NZCVM index"):
        _nzcvm.mesh_model_open(bogus, bytes(32))


def test_a_missing_index_is_an_io_error(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        _nzcvm.mesh_model_open(tmp_path / "absent.nzidx", bytes(32))


def test_a_foreign_file_beside_the_mesh_counts_as_no_index(
    mesh_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The loader builds rather than fails when the reader rejects the index,
    and says why."""
    index_path(mesh_path).write_bytes(b"\0" * 8192)
    assert open_index(mesh_path) is None
    assert "not an NZCVM index" in caplog.text
    MeshModel.from_path(mesh_path)


def test_without_an_index_the_store_is_not_walked(
    mesh_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def walked(_: Path) -> bytes:
        raise AssertionError("fingerprinted a mesh with no index")

    monkeypatch.setattr(index_module, "fingerprint", walked)
    assert open_index(mesh_path) is None


def test_a_missing_store_has_no_fingerprint(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        fingerprint(tmp_path / "absent.zarr")


def test_index_build_skips_what_it_cannot_compile(
    mesh_path: Path, tmp_path: Path
) -> None:
    """A bad argument is reported, and the meshes after it still compile."""
    stray = tmp_path / "stray.txt"
    stray.write_text("not a mesh")
    result = runner.invoke(app, ["index", "build", str(stray), str(mesh_path)])
    assert result.exit_code == 1
    assert "skipped" in result.output
    assert index_path(mesh_path).exists()
