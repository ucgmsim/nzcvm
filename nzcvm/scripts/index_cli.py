"""Compile mesh models into memory-mappable index files."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from nzcvm.models.index import index_path
from nzcvm.models.model import MB, MeshModel

app = typer.Typer(help="Compile mesh models into memory-mappable indexes.")
console = Console(stderr=True)


@app.command()
def build(
    models: Annotated[
        list[Path],
        typer.Argument(
            help="Mesh model stores to compile (``*.zarr``).",
            exists=True,
            readable=True,
        ),
    ],
    force: Annotated[
        bool, typer.Option(help="Rewrite an index that is already current.")
    ] = False,
) -> None:
    """Build each mesh's BVH once and write it beside the mesh as ``.nzidx``.

    Loading a mesh with a current index maps the file instead of rebuilding
    the tree, and every process on a node shares the mapped pages.
    """
    for model in models:
        verb = "wrote" if MeshModel.compile_index(model, force=force) else "current"
        index = index_path(model)
        console.print(f"{verb:8s} {index} ({index.stat().st_size * MB:,.1f} MB)")
