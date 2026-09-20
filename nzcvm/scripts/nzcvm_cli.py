"""Command-line interface for generating NZCVM velocity models."""

import contextlib
import logging
import sys
from json import JSONDecodeError
from pathlib import Path
from tomllib import TOMLDecodeError
from typing import Annotated, Any

import dask
import psutil
import typer
from dask.diagnostics import profile_visualize
from dask.distributed import LocalCluster
from distributed import Client
from mashumaro import MissingField
from mashumaro.exceptions import InvalidFieldValue
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from tqdm.dask import TqdmCallback

from nzcvm import formats, registry
from nzcvm.config import VelocityModelConfig, VelocityModelConfigFormat
from nzcvm.layers import pipeline
from nzcvm.layers.pipeline import execute_model_pipeline
from nzcvm.logging import LogProgress, ResourceMonitor, configure_logging
from nzcvm.scripts import (
    construct_mesh,
    convert_tiff,
    convert_tomography,
    surface_cli,
    synthetic,
    tree_stats,
    view,
)
from nzcvm.velocity_model import VelocityModel


def _extract_error_snippet(config: str, lineno: int) -> tuple[str, int]:
    """Slices out the broken line and surrounding context for syntax highlighting."""
    lines = config.splitlines(keepends=True)
    start = max(lineno - 3, 0)
    end = min(lineno + 2, len(lines))
    context = "".join(lines[start:end])
    return context, start + 1


def print_syntax_error(exc: TOMLDecodeError | JSONDecodeError):
    """Formats structural syntax violations elegantly inside one card."""
    doc = exc.doc  # ty: ignore[unresolved-attribute]
    lineno = exc.lineno  # ty: ignore[unresolved-attribute]
    colno = exc.colno  # ty: ignore[unresolved-attribute]
    msg = exc.msg  # ty: ignore[unresolved-attribute]

    snippet, start_line = _extract_error_snippet(doc, lineno)

    # Header details component
    text_content = f"  [bold]Line:[/bold]     {lineno}, [bold]Column:[/bold] {colno}, [bold]Error:[/bold] {msg}\n"

    # Build components to pack into the panel
    renderables: list[Any] = [text_content]
    lang = "json" if isinstance(exc, JSONDecodeError) else "toml"
    if snippet:
        renderables.append(
            Syntax(
                snippet,
                lang,
                theme="ansi_dark",
                line_numbers=True,
                start_line=start_line,
                highlight_lines={lineno},
            )
        )

    console.print()
    console.print(
        Panel(
            Group(*renderables),
            border_style="red",
            title=f"[bold red]{lang.upper()} Syntax Error[/bold red]",
            title_align="left",
            expand=True,
        )
    )
    console.print()


def print_config_error(exc: Exception):
    """Recursively inspects exception context to find the nested field error."""

    # Track breadcrumbs of fields down the nesting structure
    field_path = []
    current_exc = exc
    most_specific_exc = exc

    while current_exc is not None:
        if isinstance(current_exc, (InvalidFieldValue, MissingField)):
            most_specific_exc = current_exc
            if current_exc.field_name:
                field_path.append(current_exc.field_name)

        current_exc = getattr(current_exc, "__context__", None)

    path_str = ".".join(field_path) if field_path else "Unknown Configuration Property"

    if isinstance(most_specific_exc, InvalidFieldValue):
        ctx_class = most_specific_exc.holder_class
        context_name = (
            ctx_class.__name__ if hasattr(ctx_class, "__name__") else str(ctx_class)
        )

        ftype = most_specific_exc.field_type
        type_name = ftype.__name__ if hasattr(ftype, "__name__") else str(ftype)

        reason = getattr(most_specific_exc, "msg", None) or str(most_specific_exc)

        if "has invalid value" in reason and ":" in reason:
            reason = reason.split(":", 1)[1].strip()

        error_content = (
            f"  [bold]Location:[/bold]  [bold blue]{path_str}[/bold blue]\n"
            f"  [bold]Section:[/bold]   {context_name}\n"
            f"  [bold]Type:[/bold]      {type_name}\n"
            f"  [bold]Reason:[/bold]    [white]{reason}[/white]"
        )

    elif isinstance(most_specific_exc, MissingField):
        ctx_class = most_specific_exc.holder_class_name
        context_name = (
            ctx_class.__name__ if hasattr(ctx_class, "__name__") else str(ctx_class)
        )

        error_content = (
            f"[bold red]Missing Required Field:[/bold red]\n\n"
            f"  [bold]Missing Location:[/bold]  [yellow]{path_str}[/yellow]\n"
            f"  [bold]Section Context:[/bold]   {context_name}"
        )
    else:
        error_content = str(exc)

    console.print()
    console.print(
        Panel(
            error_content,
            border_style="red",
            title="[bold red]Configuration Invalid[/bold red]",
            title_align="left",
            expand=True,
        )
    )
    console.print()


def num_cores() -> int:
    """Return the number of CPU cores available to the current process."""
    process = psutil.Process()

    if hasattr(process, "cpu_affinity"):
        return len(process.cpu_affinity())
    elif cpu_count := psutil.cpu_count():
        return cpu_count
    else:
        raise RuntimeError("Cannot determine CPU count.")


console = Console(stderr=True)


app = typer.Typer(help="NZCVM velocity model toolkit.")
app.add_typer(construct_mesh.app, name="basin")
app.add_typer(convert_tomography.app, name="tomography")
app.add_typer(surface_cli.app, name="surface")
app.add_typer(tree_stats.app, name="tree-stats")
app.add_typer(view.app, name="view")
app.add_typer(convert_tiff.app, name="convert-tiff")
app.add_typer(synthetic.app, name="synthetic")


@app.command()
def generate(
    config: Annotated[
        Path,
        typer.Argument(
            help="Config path to read model grid from.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path, typer.Argument(help="Output path to write velocity model to.")
    ],
    n_threads: Annotated[
        int | None,
        typer.Option(help="Number of threads to spawn to query the model.", min=1),
    ] = None,
    output_format: Annotated[
        formats.Format,
        typer.Option(
            "--format", help="Output format. You can usually leave this as inferred."
        ),
    ] = formats.Format.INFERRED,
    config_format: Annotated[
        VelocityModelConfigFormat,
        typer.Option(
            help="Config format to read. You can usually leave this as inferred."
        ),
    ] = VelocityModelConfigFormat.INFERRED,
    quantise: Annotated[
        bool,
        typer.Option(
            help="If set, quantise the output in NetCDF/Zarr formats using ZFP"
        ),
    ] = False,
    distributed: bool = False,
    progress: bool = False,
    monitor: bool = False,
    log_level: str = "WARNING",
    log_file: Path | None = None,
) -> None:
    """Generate a NZCVM velocity model from a config file."""
    configure_logging(log_level.upper(), log_file)

    resolved_n_threads = n_threads if n_threads is not None else num_cores()
    logger = logging.getLogger(__name__)

    logger.info(f"Running with {resolved_n_threads} threads.")
    exit_stack = contextlib.ExitStack()

    with exit_stack:
        if distributed:
            cluster = LocalCluster(
                processes=False, n_workers=1, threads_per_worker=resolved_n_threads
            )

            client = Client(cluster)

            exit_stack.enter_context(cluster)
            exit_stack.enter_context(client)
            # Only need the registry pipeline manager when managing references
            # to the surface or model tree in the pickling. The built-in scheduler
            # doesn't pickle the objects in threaded mode, so skip it there.
            exit_stack.enter_context(registry.pipeline_context())
        else:
            exit_stack.enter_context(LogProgress())
            exit_stack.enter_context(
                dask.config.set(scheduler="threads", num_workers=resolved_n_threads)
            )

        if progress and distributed:
            raise ValueError(
                "Distributed scheduler does not support --progress (use the dashboard instead)."
            )
        elif progress:
            exit_stack.enter_context(TqdmCallback())

        if monitor:
            exit_stack.enter_context(ResourceMonitor())

        profilers = []

        try:
            velocity_model_spec = VelocityModelConfig.read_config(config, config_format)
        except (InvalidFieldValue, MissingField) as e:
            print_config_error(e)
            sys.exit(1)
        except (TOMLDecodeError, JSONDecodeError) as e:
            print_syntax_error(e)
            sys.exit(1)

        velocity_model = VelocityModel.from_config(velocity_model_spec)

        logger.debug("Building model pipeline")
        model_pipeline = pipeline.build_pipeline(
            velocity_model.geometry, velocity_model_spec.layers
        )
        logger.debug("Model pipeline built")

        velocity_model = execute_model_pipeline(velocity_model, model_pipeline)

        formats.write_velocity_model(
            velocity_model, output, output_format, quantise_arrays=quantise
        )

    if profilers:
        profile_visualize.visualize(profilers)
