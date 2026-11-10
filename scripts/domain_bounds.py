"""Print the NZTM bounds covering every simulation domain, for ``gdalwarp -te``.

Reads the ``domain`` corners of every ``realisation.json`` under the events
directory (``<event>/<realisation>/realisation.json``, as the workflow
writes them), unions the domains, and prints the union's bounding box,
padded and snapped outward to a multiple of the snap distance, as
``xmin ymin xmax ymax``.
"""

import json
import math
from pathlib import Path
from typing import Annotated

import shapely
import typer
from pyproj import Transformer

app = typer.Typer()


def domain_union(events: Path) -> shapely.Geometry:
    """Union of every realisation's domain under *events*, in NZTM."""
    to_nztm = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    domains = []
    for path in sorted(events.glob("*/*/realisation.json")):
        corners = json.loads(path.read_text())["domain"]["domain"]
        domains.append(
            shapely.Polygon(
                [to_nztm.transform(c["longitude"], c["latitude"]) for c in corners]
            )
        )
    if not domains:
        raise typer.BadParameter(f"No */*/realisation.json under {events}.")
    return shapely.union_all(domains)


@app.command()
def main(
    events: Annotated[
        Path,
        typer.Argument(help="Directory of events.", exists=True, file_okay=False),
    ],
    pad: Annotated[float, typer.Option(help="Padding (m) on every side.")] = 10_000.0,
    snap: Annotated[
        float, typer.Option(help="Snap the bounds outward to this multiple (m).")
    ] = 250.0,
) -> None:
    xmin, ymin, xmax, ymax = domain_union(events).bounds
    bounds = (
        math.floor((xmin - pad) / snap) * snap,
        math.floor((ymin - pad) / snap) * snap,
        math.ceil((xmax + pad) / snap) * snap,
        math.ceil((ymax + pad) / snap) * snap,
    )
    print(" ".join(f"{b:.0f}" for b in bounds))


if __name__ == "__main__":
    app()
