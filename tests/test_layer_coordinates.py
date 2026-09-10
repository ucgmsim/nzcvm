"""A layer has to hand back the coordinates of the grid it received.

A layer takes a :class:`~nzcvm.grids.grid.Grid` and returns
:class:`~nzcvm.qualities.Qualities`. Nothing in the :class:`Layer` contract
says the result keeps the grid's coordinates, and a layer that assembles its
output from raw NumPy drops them: the array has no coordinates to keep. Two
things break at once. ``map_blocks`` rejects a chunk whose coordinates differ
from the template, and the ``site`` labels a borehole grid puts on ``i`` never
reach a writer.

So each test here passes a labelled borehole grid to one registered layer,
and to the whole chain, then checks the label is still on the result.
:data:`COVERED` has to match
:attr:`~nzcvm.layers.core.Layer.registry`, so a new layer fails here until
someone covers it.
"""

from __future__ import annotations

import gzip
import importlib
import pkgutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import shapely
import shapely.ops
import xarray as xr
from pyproj import CRS, Transformer

from nzcvm import synthetic
from nzcvm.components import Component
from nzcvm.config.grids.borehole import BoreholeGridConfig, Site
from nzcvm.config.grids.model import Projection
from nzcvm.config.layers.backus import BackusAveragedLayerConfig
from nzcvm.config.layers.clamp import Bound, ClampLayerConfig
from nzcvm.config.layers.coastline import CoastlineConfig
from nzcvm.config.layers.core import LayerConfig
from nzcvm.config.layers.ely import ElyLayerConfig
from nzcvm.config.layers.offshore import (
    DepthModel,
    OffshoreBasinConfig,
    VelocityModel1D,
)
from nzcvm.config.layers.query import QueryLayerConfig
from nzcvm.config.metadata import ModelMetadata
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.grids.grid import Grid
from nzcvm.layers.core import Layer, layer_from_config
from nzcvm.layers.pipeline import build_pipeline, execute_model_pipeline
from nzcvm.models.mesh import StructuredMeshSchema
from nzcvm.qualities import Qualities, QualitiesSchema
from nzcvm.query import ModelRange
from nzcvm.scripts.convert_tomography import (
    MODEL_COLUMNS,
    ModelType,
    data_frame_to_mesh,
)

_NZTM = CRS.from_epsg(2193)
_TO_NZTM = Transformer.from_crs(4326, _NZTM, always_xy=True)

# One site inland and one offshore, so the coastline-dependent layers see both
# sides of the shoreline and can't skip their work.
SITES = [
    Site(longitude=172.15, latitude=-43.70, labels={"site": "GULL"}),
    Site(longitude=172.55, latitude=-43.60, labels={"site": "SEAB"}),
]

#: The label on the sites, treated like any other.
SITE = "site"
NAMES = [site.labels[SITE] for site in SITES]


# ---------------------------------------------------------------------------
# The synthetic data each layer needs, in the formats the layers read
# ---------------------------------------------------------------------------


def _surface(path: Path, values: np.ndarray, samples: int) -> Path:
    """Write a Zarr surface mesh holding *values* over the synthetic domain."""
    lon, lat = synthetic.DOMAIN.sample(samples, samples)
    mesh_lon, mesh_lat = np.meshgrid(lon, lat, indexing="ij")
    x, y = _TO_NZTM.transform(mesh_lon, mesh_lat)
    StructuredMeshSchema.new(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        z=values.astype(np.float32),
        i=np.arange(samples),
        j=np.arange(samples),
        name=path.stem,
    ).to_zarr(path, mode="w")
    return path


@pytest.fixture(scope="module")
def resources(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A synthetic DEM, Vs30 map, coastline and model directory."""
    root = tmp_path_factory.mktemp("resources")
    samples = 24
    lon, lat = synthetic.DOMAIN.sample(samples, samples)
    mesh_lon, mesh_lat = np.meshgrid(lon, lat, indexing="ij")

    # A surface file and a grid agree on +z down, so elevation flips.
    dem = _surface(root / "dem.zarr", -synthetic.elevation(mesh_lon, mesh_lat), samples)
    vs30 = _surface(root / "vs30.zarr", synthetic.vs30(mesh_lon, mesh_lat), samples)

    coastline = root / "coastline.wkb.gz"
    projected = shapely.ops.transform(
        _TO_NZTM.transform, shapely.Polygon(synthetic.land())
    )
    with gzip.open(coastline, "wb") as handle:
        handle.write(shapely.to_wkb(projected))

    models = root / "models"
    models.mkdir()
    mesh = data_frame_to_mesh(
        "tomography",
        synthetic.tomography(n_horizontal=6, n_depth=5),
        MODEL_COLUMNS[ModelType.EP2020],
    )
    mesh.to_zarr(models / "tomography.zarr", mode="w")

    return {"dem": dem, "vs30": vs30, "coastline": coastline, "models": models}


@pytest.fixture(scope="module")
def borehole_grid(resources: dict[str, Path]) -> Grid:
    """A two-site borehole grid, labelled on ``i`` by site."""
    return build_grids_from_config(
        BoreholeGridConfig(
            surface=resources["dem"],
            sites=SITES,
            depth=400.0,
            resolution_z=100.0,
            projection=Projection(crs=_NZTM),
        )
    )["boreholes"]


@pytest.fixture()
def concrete_grid(borehole_grid: Grid) -> Grid:
    """The same grid, computed.

    `execute_model_pipeline` hoists the chunked dispatch into one
    `map_blocks` per grid, so a layer is only ever handed a concrete chunk.
    Calling a layer on a Dask-backed grid raises inside `apply_ufunc`.
    """
    return borehole_grid.compute()


# ---------------------------------------------------------------------------
# One config per registered layer
# ---------------------------------------------------------------------------


def _configs(resources: dict[str, Path]) -> dict[str, LayerConfig]:
    return {
        "backus": BackusAveragedLayerConfig(samples=3),
        "clamp": ClampLayerConfig(clamps={Component.VS: Bound(min=4000.0)}),
        "coastline": CoastlineConfig(coastline=resources["coastline"]),
        "ely": ElyLayerConfig(vs30=resources["vs30"], depth_t=450.0),
        "offshore": OffshoreBasinConfig(
            basin_depth=[
                DepthModel(distance=0.0, bottom_depth=0.0),
                DepthModel(distance=10_000.0, bottom_depth=1000.0),
            ],
            model=[
                VelocityModel1D(
                    bottom_depth=50.0,
                    rho=1810.0,
                    vp=1800.0,
                    vs=380.0,
                    qp=100.0,
                    qs=50.0,
                    alpha=1.0,
                ),
                VelocityModel1D(
                    bottom_depth=300.0,
                    rho=1810.0,
                    vp=1800.0,
                    vs=580.0,
                    qp=100.0,
                    qs=50.0,
                    alpha=1.0,
                ),
                VelocityModel1D(
                    bottom_depth=1200.0,
                    rho=1810.0,
                    vp=1800.0,
                    vs=830.0,
                    qp=100.0,
                    qs=50.0,
                    alpha=1.0,
                ),
            ],
        ),
        "query": QueryLayerConfig(
            model_path=resources["models"], model_globs=["*.zarr"]
        ),
    }


#: The layers these tests pass a grid to.
COVERED = frozenset({"backus", "clamp", "coastline", "ely", "offshore", "query"})


def _shipped_layer_types() -> set[str]:
    """The ``type`` discriminator of every layer config in the package.

    Read off :mod:`nzcvm.config.layers` rather than
    :attr:`~nzcvm.layers.core.Layer.registry`, which picks up the dummy and
    sentinel layers the rest of the suite defines. The discriminator is a
    dataclass field default, which avoids building a config instance: several
    of them take a required path.
    """
    package = importlib.import_module("nzcvm.config.layers")
    types = set()
    for _loader, module_name, _is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        module = importlib.import_module(module_name)
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, LayerConfig)
                and value is not LayerConfig
                and value.__module__ == module_name
            ):
                types.update(
                    field.default
                    for field in fields(value)
                    if field.name == "type" and isinstance(field.default, str)
                )
    return types


def test_every_shipped_layer_is_covered() -> None:
    assert _shipped_layer_types() == COVERED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _TerminalConfig(LayerConfig):
    type: Literal["_terminal"] = "_terminal"


class _Terminal(Layer[_TerminalConfig]):
    """A terminal built with xarray, so it keeps the grid's coordinates.

    `nzcvm.layers.dummy.constant` would be the obvious choice, but it builds
    its output from `np.ones` and so hands back no coordinates at all, which
    is the property under test here.
    """

    def __init__(self) -> None:
        super().__init__(_TerminalConfig(), None, None)  # ty: ignore[invalid-argument-type]

    def __call__(
        self, grid: Grid, model_range: ModelRange = ModelRange.ALL
    ) -> Qualities:
        ones = xr.ones_like(grid.x)
        return QualitiesSchema.new(
            rho=ones * 2700.0,
            vp=ones * 6000.0,
            vs=ones * 3500.0,
            qp=ones * 200.0,
            qs=ones * 100.0,
            alpha=ones,
        )


def _with_coastline(grid: Grid, resources: dict[str, Path]) -> Grid:
    """Add the ``coastline`` coordinate that `ely` and `offshore` require.

    The coastline layer writes the coordinate onto the grid it receives, so
    calling it for that side effect is the honest way to get one.
    """
    config = CoastlineConfig(coastline=resources["coastline"])
    populated = grid.copy()
    layer_from_config(config)(config, grid.geometry, _Terminal())(populated)
    return populated


# ---------------------------------------------------------------------------
# Each layer on its own
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layer_type", sorted(COVERED))
def test_layer_preserves_the_site_label(
    layer_type: str, concrete_grid: Grid, resources: dict[str, Path]
) -> None:
    config = _configs(resources)[layer_type]
    grid = concrete_grid.copy()
    if "coastline" in config.requires:
        grid = _with_coastline(grid, resources)

    layer = layer_from_config(config)(config, grid.geometry, _Terminal())
    qualities = layer(grid)

    assert SITE in qualities.coords, layer_type
    assert list(qualities[SITE].values) == NAMES
    # A dropped label would show up as a reindex to NaN rather than an error.
    assert not np.isnan(qualities.vs.values).any(), layer_type


@pytest.mark.parametrize("layer_type", sorted(COVERED))
def test_layer_returns_the_grid_shape(
    layer_type: str, concrete_grid: Grid, resources: dict[str, Path]
) -> None:
    """A singleton j axis is easy to squeeze away by accident."""
    config = _configs(resources)[layer_type]
    grid = concrete_grid.copy()
    if "coastline" in config.requires:
        grid = _with_coastline(grid, resources)

    layer = layer_from_config(config)(config, grid.geometry, _Terminal())
    assert layer(grid).vs.shape == concrete_grid.x.shape, layer_type


# ---------------------------------------------------------------------------
# The whole chain, chunked
# ---------------------------------------------------------------------------


def test_full_chain_preserves_the_site_label(
    concrete_grid: Grid, resources: dict[str, Path]
) -> None:
    """Ordered as a real config: outermost first, `query` last."""
    configs = _configs(resources)
    pipeline = build_pipeline(
        concrete_grid.geometry,
        [
            configs["clamp"],
            configs["coastline"],
            configs["offshore"],
            configs["ely"],
            configs["backus"],
            configs["query"],
        ],
    )
    qualities = pipeline(concrete_grid.copy())

    assert list(qualities[SITE].values) == NAMES
    assert qualities.vs.shape == concrete_grid.x.shape


def test_full_chain_survives_map_blocks(
    borehole_grid: Grid, resources: dict[str, Path]
) -> None:
    """`execute_model_pipeline` compares each chunk against a template built
    from the grid, so a layer that drops the label fails the whole run."""
    from nzcvm.velocity_model import VelocityModel

    configs = _configs(resources)
    pipeline = build_pipeline(
        borehole_grid.geometry,
        [
            configs["clamp"],
            configs["coastline"],
            configs["offshore"],
            configs["ely"],
            configs["query"],
        ],
    )
    model = execute_model_pipeline(
        VelocityModel(grids={"boreholes": borehole_grid}, metadata=ModelMetadata()),
        pipeline,
    )

    qualities = model.qualities["boreholes"]
    assert list(qualities[SITE].values) == NAMES
    assert not np.isnan(qualities.vs.values).any()


# ---------------------------------------------------------------------------
# The failure mode this guards against
# ---------------------------------------------------------------------------


def test_a_numpy_terminal_drops_the_label(concrete_grid: Grid) -> None:
    """Documents why the preceding tests exist rather than trusting the contract.

    `nzcvm.layers.dummy.constant` builds its output from `np.ones`, so the
    result has no coordinates and `map_blocks` refuses it.
    """
    from nzcvm.layers.dummy import ConstantLayer

    qualities = ConstantLayer(vs=1234.0)(concrete_grid)
    assert SITE not in qualities.coords
