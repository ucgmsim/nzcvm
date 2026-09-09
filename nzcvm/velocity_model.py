import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Self

import shapely
import xarray as xr

from nzcvm.config.metadata import ModelMetadata
from nzcvm.config.velocity_model import VelocityModelConfig
from nzcvm.coordinates import Coordinate
from nzcvm.grids.builder import build_grids_from_config
from nzcvm.grids.grid import Grid, GridSchema
from nzcvm.qualities import Qualities, QualitiesSchema


@dataclass(frozen=True)
class VelocityModel:
    grids: dict[str, Grid]
    metadata: ModelMetadata
    qualities: dict[str, Qualities] = field(default_factory=dict)

    def __post_init__(self):
        if self.qualities and set(self.qualities) != set(self.grids):
            qualities = sorted(self.qualities)
            grids = sorted(self.grids)

            raise ValueError(
                f"Length of assigned qualities ({', '.join(qualities)}) "
                f"does not match number of grids ({', '.join(grids)})"
            )

    @property
    def geometry(self) -> shapely.Geometry:
        return shapely.union_all([grid.geometry for grid in self.grids.values()])

    @classmethod
    def from_config(cls, config: VelocityModelConfig) -> Self:
        grids = build_grids_from_config(config.grid)
        return cls(grids=grids, metadata=config.metadata)

    @property
    def pairwise(self) -> dict[str, tuple[Grid, Qualities]]:
        if not self.qualities:
            # No need to recheck this dataset because: (a) the velocity model
            # never mutates, and (b) __post_init__ already checks the invariant
            # set(self.qualities) == set(self.grids).
            raise ValueError(
                "Cannot traverse pairwise when assigned qualities are empty"
            )
        quality_keys = set(self.qualities)
        return {
            k: (self.grids[k], self.qualities[k])
            for k in self.grids
            if k in quality_keys
        }

    @classmethod
    def from_datatree(cls, dtree: xr.DataTree) -> Self:
        qualities = {
            k: QualitiesSchema.from_dataset(v.to_dataset())
            for k, v in dtree["qualities"].children.items()
        }
        grids = {
            k: GridSchema.from_dataset(v.to_dataset())
            for k, v in dtree["grids"].children.items()
        }
        metadata = ModelMetadata.from_dict(dtree.attrs)
        return cls(grids=grids, qualities=qualities, metadata=metadata)

    def map(self, f: Callable[[xr.Dataset], xr.Dataset]) -> Self:
        grids = {k: f(grid) for k, grid in self.grids.items()}
        qualities = {k: f(qualities) for k, qualities in self.qualities.items()}
        return dataclasses.replace(self, grids=grids, qualities=qualities)

    def orient(self, *coordinates: Coordinate) -> Self:
        return self.map(lambda dset: dset.transpose(*coordinates))

    def flip(self, coordinate: Coordinate) -> Self:
        return self.map(lambda dset: dset.sel({coordinate: slice(None, None, -1)}))

    def to_datatree(self) -> xr.DataTree:
        dtree = xr.DataTree.from_dict(
            {"grids": self.grids, "qualities": self.qualities},
            nested=True,
        )
        dtree.attrs = self.metadata.to_dict()
        return dtree
