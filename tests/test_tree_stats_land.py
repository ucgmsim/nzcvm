"""Tests for how ``nzcvm tree-stats`` reads the New Zealand land polygon.

Natural Earth serves the real polygon. These tests replace
``geopandas.read_file`` with a small stand-in world and run offline.
"""

import geopandas as gpd
import pytest
import shapely

from nzcvm.scripts import tree_stats

NATURAL_EARTH_URL = (
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
)


@pytest.fixture()
def world_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve a two-country world in WGS84 and record the URLs read."""
    world = gpd.GeoDataFrame(
        {"ADMIN": ["Australia", "New Zealand", "New Zealand"]},
        geometry=[
            shapely.box(140.0, -30.0, 150.0, -20.0),
            # Boxes around Wellington and Auckland.
            shapely.box(174.7, -41.4, 174.9, -41.2),
            shapely.box(174.6, -37.0, 174.9, -36.8),
        ],
        crs="EPSG:4326",
    )
    urls = []

    def read_file(url):
        urls.append(url)
        return world

    monkeypatch.setattr(gpd, "read_file", read_file)
    return urls


def test_read_nz_land_polygon_selects_new_zealand_in_nztm(
    world_reads: list[str],
) -> None:
    polygon = tree_stats._read_nz_land_polygon()

    assert world_reads == [NATURAL_EARTH_URL]
    # The query merges both New Zealand pieces into one geometry.
    assert polygon.geom_type == "MultiPolygon"
    assert len(polygon.geoms) == 2
    # The geometry is in NZTM metres, and excludes Australia.
    min_x, min_y, max_x, max_y = polygon.bounds
    assert 1_740_000 < min_x < max_x < 1_770_000
    assert 5_410_000 < min_y < max_y < 5_930_000
    # Wellington and Auckland, in NZTM.
    assert polygon.contains(shapely.Point(1_749_000, 5_428_000))
    assert polygon.contains(shapely.Point(1_757_000, 5_920_000))
    assert not polygon.contains(shapely.Point(1_750_000, 5_700_000))
