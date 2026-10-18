"""Tests for the site parameters the site database script publishes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nzcvm.components import Component
from nzcvm.coordinates import Coordinate

# The script lives outside the package, so load it from its path.
_SCRIPT = Path(__file__).parents[1] / "scripts" / "site_database.py"
_spec = importlib.util.spec_from_file_location("site_database", _SCRIPT)
assert _spec is not None and _spec.loader is not None
site_database = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(site_database)

DEPTHS = [0.0, 10.0, 20.0, 30.0, 40.0]


def _stations(*names: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": list(names),
            "longitude": 172.0 + np.arange(len(names)),
            "latitude": -43.0 - np.arange(len(names)),
        }
    )


def _profiles(
    stations: pd.DataFrame, depths: list[float], vs: list[list[float]]
) -> pd.DataFrame:
    """Profiles ordered by station then depth, as ``query_profiles`` returns."""
    return pd.DataFrame(
        {
            "station": np.repeat(stations["station"].to_numpy(), len(depths)),
            Coordinate.DEPTH: np.tile(depths, len(stations)),
            Component.VS: np.ravel(vs),
        }
    )


def _summarise(depths: list[float], vs: list[list[float]]) -> pd.DataFrame:
    stations = _stations(*(f"S{n:02d}" for n in range(len(vs))))
    return site_database.summarise(_profiles(stations, depths, vs), stations)


def test_vs30_of_a_uniform_profile_is_its_vs() -> None:
    summary = _summarise(DEPTHS, [[400.0] * 5])
    assert summary["vs30_m_per_s"].iloc[0] == pytest.approx(400.0)


def test_vs30_is_the_travel_time_average_over_30_m() -> None:
    # Each sample stands for the 10 m below it; the samples at 30 m and
    # below lie outside the average.
    summary = _summarise(DEPTHS, [[200.0, 400.0, 800.0, 1200.0, 3000.0]])
    expected = 30.0 / (10.0 / 200.0 + 10.0 / 400.0 + 10.0 / 800.0)
    assert summary["vs30_m_per_s"].iloc[0] == pytest.approx(expected)


def test_vs30_of_samples_coarser_than_30_m_is_the_surface_vs() -> None:
    summary = _summarise([0.0, 50.0, 100.0], [[300.0, 600.0, 900.0]])
    assert summary["vs30_m_per_s"].iloc[0] == pytest.approx(300.0)


def test_z1p0_and_z2p5_are_the_first_crossing_depths() -> None:
    summary = _summarise(DEPTHS, [[200.0, 400.0, 1000.0, 2000.0, 2500.0]])
    assert summary["z1p0_m"].iloc[0] == 20.0
    assert summary["z2p5_m"].iloc[0] == 40.0


def test_z1p0_is_zero_when_the_surface_already_reaches_it() -> None:
    summary = _summarise(DEPTHS, [[1500.0, 1200.0, 1800.0, 2600.0, 3000.0]])
    assert summary["z1p0_m"].iloc[0] == 0.0
    assert summary["z2p5_m"].iloc[0] == 30.0


def test_unreached_thresholds_are_nan() -> None:
    summary = _summarise(DEPTHS, [[300.0, 500.0, 700.0, 900.0, 990.0]])
    assert np.isnan(summary["z1p0_m"].iloc[0])
    assert np.isnan(summary["z2p5_m"].iloc[0])


def test_rows_follow_the_stations_and_their_own_profiles() -> None:
    # Names out of sorted order, so a sort anywhere would show up.
    stations = _stations("ZZZ", "AAA", "MMM")
    vs = [
        [400.0] * 5,
        [200.0, 400.0, 800.0, 1200.0, 3000.0],
        [1500.0, 1500.0, 1500.0, 1500.0, 1500.0],
    ]
    summary = site_database.summarise(_profiles(stations, DEPTHS, vs), stations)

    assert summary["station"].tolist() == ["ZZZ", "AAA", "MMM"]
    np.testing.assert_array_equal(summary["longitude"], stations["longitude"])
    np.testing.assert_array_equal(summary["latitude"], stations["latitude"])
    np.testing.assert_allclose(
        summary["vs30_m_per_s"],
        [400.0, 30.0 / (10.0 / 200.0 + 10.0 / 400.0 + 10.0 / 800.0), 1500.0],
    )
    np.testing.assert_array_equal(summary["z1p0_m"], [np.nan, 30.0, 0.0])
    np.testing.assert_array_equal(summary["z2p5_m"], [np.nan, 40.0, np.nan])
    np.testing.assert_array_equal(summary["profile_depth_m"], [40.0] * 3)


def test_profile_depth_is_the_last_sample() -> None:
    summary = _summarise([5.0, 25.0, 45.0, 65.0], [[500.0] * 4])
    assert summary["profile_depth_m"].iloc[0] == 65.0
