from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_cftc_concentration_topology_alpha import (
    concentration_features,
    fit_threshold,
    release_signals,
)


def _reports(rows: int, start: str = "2019-01-01") -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "release_time": pd.date_range(start, periods=rows, freq="7D"),
            "conc_net_le_4_tdr_long_all": 45.0 + 3.0 * np.sin(index / 7.0),
            "conc_net_le_4_tdr_short_all": 35.0 + 2.0 * np.cos(index / 9.0),
            "conc_net_le_8_tdr_long_all": 70.0 + 2.0 * np.sin(index / 11.0),
            "conc_net_le_8_tdr_short_all": 65.0 + 2.0 * np.cos(index / 13.0),
            "traders_tot_all": 100.0 + index / 10.0 + 3.0 * np.sin(index / 5.0),
        }
    )


def test_side_swap_negates_topology_and_preserves_fragility() -> None:
    reports = _reports(220)

    base = concentration_features(reports, topology="rank_curvature", breadth_horizon=4)
    swapped = concentration_features(
        reports,
        topology="rank_curvature",
        breadth_horizon=4,
        swap_sides=True,
    )
    finite = base["topology_z"].notna() & swapped["topology_z"].notna()

    np.testing.assert_allclose(swapped.loc[finite, "topology_z"], -base.loc[finite, "topology_z"])
    np.testing.assert_allclose(swapped.loc[finite, "fragility"], base.loc[finite, "fragility"])
    np.testing.assert_array_equal(
        swapped.loc[finite, "concentrated_side"],
        -base.loc[finite, "concentrated_side"],
    )


def test_concentration_feature_prefix_is_future_suffix_independent() -> None:
    prefix = _reports(220)
    suffix = _reports(30, start="2023-04-25")
    suffix.loc[:, "conc_net_le_4_tdr_long_all"] = 80.0
    suffix.loc[:, "conc_net_le_8_tdr_long_all"] = 95.0
    suffix.loc[:, "traders_tot_all"] = 1_000_000.0
    full = pd.concat([prefix, suffix], ignore_index=True)

    expected = concentration_features(prefix, topology="rank_odds", breadth_horizon=13)
    actual = concentration_features(full, topology="rank_odds", breadth_horizon=13).iloc[: len(prefix)]

    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))


def test_fit_threshold_ignores_2023_values() -> None:
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="7D")
    base = pd.DataFrame({"release_time": dates, "fragility": np.linspace(0.0, 1.0, len(dates))})
    changed = base.copy()
    changed.loc[changed.release_time >= "2023-01-01", "fragility"] = 1_000_000.0

    assert fit_threshold(changed, 0.70) == fit_threshold(base, 0.70)


def test_release_signal_uses_first_bar_at_or_after_release_and_mapping_flips() -> None:
    dates = pd.Series(pd.date_range("2022-01-01", periods=12, freq="5min"))
    features = pd.DataFrame(
        {
            "release_time": [pd.Timestamp("2022-01-01 00:07"), pd.Timestamp("2022-01-01 00:31")],
            "fragility": [2.0, 2.0],
            "concentrated_side": [1.0, -1.0],
        }
    )

    fade_long, fade_short = release_signals(features, dates, 1.0, "fade")
    follow_long, follow_short = release_signals(features, dates, 1.0, "follow")

    np.testing.assert_array_equal(np.flatnonzero(fade_short), [2])
    np.testing.assert_array_equal(np.flatnonzero(fade_long), [7])
    np.testing.assert_array_equal(fade_long, follow_short)
    np.testing.assert_array_equal(fade_short, follow_long)
