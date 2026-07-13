from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_orderflow_trophic_succession_alpha import (
    _phase_sum,
    build_phase,
    fit_quantile,
    role_scores,
    sequence_signals,
)


def _market(rows: int) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100.0 * np.exp(0.0001 * index + 0.001 * np.sin(index / 13.0))
    quote = 1_000_000.0 + 10_000.0 * np.sin(index / 17.0) + index
    trades = 1_000.0 + 20.0 * np.cos(index / 19.0)
    imbalance = 0.1 * np.sin(index / 11.0)
    return pd.DataFrame(
        {
            "close": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "quote_asset_volume": quote,
            "number_of_trades": trades,
            "taker_buy_quote": quote * (1.0 + imbalance) / 2.0,
        }
    )


def _role_frame(rows: int = 5) -> pd.DataFrame:
    frame: dict[str, np.ndarray] = {}
    for prefix in ("s", "c", "a"):
        frame[f"{prefix}_imbalance"] = np.ones(rows)
        frame[f"{prefix}_imbalance_z"] = np.ones(rows)
        frame[f"{prefix}_ticket_z"] = np.ones(rows)
        frame[f"{prefix}_intensity_z"] = np.ones(rows)
        frame[f"{prefix}_return_z"] = np.ones(rows)
        frame[f"{prefix}_impact"] = np.ones(rows)
        frame[f"{prefix}_impact_z"] = np.ones(rows)
        frame[f"{prefix}_clv"] = np.ones(rows)
    return pd.DataFrame(frame)


def test_phase_sum_uses_exact_completed_non_overlapping_window() -> None:
    values = pd.Series(np.arange(10, dtype=float))

    actual = _phase_sum(values, length=3, end_shift=2)

    assert np.isnan(actual.iloc[3])
    assert actual.iloc[4] == 0.0 + 1.0 + 2.0
    assert actual.iloc[7] == 3.0 + 4.0 + 5.0


def test_phase_features_prefix_does_not_depend_on_future_suffix() -> None:
    prefix = _market(2_500)
    suffix = _market(200) * 100.0
    full = pd.concat([prefix, suffix], ignore_index=True)

    expected = build_phase(prefix, length=6, end_shift=3, prefix="x")
    actual = build_phase(full, length=6, end_shift=3, prefix="x").iloc[: len(prefix)]

    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))


def test_fit_quantile_ignores_2023_selection_values() -> None:
    dates = pd.Series(pd.date_range("2020-06-01", "2023-12-31", freq="3h"))
    base = pd.Series(np.linspace(-1.0, 1.0, len(dates)))
    changed = base.copy()
    changed.loc[dates >= pd.Timestamp("2023-01-01")] = 1_000_000.0

    expected = fit_quantile(base, dates, 0.90)
    actual = fit_quantile(changed, dates, 0.90)

    assert actual == expected


def test_role_scores_reward_large_sponsor_then_small_busy_crowd() -> None:
    features = _role_frame(1)
    features.loc[0, ["s_ticket_z", "s_imbalance_z", "s_return_z"]] = [2.0, 2.0, 2.0]
    features.loc[0, "s_intensity_z"] = -1.0
    features.loc[0, ["c_ticket_z", "c_imbalance_z", "c_intensity_z", "c_return_z"]] = [-1.0, 2.0, 2.0, 2.0]
    features.loc[0, ["s_impact", "c_impact"]] = [2.0, 0.5]

    _, sponsor, crowd, _ = role_scores(features)
    _, swapped_sponsor, swapped_crowd, _ = role_scores(features, ticket_role_swap=True)

    assert sponsor[0] > swapped_sponsor[0]
    assert crowd[0] > swapped_crowd[0]


def test_sequence_branches_and_flip_have_exact_opposite_mapping() -> None:
    features = _role_frame(5)
    features.loc[:, "a_impact_z"] = [10.0, -10.0, 10.0, -10.0, 10.0]
    thresholds = {"sponsor_role": 0.0, "crowd_role": 0.0, "absorption_role": 0.0}

    cont_long, cont_short, _ = sequence_signals(features, thresholds, "continuation")
    rev_long, rev_short, _ = sequence_signals(features, thresholds, "absorption_reversal")
    flip_long, flip_short, _ = sequence_signals(features, thresholds, "continuation", flip=True)

    np.testing.assert_array_equal(cont_long, flip_short)
    np.testing.assert_array_equal(cont_short, flip_long)
    assert not np.any(cont_long & cont_short)
    assert not np.any(rev_long & rev_short)
    assert np.any(cont_long)
    assert np.any(rev_short)
