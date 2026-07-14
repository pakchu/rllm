from __future__ import annotations

import numpy as np
import pandas as pd

from training.search_confirmed_pullback_squeeze_alpha import fit_confirmation_masks
from training.search_specific_pullback_squeeze_alpha import fit_rule_masks, simulate_mask


def _feature_fixture(rows: int = 5_000) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    rng = np.random.default_rng(19)
    dates = pd.Series(pd.date_range("2020-01-01", periods=rows, freq="h"))
    features = pd.DataFrame(
        {
            "funding_available": np.ones(rows),
            "funding_rate": rng.normal(0.0, 1.0, rows),
            "trend_96": rng.normal(0.0, 1.0, rows),
            "premium_available": np.ones(rows),
            "premium_index_change": rng.normal(0.0, 1.0, rows),
            "htf_1d_return_4": rng.normal(0.0, 1.0, rows),
            "rex_576_range_pos": rng.uniform(0.0, 1.0, rows),
            "htf_1d_return_1": rng.normal(0.0, 1.0, rows),
            "htf_3d_return_1": rng.normal(0.0, 1.0, rows),
        }
    )
    return features, dates, np.ones(rows, dtype=bool)


def test_rule_thresholds_ignore_post_fit_rows() -> None:
    features, dates, decision = _feature_fixture()
    fit_end = dates.iloc[4_000]
    original = fit_rule_masks(
        features,
        dates,
        decision,
        fit_start=dates.iloc[0],
        fit_end=fit_end,
    )

    changed = features.copy()
    changed.loc[4_000:, :] = 1_000_000.0
    replay = fit_rule_masks(
        changed,
        dates,
        decision,
        fit_start=dates.iloc[0],
        fit_end=fit_end,
    )

    assert replay["base_thresholds"] == original["base_thresholds"]
    assert replay["context_thresholds"] == original["context_thresholds"]
    np.testing.assert_array_equal(replay["active"][:4_000], original["active"][:4_000])


def test_rule_requires_availability_and_source_specific_overheat_checks() -> None:
    features, dates, decision = _feature_fixture()
    fitted = fit_rule_masks(
        features,
        dates,
        decision,
        fit_start=dates.iloc[0],
        fit_end=dates.iloc[4_000],
    )
    thresholds = fitted["context_thresholds"]

    funding_positions = np.flatnonzero(fitted["funding_active"])
    premium_positions = np.flatnonzero(fitted["premium_active"])
    assert len(funding_positions) > 0
    assert len(premium_positions) > 0
    assert (features.loc[funding_positions, "funding_available"] > 0.5).all()
    assert (features.loc[premium_positions, "premium_available"] > 0.5).all()
    assert (features.loc[funding_positions, "htf_1d_return_1"] <= thresholds["funding_daily_overheat"]).all()
    assert (features.loc[premium_positions, "htf_3d_return_1"] <= thresholds["premium_3d_overheat"]).all()
    assert (features.loc[np.flatnonzero(fitted["active"]), "rex_576_range_pos"] <= thresholds["rex_576_range_pos"]).all()

    unavailable = features.copy()
    unavailable.loc[4_000:, ["funding_available", "premium_available"]] = 0.0
    no_future_signals = fit_rule_masks(
        unavailable,
        dates,
        decision,
        fit_start=dates.iloc[0],
        fit_end=dates.iloc[4_000],
    )
    assert not no_future_signals["active"][4_000:].any()


def test_simulator_uses_next_bar_purges_period_exit_and_prevents_overlap() -> None:
    rows = 20
    dates = pd.Series(pd.date_range("2024-01-01", periods=rows, freq="5min"))
    opens = np.linspace(100.0, 119.0, rows)
    market = pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": opens * 1.02,
            "low": opens * 0.98,
            "close": opens,
        }
    )
    active = np.zeros(rows, dtype=bool)
    active[[1, 2, 8, 17]] = True

    result = simulate_mask(
        market,
        dates,
        active,
        start=dates.iloc[0],
        end=dates.iloc[-1],
        hold_bars=3,
        entry_delay_bars=1,
        leverage=0.5,
        fee_rate=0.0005,
        slippage_rate=0.0001,
    )

    assert result["signal_positions"] == [1, 8]
    assert result["trade_count"] == 2
    assert result["absolute_return_pct"] > 0.0
    assert result["strict_mdd_pct"] > 0.0


def test_confirmation_thresholds_are_fit_only_on_candidate_history() -> None:
    features, dates, decision = _feature_fixture()
    features["bb_z"] = np.linspace(-2.0, 2.0, len(features))
    features["quote_vol_z_1d"] = np.linspace(-1.0, 3.0, len(features))
    base_active = np.ones(len(features), dtype=bool)
    fit_end = dates.iloc[4_000]
    original = fit_confirmation_masks(
        features,
        dates,
        base_active,
        fit_start=dates.iloc[0],
        fit_end=fit_end,
    )

    changed = features.copy()
    changed.loc[4_000:, ["bb_z", "quote_vol_z_1d"]] = 1_000_000.0
    replay = fit_confirmation_masks(
        changed,
        dates,
        base_active,
        fit_start=dates.iloc[0],
        fit_end=fit_end,
    )

    assert replay["thresholds"] == original["thresholds"]
    np.testing.assert_array_equal(replay["active"][:4_000], original["active"][:4_000])
    active_positions = np.flatnonzero(original["active"])
    assert (features.loc[active_positions, "bb_z"] <= original["thresholds"]["bb_z_q"]).all()
    assert (
        features.loc[active_positions, "quote_vol_z_1d"]
        <= original["thresholds"]["quote_vol_z_1d_q"]
    ).all()
