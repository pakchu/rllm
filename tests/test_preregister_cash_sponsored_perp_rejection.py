from __future__ import annotations

import numpy as np
import pandas as pd

from training import preregister_cash_sponsored_perp_rejection as cspr


def _frame() -> pd.DataFrame:
    rows = 20
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "quarantined": False,
            "perp_quarantined": False,
            "spot_quarantined": False,
            "spot_source_complete": True,
            "spot_signed_quote_notional": 10.0,
            "spot_micro_log_return": 0.01,
            "spot_flow_coherence": 0.8,
            "spot_buyer_execution_centroid": 99.0,
            "spot_seller_execution_centroid": 100.0,
            "spot_close": 101.0,
            "signed_quote_notional": -10.0,
            "signed_event_imbalance": -0.5,
            "micro_log_return": 0.01,
            "flow_coherence": 0.8,
            "agg_trade_count": 100,
        }
    )
    return frame


def test_prior_quantile_excludes_current_value() -> None:
    values = pd.Series([1.0, 2.0, 100.0])
    clean = pd.Series(True, index=values.index)
    result = cspr.prior_quantile(values, clean, quantile=0.5, window=2, min_periods=2)
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 1.5


def test_primary_classification_is_mirror_symmetric() -> None:
    frame = _frame()
    cfg = cspr.Config(baseline_bars=4, baseline_min_periods=2)
    _, controls = cspr.classify_events(frame, cfg, quantile=0.5)
    assert controls["primary"].iloc[-1]

    mirrored = frame.copy()
    mirrored["spot_signed_quote_notional"] *= -1
    mirrored["spot_micro_log_return"] *= -1
    mirrored["spot_buyer_execution_centroid"] = 101.0
    mirrored["spot_seller_execution_centroid"] = 100.0
    mirrored["spot_close"] = 99.0
    mirrored["signed_quote_notional"] *= -1
    mirrored["signed_event_imbalance"] *= -1
    mirrored["micro_log_return"] *= -1
    signal, controls = cspr.classify_events(mirrored, cfg, quantile=0.5)
    assert controls["primary"].iloc[-1]
    assert signal["side"].iloc[-1] == -1


def test_missing_spot_or_centroid_fails_closed() -> None:
    frame = _frame()
    cfg = cspr.Config(baseline_bars=4, baseline_min_periods=2)
    frame.loc[19, "spot_buyer_execution_centroid"] = np.nan
    _, controls = cspr.classify_events(frame, cfg, quantile=0.5)
    assert not controls["primary"].iloc[-1]
    frame.loc[19, "spot_buyer_execution_centroid"] = 99.0
    frame.loc[19, "quarantined"] = True
    _, controls = cspr.classify_events(frame, cfg, quantile=0.5)
    assert not controls["primary"].iloc[-1]


def test_component_ablations_are_supersets_of_primary() -> None:
    frame = _frame()
    cfg = cspr.Config(baseline_bars=4, baseline_min_periods=2)
    _, controls = cspr.classify_events(frame, cfg, quantile=0.5)
    assert (controls["primary"] & ~controls["no_centroid"]).sum() == 0
    assert (controls["primary"] & ~controls["no_perp_event_confirmation"]).sum() == 0


def test_lag_placebo_does_not_depend_on_current_spot_values() -> None:
    frame = _frame()
    cfg = cspr.Config(baseline_bars=4, baseline_min_periods=2)
    _, original = cspr.classify_events(frame, cfg, quantile=0.5)
    changed = frame.copy()
    changed.loc[19, "spot_signed_quote_notional"] = -999.0
    changed.loc[19, "spot_micro_log_return"] = -1.0
    changed.loc[19, "spot_flow_coherence"] = 0.0
    changed.loc[19, "spot_quarantined"] = True
    changed.loc[19, "quarantined"] = True
    _, altered = cspr.classify_events(changed, cfg, quantile=0.5)
    assert original["spot_lag_1h"].iloc[19] == altered["spot_lag_1h"].iloc[19]


def test_direction_flip_and_delay_controls_reserve_the_frozen_clock() -> None:
    frame = _frame()
    cfg = cspr.Config(baseline_bars=4, baseline_min_periods=2)
    _, controls = cspr.classify_events(frame, cfg, quantile=0.5)
    assert controls["direction_flip"].equals(controls["primary"])
    assert controls["signal_delay_1bar"].iloc[-1] == controls["primary"].iloc[-2]


def test_support_summary_enforces_both_sides_and_halves() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="180D"),
            "quarantined": False,
        }
    )
    schedule = pd.DataFrame(
        {
            "entry_date": frame["date"].astype(str),
            "side": [1, -1] * 5,
        }
    )
    cfg = cspr.Config(
        minimum_nonoverlap_total=1,
        minimum_nonoverlap_per_year=0,
        minimum_nonoverlap_per_2023_half=0,
        minimum_side_share=0.25,
    )
    result = cspr._support(schedule, frame, cfg)
    assert result["long_share"] == 0.5
    assert result["short_share"] == 0.5
    assert result["passes_count_support"]


def test_future_feature_gap_does_not_cancel_entered_fixed_hold() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=20, freq="5min"),
            "quarantined": [False] * 2 + [True] * 18,
        }
    )
    signal = pd.DataFrame(
        {
            "side": [1] + [0] * 19,
            "branch": ["cash_sponsored_rejection"] + ["none"] * 19,
            "hold_bars": [12] + [0] * 19,
        }
    )
    schedule = cspr.nonoverlapping_schedule(signal, frame)
    assert len(schedule) == 1
    assert schedule.loc[0, "exit_position"] == 13


def test_jaccard_handles_empty_and_partial_overlap() -> None:
    assert cspr._jaccard(pd.Series([False]), pd.Series([False])) == 0.0
    assert cspr._jaccard(
        pd.Series([True, True, False]), pd.Series([True, False, True])
    ) == 1.0 / 3.0
