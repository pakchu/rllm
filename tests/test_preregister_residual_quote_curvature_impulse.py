from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import preregister_residual_quote_curvature_impulse as rqci


def _small_cfg(**changes: object) -> rqci.Config:
    values: dict[str, object] = {
        "baseline_window_bars": 8,
        "baseline_minimum_bars": 4,
        "threshold_quantiles": (0.75, 0.50),
        "hold_bars": 3,
        "minimum_nonoverlap_total": 1,
        "minimum_nonoverlap_per_half": 0,
        "minimum_nonoverlap_per_quarter": 0,
        "minimum_side_share": 0.0,
        "maximum_quarter_share": 1.0,
    }
    values.update(changes)
    return replace(rqci.Config(), **values)


def _frame(rows: int = 30) -> pd.DataFrame:
    base = np.linspace(0.0, 0.003, rows)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "center_quote_median": np.linspace(100.0, 101.0, rows),
            "source_complete": True,
            "skew_2_median": base,
            "skew_3_median": base + np.linspace(0.0, 0.001, rows),
            "skew_4_median": base + np.linspace(0.0, 0.0015, rows),
            "skew_5_median": base + np.linspace(0.0, 0.004, rows),
        }
    )


def test_curvature_is_outer_slope_minus_inner_slope() -> None:
    skew = pd.DataFrame(
        {
            "skew_2_median": [0.0, 0.0],
            "skew_3_median": [1.0, 2.0],
            "skew_4_median": [2.0, 2.0],
            "skew_5_median": [5.0, 1.0],
        }
    )
    assert rqci.curvature(skew).tolist() == [2.0, -3.0]


def test_curvature_features_require_complete_t_minus_six_through_t() -> None:
    frame = _frame()
    frame.loc[10, "source_complete"] = False
    frame.loc[
        10,
        ["center_quote_median", *rqci.shared.SKEW_COLUMNS],
    ] = np.nan
    features = rqci.curvature_features(frame, _small_cfg())
    assert not bool(features.loc[10:16, "source_streak_complete"].any())
    assert bool(features.loc[17, "source_streak_complete"])


def test_curvature_features_are_prefix_invariant() -> None:
    frame = _frame(40)
    cfg = _small_cfg()
    first = rqci.curvature_features(frame, cfg)
    changed = frame.copy()
    changed.loc[35:, "skew_5_median"] = 1_000_000.0
    second = rqci.curvature_features(changed, cfg)
    pd.testing.assert_frame_equal(first.loc[:34], second.loc[:34])


def test_signal_requires_dominance_quiet_center_and_threshold_crossing() -> None:
    rows = 20
    frame = pd.DataFrame(
        {"date": pd.date_range("2023-01-01", periods=rows, freq="5min")}
    )
    features = pd.DataFrame(
        {
            "residual_impulse": [3.0, 2.0, 1.0, 1.0, 10.0] + [1.0] * 15,
            "residual_dominance": [1.0] * rows,
            "center_move_30m": [0.0] * rows,
            "quiet_center_threshold": [1.0] * rows,
        }
    )
    cfg = _small_cfg(baseline_window_bars=4, baseline_minimum_bars=3)
    signal = rqci.build_signal(frame, features, cfg, quantile=0.75)
    assert bool(signal.loc[4, "candidate"])
    assert signal.loc[4, "side"] == 1
    assert signal.loc[4, "hold_bars"] == 3

    weak = features.copy()
    weak["residual_dominance"] = 0.249
    assert not bool(
        rqci.build_signal(frame, weak, cfg, quantile=0.75)["candidate"].any()
    )
    loud = features.copy()
    loud["center_move_30m"] = 2.0
    assert not bool(
        rqci.build_signal(frame, loud, cfg, quantile=0.75)["candidate"].any()
    )


def test_signal_is_prefix_invariant() -> None:
    rows = 30
    frame = pd.DataFrame(
        {"date": pd.date_range("2023-01-01", periods=rows, freq="5min")}
    )
    features = pd.DataFrame(
        {
            "residual_impulse": np.sin(np.arange(rows, dtype=float)),
            "residual_dominance": [1.0] * rows,
            "center_move_30m": [0.0] * rows,
            "quiet_center_threshold": [1.0] * rows,
        }
    )
    cfg = _small_cfg()
    first = rqci.build_signal(frame, features, cfg, quantile=0.75)
    changed = features.copy()
    changed.loc[25:, "residual_impulse"] = 1_000_000.0
    second = rqci.build_signal(frame, changed, cfg, quantile=0.75)
    pd.testing.assert_frame_equal(first.loc[:24], second.loc[:24])


def test_support_summary_enforces_frozen_incidence_distribution() -> None:
    rows = []
    for quarter in rqci.shared.QUARTERS:
        for number in range(45):
            rows.append({"quarter": quarter, "side": 1 if number % 2 else -1})
    schedule = pd.DataFrame(rows)
    summary = rqci.support_summary(schedule, rqci.Config())
    assert summary["nonoverlap_total"] == 180
    assert summary["h1"] == 90
    assert summary["h2"] == 90
    assert summary["passes_incidence"] is True

    concentrated = schedule.loc[~schedule["quarter"].eq("q4")].copy()
    assert rqci.support_summary(concentrated, rqci.Config())[
        "passes_incidence"
    ] is False


def test_event_clock_hash_binds_calendar_side_and_quantile() -> None:
    schedule = pd.DataFrame(
        {
            "quarter": ["q1"],
            "signal_position": [10],
            "entry_position": [11],
            "exit_position": [35],
            "signal_date": ["2023-01-01 00:50:00"],
            "entry_date": ["2023-01-01 00:55:00"],
            "exit_date": ["2023-01-01 02:55:00"],
            "side": [1],
            "branch": ["x"],
            "hold_bars": [24],
        }
    )
    baseline = rqci.event_clock_hash(schedule, selected_quantile=0.99)
    changed = schedule.copy()
    changed.loc[0, "side"] = -1
    assert rqci.event_clock_hash(changed, selected_quantile=0.99) != baseline
    changed = schedule.copy()
    changed.loc[0, "quarter"] = "q2"
    assert rqci.event_clock_hash(changed, selected_quantile=0.99) != baseline
    assert rqci.event_clock_hash(schedule, selected_quantile=0.975) != baseline


def test_fixed_book_null_suite_has_zero_rqci_events() -> None:
    result = rqci.synthetic_control(rqci.Config())
    assert result["passes"] is True
    assert all(
        trial == {"raw_events": 0, "nonoverlap_events": 0}
        for scenario in result["scenarios"].values()
        for trial in scenario.values()
    )


def test_synthetic_raw_false_event_fails_before_source_load(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5, freq="5min"),
            "source_complete": True,
        }
    )
    monkeypatch.setattr(
        rqci.shared,
        "synthetic_null_suite",
        lambda: {"tail_only": frame},
    )
    monkeypatch.setattr(
        rqci,
        "curvature_features",
        lambda frame, cfg: pd.DataFrame(index=frame.index),
    )

    def tail_signal(
        frame: pd.DataFrame,
        features: pd.DataFrame,
        cfg: rqci.Config,
        *,
        quantile: float,
    ) -> pd.DataFrame:
        del features, quantile
        signal = pd.DataFrame(
            {
                "candidate": [False] * 4 + [True],
                "side": [0, 0, 0, 0, 1],
                "branch": ["none"] * 4 + ["false_event"],
                "hold_bars": [0] * 4 + [cfg.hold_bars],
            }
        )
        return signal

    monkeypatch.setattr(rqci, "build_signal", tail_signal)
    synthetic = rqci.synthetic_control(rqci.Config())
    assert synthetic["passes"] is False
    assert all(
        trial == {"raw_events": 1, "nonoverlap_events": 0}
        for trial in synthetic["scenarios"]["tail_only"].values()
    )

    monkeypatch.setattr(rqci, "synthetic_control", lambda cfg: synthetic)

    def forbidden_source_load() -> None:
        raise AssertionError("real source must remain unopened")

    monkeypatch.setattr(rqci.shared, "load_source", forbidden_source_load)
    result, clock = rqci.run_support(rqci.Config())
    assert result["source_loaded"] is False
    assert clock is None


def test_protocol_config_and_shared_utility_are_frozen() -> None:
    cfg = rqci.Config()
    rqci._validate_config(cfg)
    assert rqci.sha256_file(rqci.SHARED_SOURCE) == rqci.SHARED_SOURCE_SHA256
    payload = rqci.protocol(cfg)
    assert payload["outcomes_opened"] is False
    assert payload["external_ohlc_funding_return_or_equity_loaded"] is False
    assert payload["selection_end_exclusive"] == "2024-01-01 00:00:00"
    assert payload["side"].endswith("negative short")
    with pytest.raises(ValueError, match="configuration is frozen"):
        rqci._validate_config(replace(cfg, hold_bars=25))
