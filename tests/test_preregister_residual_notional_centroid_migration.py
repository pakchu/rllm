from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import preregister_residual_notional_centroid_migration as rncm


def _small_cfg(**changes: object) -> rncm.Config:
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
    return replace(rncm.Config(), **values)


def test_prior_quantile_excludes_current_value() -> None:
    values = pd.Series([1.0, 2.0, 100.0])
    threshold = rncm.prior_quantile(
        values,
        quantile=0.5,
        window=2,
        minimum=2,
    )
    assert pd.isna(threshold.iloc[1])
    assert threshold.iloc[2] == pytest.approx(1.5)


def test_prior_beta_excludes_current_pair_and_recovers_slope() -> None:
    driver = pd.Series(np.arange(1.0, 9.0))
    response = 3.0 * driver + 7.0
    first = rncm.prior_beta(response, driver, window=6, minimum=4)
    changed = response.copy()
    changed.iloc[-1] = -1_000_000.0
    second = rncm.prior_beta(changed, driver, window=6, minimum=4)
    assert first.iloc[-1] == pytest.approx(3.0)
    assert second.iloc[-1] == pytest.approx(first.iloc[-1])


def test_migration_requires_every_source_row_from_t_minus_six_to_t() -> None:
    rows = 30
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "center_quote_median": np.linspace(100.0, 101.0, rows),
            "source_complete": True,
            **{
                column: np.linspace(0.0, 0.01 * number, rows)
                for number, column in enumerate(rncm.SKEW_COLUMNS, start=1)
            },
        }
    )
    frame.loc[10, "source_complete"] = False
    frame.loc[10, ["center_quote_median", *rncm.SKEW_COLUMNS]] = np.nan
    features = rncm.migration_features(frame, _small_cfg())
    assert not bool(features.loc[10:16, "source_streak_complete"].any())
    assert bool(features.loc[17, "source_streak_complete"])


def test_signal_is_prefix_invariant_and_requires_residual_dominance() -> None:
    rows = 20
    frame = pd.DataFrame(
        {"date": pd.date_range("2023-01-01", periods=rows, freq="5min")}
    )
    features = pd.DataFrame(
        {
            "intensity": [1.0, 2.0, 1.0, 2.0, 10.0] + [1.0] * 15,
            "residual_dominance": [1.0] * rows,
            "center_move_30m": [0.0] * rows,
            "quiet_center_threshold": [1.0] * rows,
            "coherent_positive": [True] * rows,
            "coherent_negative": [False] * rows,
        }
    )
    cfg = _small_cfg(baseline_window_bars=4, baseline_minimum_bars=3)
    first = rncm.build_signal(frame, features, cfg, quantile=0.75)
    changed = features.copy()
    changed.loc[10:, "intensity"] = 1_000_000.0
    second = rncm.build_signal(frame, changed, cfg, quantile=0.75)
    pd.testing.assert_frame_equal(first.loc[:9], second.loc[:9])

    weak = features.copy()
    weak["residual_dominance"] = 0.249
    rejected = rncm.build_signal(frame, weak, cfg, quantile=0.75)
    assert not bool(rejected["candidate"].any())

    loud = features.copy()
    loud["center_move_30m"] = 2.0
    assert not bool(
        rncm.build_signal(frame, loud, cfg, quantile=0.75)["candidate"].any()
    )


def test_quarterly_schedule_is_next_open_fixed_hold_and_nonoverlapping() -> None:
    rows = 200
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            # A future source gap must not retroactively cancel a selected event.
            "source_complete": [True] * 5 + [False] + [True] * (rows - 6),
        }
    )
    signal = pd.DataFrame(
        {
            "side": np.zeros(rows, dtype=np.int8),
            "hold_bars": np.zeros(rows, dtype=np.int16),
            "branch": ["none"] * rows,
        }
    )
    for position, side in ((1, 1), (2, -1), (5, -1), (76, -1)):
        signal.loc[position, "side"] = side
        signal.loc[position, "hold_bars"] = 72
        signal.loc[position, "branch"] = "event"
    schedule = rncm.quarterly_nonoverlap_schedule(signal, frame)
    assert schedule["signal_position"].tolist() == [1, 76]
    assert schedule["entry_position"].tolist() == [2, 77]
    assert schedule["exit_position"].tolist() == [74, 149]
    assert schedule["side"].tolist() == [1, -1]


def test_support_summary_enforces_balance_and_quarter_distribution() -> None:
    rows = []
    for quarter in rncm.QUARTERS:
        for number in range(30):
            rows.append({"quarter": quarter, "side": 1 if number % 2 else -1})
    schedule = pd.DataFrame(rows)
    summary = rncm.support_summary(schedule, rncm.Config())
    assert summary["nonoverlap_total"] == 120
    assert summary["h1"] == 60
    assert summary["h2"] == 60
    assert summary["passes_incidence"] is True

    concentrated = schedule.loc[~schedule["quarter"].eq("q4")].copy()
    assert rncm.support_summary(concentrated, rncm.Config())[
        "passes_incidence"
    ] is False


def test_support_stopping_rule_selects_strictest_passing_quantile() -> None:
    trials = [
        {"quantile": 0.995, "support": {"passes_incidence": False}},
        {"quantile": 0.99, "support": {"passes_incidence": True}},
        {"quantile": 0.975, "support": {"passes_incidence": True}},
    ]
    assert rncm.select_strictest_passing(trials) is trials[1]


def test_tolerant_event_jaccard_uses_one_to_one_matches() -> None:
    result = rncm.tolerant_event_jaccard(
        [10, 20, 100],
        [11, 12, 24, 200],
        tolerance_bars=5,
    )
    assert result == {
        "first_event_count": 3,
        "second_event_count": 4,
        "matched_event_count": 2,
        "tolerance_bars": 5,
        "jaccard": 0.4,
    }


def test_event_clock_hash_binds_side_and_execution_positions() -> None:
    schedule = pd.DataFrame(
        {
            "quarter": ["q1"],
            "signal_position": [10],
            "entry_position": [11],
            "exit_position": [83],
            "signal_date": ["2023-01-01 00:50:00"],
            "entry_date": ["2023-01-01 00:55:00"],
            "exit_date": ["2023-01-01 06:55:00"],
            "side": [1],
            "branch": ["x"],
            "hold_bars": [72],
        }
    )
    baseline = rncm.rncm_event_clock_hash(schedule, selected_quantile=0.99)
    changed = schedule.copy()
    changed.loc[0, "side"] = -1
    assert rncm.rncm_event_clock_hash(changed, selected_quantile=0.99) != baseline
    changed = schedule.copy()
    changed.loc[0, "exit_position"] = 84
    assert rncm.rncm_event_clock_hash(changed, selected_quantile=0.99) != baseline
    changed = schedule.copy()
    changed.loc[0, "quarter"] = "q2"
    assert rncm.rncm_event_clock_hash(changed, selected_quantile=0.99) != baseline
    assert rncm.rncm_event_clock_hash(
        schedule, selected_quantile=0.985
    ) != baseline


def test_fixed_book_moving_band_control_has_zero_false_events() -> None:
    result = rncm.synthetic_control(rncm.Config())
    assert result["passes"] is True
    assert all(
        trial["nonoverlap_events"] == 0
        for scenario in result["scenarios"].values()
        for trial in scenario.values()
    )


def test_protocol_and_config_keep_outcomes_sealed() -> None:
    cfg = rncm.Config()
    payload = rncm.protocol(cfg)
    assert payload["outcomes_opened"] is False
    assert payload["external_ohlc_funding_return_or_equity_loaded"] is False
    assert payload["selection_end_exclusive"] == "2024-01-01 00:00:00"
    assert payload["side"] == "positive residual migration long; negative short"
    assert rncm.SKEW_COLUMNS == (
        "skew_2_median",
        "skew_3_median",
        "skew_4_median",
        "skew_5_median",
    )
    with pytest.raises(ValueError, match="configuration is frozen"):
        rncm._validate_config(replace(cfg, hold_bars=73))
