from __future__ import annotations

import numpy as np
import pandas as pd

from training import preregister_cash_auction_transfer_catchup_handoff as catch
from training.build_binance_cross_venue_minute_leadership import OUTPUT_COLUMNS


_COMPONENT_COLUMNS = (
    "spot_flow_fraction",
    "um_flow_fraction",
    "spot_flow_coherence",
    "um_flow_coherence",
    "spot_log_return_5m",
    "um_log_return_5m",
    "basis_change_bp",
    "um_minus_spot_activity_time_centroid",
    "spot_to_um_lagged_directional_alignment",
    "um_to_spot_lagged_directional_alignment",
    "lagged_directional_alignment_diff",
    "reverse_spot_to_um_lagged_directional_alignment",
    "reverse_lagged_directional_alignment_diff",
    "simultaneous_flow_sign_agreement",
    "simultaneous_return_sign_agreement",
    "flow_transfer_asymmetry",
    "return_leadership_asymmetry",
)


def _catch_cfg() -> catch.Config:
    return catch.Config(baseline_bars=4, baseline_min_periods=2, hold_bars=12)


def _frame(rows: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=rows, freq="5min")
    frame = pd.DataFrame(index=range(rows))
    for column in OUTPUT_COLUMNS:
        frame[column] = 0
    frame["date"] = dates
    frame["feature_available_time_utc"] = dates + pd.Timedelta("5min")
    frame["trade_earliest_time_utc"] = frame["feature_available_time_utc"]
    frame["feature_invalid_reason"] = "ok"
    frame["source_complete"] = True
    frame["cross_venue_feature_valid"] = True
    frame["quarantined"] = False

    # Low but valid prior observations establish strictly-prior thresholds.
    for column in _COMPONENT_COLUMNS:
        frame[column] = 0.25
    frame["spot_flow_fraction"] = 0.4
    frame["um_flow_fraction"] = 0.3
    frame["spot_log_return_5m"] = 0.01
    frame["um_log_return_5m"] = 0.01
    frame["basis_change_bp"] = -0.25
    frame["um_minus_spot_activity_time_centroid"] = 0.25
    frame["spot_to_um_lagged_directional_alignment"] = 0.25
    frame["lagged_directional_alignment_diff"] = 0.25
    frame["reverse_spot_to_um_lagged_directional_alignment"] = 0.20
    frame["reverse_lagged_directional_alignment_diff"] = 0.20
    frame["um_to_spot_lagged_directional_alignment"] = 0.10
    frame["simultaneous_flow_sign_agreement"] = 0.20
    frame["simultaneous_return_sign_agreement"] = 0.20
    frame["flow_transfer_asymmetry"] = 0.20
    frame["return_leadership_asymmetry"] = 0.20

    # Final row is the high-score row used by most behavioral assertions.
    last = rows - 1
    frame.loc[last, "spot_flow_fraction"] = 0.9
    frame.loc[last, "um_flow_fraction"] = 0.7
    frame.loc[last, "spot_flow_coherence"] = 0.9
    frame.loc[last, "um_flow_coherence"] = 0.8
    frame.loc[last, "spot_log_return_5m"] = 0.02
    frame.loc[last, "um_log_return_5m"] = 0.02
    frame.loc[last, "basis_change_bp"] = -1.0
    frame.loc[last, "um_minus_spot_activity_time_centroid"] = 0.8
    frame.loc[last, "spot_to_um_lagged_directional_alignment"] = 0.8
    frame.loc[last, "lagged_directional_alignment_diff"] = 0.8
    frame.loc[last, "reverse_spot_to_um_lagged_directional_alignment"] = 0.7
    frame.loc[last, "reverse_lagged_directional_alignment_diff"] = 0.7
    frame.loc[last, "um_to_spot_lagged_directional_alignment"] = 0.05
    frame.loc[last, "simultaneous_flow_sign_agreement"] = 0.8
    frame.loc[last, "simultaneous_return_sign_agreement"] = 0.8
    frame.loc[last, "flow_transfer_asymmetry"] = 0.8
    frame.loc[last, "return_leadership_asymmetry"] = 0.8
    return frame


def test_prior_quantile_is_strictly_lagged() -> None:
    values = pd.Series([1.0, 2.0, 100.0])
    clean = pd.Series(True, index=values.index)
    cfg = catch.Config(baseline_bars=2, baseline_min_periods=2)

    result = catch.prior_quantile(values, clean, quantile=0.5, cfg=cfg)

    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 1.5


def test_primary_requires_handoff_spot_earlier_accepted_spot_flow_and_residual_basis() -> None:
    cfg = _catch_cfg()
    frame = _frame()
    _, controls, _, diagnostics = catch.classify_events(frame, cfg, quantile=0.5)
    assert controls["primary"].iloc[-1]

    mutations = {
        "positive directed handoff": {
            "spot_to_um_lagged_directional_alignment": 0.0,
        },
        "Spot-earlier activity": {"um_minus_spot_activity_time_centroid": -0.1},
        "accepted Spot flow": {"spot_log_return_5m": -0.02},
        "residual basis": {"basis_change_bp": 1.0},
    }
    for expected_requirement, updates in mutations.items():
        changed = frame.copy()
        for column, value in updates.items():
            changed.loc[changed.index[-1], column] = value
        _, changed_controls, _, changed_diagnostics = catch.classify_events(
            changed, cfg, quantile=0.5
        )
        assert not changed_controls["primary"].iloc[-1], expected_requirement
        assert not changed_diagnostics["base_primary"].iloc[-1], expected_requirement


def test_primary_supports_long_short_symmetry() -> None:
    cfg = _catch_cfg()
    frame = _frame()
    mirrored = frame.copy()
    mirrored["spot_flow_fraction"] *= -1
    mirrored["spot_log_return_5m"] *= -1
    mirrored["basis_change_bp"] *= -1

    signal, controls, _, _ = catch.classify_events(mirrored, cfg, quantile=0.5)

    assert controls["primary"].iloc[-1]
    assert signal["side"].iloc[-1] == -1


def test_weakening_any_primary_score_block_lowers_score() -> None:
    cfg = _catch_cfg()
    frame = _frame()
    _, _, _, diagnostics = catch.classify_events(frame, cfg, quantile=0.5)
    original = diagnostics["score_primary"].iloc[-1]

    for column in (
        "spot_to_um_lagged_directional_alignment",
        "um_minus_spot_activity_time_centroid",
        "spot_flow_coherence",
    ):
        changed = frame.copy()
        changed.loc[changed.index[-1], column] = 0.1
        _, _, _, changed_diagnostics = catch.classify_events(changed, cfg, quantile=0.5)
        assert changed_diagnostics["score_primary"].iloc[-1] < original, column


def test_reverse_time_uses_reverse_fields_and_aggregate_only_ignores_lagged_fields() -> None:
    cfg = _catch_cfg()
    frame = _frame()
    _, controls, _, diagnostics = catch.classify_events(frame, cfg, quantile=0.5)
    assert controls["reverse_time"].iloc[-1]
    aggregate_score = diagnostics["score_aggregate_only"].copy()

    changed = frame.copy()
    changed.loc[changed.index[-1], "spot_to_um_lagged_directional_alignment"] = 0.0
    changed.loc[changed.index[-1], "lagged_directional_alignment_diff"] = 0.0
    changed.loc[changed.index[-1], "um_to_spot_lagged_directional_alignment"] = 0.0
    _, changed_controls, _, changed_diagnostics = catch.classify_events(
        changed, cfg, quantile=0.5
    )
    assert changed_controls["reverse_time"].iloc[-1]
    pd.testing.assert_series_equal(
        aggregate_score,
        changed_diagnostics["score_aggregate_only"],
        check_names=False,
    )

    changed_reverse = frame.copy()
    changed_reverse.loc[
        changed_reverse.index[-1], "reverse_spot_to_um_lagged_directional_alignment"
    ] = 0.0
    _, changed_reverse_controls, _, _ = catch.classify_events(
        changed_reverse, cfg, quantile=0.5
    )
    assert not changed_reverse_controls["reverse_time"].iloc[-1]


def test_direction_flip_side_is_negative_primary_and_venue_swap_uses_um_flow() -> None:
    cfg = _catch_cfg()
    frame = _frame()
    last = frame.index[-1]
    frame.loc[last, "um_flow_fraction"] = -0.7
    frame.loc[last, "um_log_return_5m"] = -0.02
    frame.loc[last, "basis_change_bp"] = -1.0
    frame.loc[last, "um_minus_spot_activity_time_centroid"] = -0.8
    frame.loc[last, "um_to_spot_lagged_directional_alignment"] = 0.8
    frame.loc[last, "lagged_directional_alignment_diff"] = -0.8

    _, controls, control_sides, _ = catch.classify_events(frame, cfg, quantile=0.5)

    assert control_sides["direction_flip"].iloc[-1] == -control_sides["primary"].iloc[-1]
    assert controls["venue_swap"].iloc[-1]
    assert control_sides["venue_swap"].iloc[-1] == -1


def test_invalid_or_quarantined_signal_fails_closed() -> None:
    cfg = _catch_cfg()
    frame = _frame()
    quarantined = frame.copy()
    quarantined.loc[quarantined.index[-1], "quarantined"] = True
    signal, controls, _, _ = catch.classify_events(quarantined, cfg, quantile=0.5)
    assert not controls["primary"].iloc[-1]
    assert signal["side"].iloc[-1] == 0

    invalid = frame.copy()
    invalid.loc[invalid.index[-1], "spot_flow_fraction"] = 0.0
    signal, controls, _, _ = catch.classify_events(invalid, cfg, quantile=0.5)
    assert not controls["primary"].iloc[-1]
    assert signal["side"].iloc[-1] == 0


def test_nonoverlapping_scheduler_enters_next_row_exits_12_bars_later_and_ignores_future_quarantine() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=20, freq="5min"),
            "quarantined": [False, False] + [True] * 18,
        }
    )
    signal = pd.DataFrame(
        {
            "side": [1, 1] + [0] * 18,
            "branch": ["catch12", "catch12"] + ["none"] * 18,
            "hold_bars": [12, 12] + [0] * 18,
        }
    )

    schedule = catch.nonoverlapping_schedule(signal, frame)

    assert len(schedule) == 1
    assert schedule.loc[0, "signal_position"] == 0
    assert schedule.loc[0, "entry_position"] == 1
    assert schedule.loc[0, "exit_position"] == 13


def _support_schedule() -> pd.DataFrame:
    dates = [
        "2020-01-01",
        "2020-01-02",
        "2021-01-01",
        "2021-01-02",
        "2022-01-01",
        "2022-01-02",
        "2023-01-01",
        "2023-01-02",
        "2023-04-01",
        "2023-04-02",
        "2023-07-01",
        "2023-07-02",
        "2023-10-01",
        "2023-10-02",
    ]
    return pd.DataFrame(
        {
            "entry_date": dates,
            "side": [1, -1] * (len(dates) // 2),
        }
    )


def _support_cfg() -> catch.Config:
    return catch.Config(
        minimum_nonoverlap_total=14,
        minimum_nonoverlap_per_year=2,
        minimum_nonoverlap_per_2023_half=4,
        minimum_nonoverlap_per_2023_quarter=2,
        minimum_side_share=0.35,
        minimum_side_events_per_year=1,
        minimum_active_months=4,
        minimum_events_per_active_month=2,
    )


def test_support_gate_enforces_year_half_quarter_month_and_side_floors() -> None:
    cfg = _support_cfg()
    schedule = _support_schedule()
    assert catch._support(schedule, cfg)["passes_count_support"]
    empty = pd.DataFrame(columns=catch.SCHEDULE_COLUMNS)
    assert not catch._support(empty, cfg)["passes_count_support"]

    missing_year = schedule.drop(index=2).reset_index(drop=True)
    assert not catch._support(missing_year, cfg)["passes_count_support"]

    missing_half = schedule[~schedule["entry_date"].str.startswith("2023-07")].reset_index(
        drop=True
    )
    assert not catch._support(missing_half, cfg)["passes_count_support"]

    missing_quarter = schedule[~schedule["entry_date"].str.startswith("2023-10")].reset_index(
        drop=True
    )
    assert not catch._support(missing_quarter, cfg)["passes_count_support"]

    missing_month = schedule.drop(index=[6, 7]).reset_index(drop=True)
    assert not catch._support(missing_month, cfg)["passes_count_support"]

    one_sided = schedule.copy()
    one_sided["side"] = 1
    assert not catch._support(one_sided, cfg)["passes_count_support"]


def test_overlap_returns_jaccard_and_primary_containment() -> None:
    result = catch._overlap(
        pd.Series([True, True, False, False]),
        pd.Series([True, False, True, False]),
    )

    assert result["intersection"] == 1
    assert result["union"] == 3
    assert result["jaccard"] == 1.0 / 3.0
    assert result["primary_containment"] == 0.5
