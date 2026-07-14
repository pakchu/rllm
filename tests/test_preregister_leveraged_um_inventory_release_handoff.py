from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import preregister_leveraged_um_inventory_release_handoff as luri


def _frame(rows: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "quarantined": False,
            "um_signed_quote_notional": 10.0,
            "um_quote_notional": 100.0,
            "spot_signed_quote_notional": -10.0,
            "spot_quote_notional": 100.0,
            "um_log_return_5m": 0.01,
            "spot_log_return_5m": -0.01,
            "open_basis_bp": np.arange(rows, dtype=float),
            "close_basis_bp": np.arange(rows, dtype=float) + 2.0,
            "basis_change_bp": 1.0,
            "um_to_spot_lagged_directional_alignment": 0.8,
            "lagged_directional_alignment_diff": -0.4,
            "reverse_um_to_spot_lagged_directional_alignment": 0.1,
            "reverse_lagged_directional_alignment_diff": -0.1,
            "spot_to_um_lagged_directional_alignment": 0.8,
            "reverse_spot_to_um_lagged_directional_alignment": 0.1,
            "um_minus_spot_activity_time_centroid": -0.5,
            "um_flow_fraction": 0.5,
            "spot_flow_fraction": -0.5,
            "simultaneous_flow_sign_agreement": 1.0,
            "simultaneous_return_sign_agreement": 1.0,
        }
    )
    return frame


def test_trailing_flow_fraction_ends_at_previous_bar() -> None:
    frame = _frame(4)
    frame["um_signed_quote_notional"] = [10.0, 20.0, 9_999.0, 0.0]
    result = luri._trailing_flow_fraction(frame, "um", 2)
    assert np.isclose(result.iloc[2], 30.0 / 200.0)
    assert not np.isclose(result.iloc[2], (20.0 + 9_999.0) / 200.0)


def test_history_clean_rejects_any_prior_gap_but_not_future_gap() -> None:
    clean = pd.Series([True, False, True, True, False])
    result = luri._history_clean(clean, 2)
    assert not result.iloc[2]
    assert not result.iloc[3]
    assert result.iloc[4]


def test_prior_quantile_is_strictly_shifted() -> None:
    values = pd.Series([1.0, 100.0, 3.0, 4.0])
    clean = pd.Series(True, index=values.index)
    cfg = luri.Config(baseline_bars=2, baseline_min_periods=1)
    threshold = luri.prior_quantile(values, clean, quantile=0.5, cfg=cfg)
    assert threshold.iloc[1] == 1.0
    assert threshold.iloc[2] == 50.5


def test_primary_uses_inventory_opposite_side_and_forward_time_dominance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    trigger = 6
    frame.loc[trigger, "um_flow_fraction"] = -0.5
    frame.loc[trigger, "um_log_return_5m"] = -0.01
    frame.loc[trigger, "basis_change_bp"] = -1.0
    monkeypatch.setattr(
        luri,
        "prior_quantile",
        lambda values, clean, *, quantile, cfg: pd.Series(0.0, index=values.index),
    )
    cfg = luri.Config(formation_bars=2, baseline_bars=2, baseline_min_periods=1)
    signal, controls, sides, diagnostics = luri.classify_events(
        frame, cfg, basis_quantile=0.65
    )
    assert controls["primary"].iloc[trigger]
    assert signal["side"].iloc[trigger] == -1
    assert sides["direction_flip"].iloc[trigger] == 1
    assert (
        diagnostics["forward_handoff"].iloc[trigger]
        > diagnostics["reverse_handoff"].iloc[trigger]
    )

    reversed_frame = frame.copy()
    reversed_frame.loc[trigger, "reverse_um_to_spot_lagged_directional_alignment"] = 1.0
    reversed_frame.loc[trigger, "reverse_lagged_directional_alignment_diff"] = -1.0
    _, reversed_controls, _, _ = luri.classify_events(
        reversed_frame, cfg, basis_quantile=0.65
    )
    assert not reversed_controls["primary"].iloc[trigger]
    assert reversed_controls["reverse_time"].iloc[trigger]


def test_basis_percentile_history_excludes_nonpositive_displacements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    frame["open_basis_bp"] = 0.0
    frame["close_basis_bp"] = [2.0, -2.0] * 5
    captured: list[pd.Series] = []

    def capture(
        values: pd.Series,
        clean: pd.Series,
        *,
        quantile: float,
        cfg: luri.Config,
    ) -> pd.Series:
        captured.append(values.copy())
        return pd.Series(0.0, index=values.index)

    monkeypatch.setattr(luri, "prior_quantile", capture)
    cfg = luri.Config(formation_bars=2, baseline_bars=2, baseline_min_periods=1)
    luri.classify_events(frame, cfg, basis_quantile=0.40)
    assert len(captured) == 2
    assert all(series.isna().any() for series in captured)
    assert all((series.dropna() > 0.0).all() for series in captured)


def test_spot_inventory_swap_mirrors_basis_and_release_signs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    trigger = 6
    frame["spot_signed_quote_notional"] = 10.0
    frame["um_signed_quote_notional"] = -10.0
    frame["um_log_return_5m"] = -0.01
    frame["open_basis_bp"] = 0.0
    frame["close_basis_bp"] = -2.0
    frame.loc[trigger, "spot_flow_fraction"] = -0.5
    frame.loc[trigger, "spot_log_return_5m"] = -0.01
    frame.loc[trigger, "basis_change_bp"] = 1.0
    frame.loc[trigger, "um_minus_spot_activity_time_centroid"] = 0.5
    frame.loc[trigger, "lagged_directional_alignment_diff"] = 0.4
    frame.loc[trigger, "reverse_spot_to_um_lagged_directional_alignment"] = 0.1
    monkeypatch.setattr(
        luri,
        "prior_quantile",
        lambda values, clean, *, quantile, cfg: pd.Series(0.0, index=values.index),
    )
    cfg = luri.Config(formation_bars=2, baseline_bars=2, baseline_min_periods=1)
    _, controls, sides, _ = luri.classify_events(frame, cfg, basis_quantile=0.40)
    assert controls["spot_inventory_swap"].iloc[trigger]
    assert sides["spot_inventory_swap"].iloc[trigger] == -1


def test_catch_venue_swap_reference_reserves_its_own_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=40, freq="5min"),
            "quarantined": False,
        }
    )
    mask = pd.Series(False, index=frame.index)
    mask.iloc[1] = True
    side = pd.Series(1.0, index=frame.index)
    monkeypatch.setattr(
        luri.cross,
        "classify_events",
        lambda *_args, **_kwargs: (
            pd.DataFrame(),
            {"venue_swap": mask},
            {"venue_swap": side},
            {},
        ),
    )
    observed_mask, schedule = luri._catch_venue_swap_reference(frame)
    assert observed_mask.equals(mask)
    assert schedule["signal_position"].tolist() == [1]
    assert schedule["entry_position"].tolist() == [2]
    assert schedule["exit_position"].tolist() == [14]
    assert schedule["branch"].tolist() == ["control_venue_swap"]


def test_nonoverlapping_schedule_is_next_open_fixed_48_bars() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=120, freq="5min"),
            "quarantined": False,
        }
    )
    active = pd.Series(False, index=frame.index)
    active.iloc[[1, 20, 50]] = True
    signal = pd.DataFrame(
        {
            "side": np.where(active, 1, 0),
            "branch": np.where(active, "luri48", "none"),
            "hold_bars": np.where(active, 48, 0),
        }
    )
    schedule = luri.nonoverlapping_schedule(signal, frame)
    assert schedule["signal_position"].tolist() == [1, 50]
    assert schedule["entry_position"].tolist() == [2, 51]
    assert schedule["exit_position"].tolist() == [50, 99]


def test_support_floor_is_fail_closed() -> None:
    schedule = pd.DataFrame(
        [
            {
                "entry_date": "2023-01-01 00:05:00",
                "side": 1,
            },
            {
                "entry_date": "2023-07-01 00:05:00",
                "side": -1,
            },
        ]
    )
    permissive = luri.Config(
        minimum_nonoverlap_total=0,
        minimum_nonoverlap_per_year=0,
        minimum_nonoverlap_per_2023_half=0,
        minimum_nonoverlap_per_2023_quarter=0,
        minimum_side_share=0.0,
        minimum_side_events_per_year=0,
        minimum_active_months=0,
    )
    assert luri._support(schedule, permissive)["passes_count_support"]
    strict = luri.Config(
        minimum_nonoverlap_total=3,
        minimum_nonoverlap_per_year=0,
        minimum_nonoverlap_per_2023_half=0,
        minimum_nonoverlap_per_2023_quarter=0,
        minimum_side_share=0.0,
        minimum_side_events_per_year=0,
        minimum_active_months=0,
    )
    assert not luri._support(schedule, strict)["passes_count_support"]


def test_overlap_and_retention_are_directional() -> None:
    primary = pd.Series([True, True, False, False])
    control = pd.Series([True, False, True, False])
    overlap = luri._overlap(primary, control)
    assert overlap == {
        "intersection": 1,
        "union": 3,
        "jaccard": 1 / 3,
        "primary_containment": 0.5,
    }
    assert luri._retention(2, 4) == 0.5


def test_simultaneous_overlap_uses_distinct_raw_and_scheduled_limits() -> None:
    raw = {"simultaneous_only": {"jaccard": 0.14, "primary_containment": 0.79}}
    scheduled = {"simultaneous_only": {"jaccard": 0.14, "primary_containment": 0.59}}
    assert luri._both_overlap_pass(
        raw,
        scheduled,
        "simultaneous_only",
        jaccard=0.15,
        raw_containment=0.80,
        scheduled_containment=0.60,
    )
    scheduled["simultaneous_only"]["primary_containment"] = 0.61
    assert not luri._both_overlap_pass(
        raw,
        scheduled,
        "simultaneous_only",
        jaccard=0.15,
        raw_containment=0.80,
        scheduled_containment=0.60,
    )


def test_load_causal_frame_rejects_execution_ohlc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    frame["open"] = 100.0
    monkeypatch.setattr(
        luri.cross,
        "load_causal_frame",
        lambda _cfg: (frame, {}),
    )
    with pytest.raises(ValueError, match="outcome columns"):
        luri.load_causal_frame()


def test_control_sets_cover_every_non_primary_policy() -> None:
    assert set(luri.SCORE_BEARING_CONTROLS) == set(luri.CONTROL_SIDE_RULES) - {
        "primary",
        "direction_flip",
        "signal_delay_1bar",
    }
