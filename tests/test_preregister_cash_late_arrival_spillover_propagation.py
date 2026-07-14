from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import preregister_cash_late_arrival_spillover_propagation as clasp


def _frame(rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "quarantined": False,
            "spot_quote_notional": 200.0,
            "um_quote_notional": 100.0,
            "spot_trade_count": 10.0,
            "um_trade_count": 10.0,
            "spot_flow_fraction": 0.5,
            "um_flow_fraction": 0.1,
            "spot_flow_coherence": 0.8,
            "spot_log_return_5m": 0.001,
            "um_log_return_5m": 0.0005,
            "spot_abs_path_return_bp": 20.0,
            "um_abs_path_return_bp": 20.0,
            "spot_activity_time_centroid": 0.8,
            "um_activity_time_centroid": 0.2,
            "spot_flow_time_centroid": 0.9,
            "um_flow_time_centroid": 0.3,
            "spot_return_time_centroid": 0.85,
            "um_return_time_centroid": 0.25,
            "basis_change_bp": -1.0,
            "spot_to_um_lagged_flow_response_bp": 2.0,
            "um_to_spot_lagged_flow_response_bp": -1.0,
            "lagged_flow_response_diff_bp": 1.0,
            "reverse_spot_to_um_lagged_flow_response_bp": 0.0,
            "reverse_um_to_spot_lagged_flow_response_bp": 0.0,
            "reverse_lagged_flow_response_diff_bp": 0.0,
        }
    )


def _zero_threshold(
    score: pd.Series,
    eligible: pd.Series,
    *,
    quantile: float,
    cfg: clasp.Config,
) -> pd.Series:
    del eligible, quantile, cfg
    return pd.Series(0.0, index=score.index)


def test_ticket_median_is_strictly_prior() -> None:
    values = pd.Series([1.0, 100.0, 3.0, 4.0])
    clean = pd.Series(True, index=values.index)
    cfg = clasp.Config(ticket_baseline_bars=2, ticket_min_periods=1)
    result = clasp.strictly_prior_ticket_median(values, clean, cfg)
    assert result.iloc[1] == 1.0
    assert result.iloc[2] == 50.5


def test_event_quantile_uses_prior_eligible_events_only() -> None:
    score = pd.Series([1.0, 100.0, 3.0, 4.0])
    eligible = pd.Series([True, False, True, True])
    cfg = clasp.Config(event_baseline_bars=3, event_min_periods=1)
    result = clasp.prior_event_quantile(score, eligible, quantile=0.5, cfg=cfg)
    assert result.iloc[1] == 1.0
    assert result.iloc[2] == 1.0
    assert result.iloc[3] == 2.0


def test_primary_requires_late_cash_and_forward_over_reverse_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    monkeypatch.setattr(
        clasp,
        "strictly_prior_ticket_median",
        lambda ticket, clean, cfg: pd.Series(0.0, index=ticket.index),
    )
    monkeypatch.setattr(clasp, "prior_event_quantile", _zero_threshold)
    signal, controls, sides, diagnostics = clasp.classify_events(
        frame, clasp.Config(), quantile=0.75
    )
    assert controls["primary"].all()
    assert signal["side"].eq(1).all()
    assert sides["direction_flip"].eq(-1).all()
    assert diagnostics["base_primary"].all()

    reverse = frame.copy()
    reverse["reverse_spot_to_um_lagged_flow_response_bp"] = 3.0
    reverse["reverse_lagged_flow_response_diff_bp"] = 3.0
    _, reverse_controls, _, reverse_diagnostics = clasp.classify_events(
        reverse, clasp.Config(), quantile=0.75
    )
    assert not reverse_controls["primary"].any()
    assert not reverse_diagnostics["base_primary"].any()


def test_early_cash_is_a_mutually_exclusive_temporal_placebo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    monkeypatch.setattr(
        clasp,
        "strictly_prior_ticket_median",
        lambda ticket, clean, cfg: pd.Series(0.0, index=ticket.index),
    )
    monkeypatch.setattr(clasp, "prior_event_quantile", _zero_threshold)
    _, controls, _, _ = clasp.classify_events(frame, clasp.Config(), quantile=0.75)
    assert controls["primary"].all()
    assert not controls["early_cash"].any()

    early = frame.copy()
    for suffix in (
        "activity_time_centroid",
        "flow_time_centroid",
        "return_time_centroid",
    ):
        early[f"spot_{suffix}"] = 0.2
        early[f"um_{suffix}"] = 0.8
    _, early_controls, _, _ = clasp.classify_events(
        early, clasp.Config(), quantile=0.75
    )
    assert not early_controls["primary"].any()
    assert early_controls["early_cash"].all()


def test_missing_timing_or_response_component_cannot_form_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    frame.loc[3, "spot_flow_time_centroid"] = np.nan
    frame.loc[4, "lagged_flow_response_diff_bp"] = np.nan
    monkeypatch.setattr(
        clasp,
        "strictly_prior_ticket_median",
        lambda ticket, clean, cfg: pd.Series(0.0, index=ticket.index),
    )
    monkeypatch.setattr(clasp, "prior_event_quantile", _zero_threshold)
    _, controls, _, _ = clasp.classify_events(frame, clasp.Config(), quantile=0.75)
    assert not controls["primary"].iloc[3]
    assert not controls["primary"].iloc[4]


def test_venue_swap_mirrors_direction_basis_ticket_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    frame["spot_quote_notional"] = 100.0
    frame["um_quote_notional"] = 200.0
    frame["spot_flow_fraction"] = 0.1
    frame["um_flow_fraction"] = -0.5
    frame["spot_log_return_5m"] = -0.0005
    frame["um_log_return_5m"] = -0.001
    frame["spot_abs_path_return_bp"] = 20.0
    frame["um_abs_path_return_bp"] = 10.0
    frame["basis_change_bp"] = -1.0
    frame["spot_activity_time_centroid"] = 0.2
    frame["um_activity_time_centroid"] = 0.8
    frame["spot_flow_time_centroid"] = 0.2
    frame["um_flow_time_centroid"] = 0.8
    frame["spot_return_time_centroid"] = 0.2
    frame["um_return_time_centroid"] = 0.8
    frame["um_to_spot_lagged_flow_response_bp"] = 2.0
    frame["lagged_flow_response_diff_bp"] = -1.0
    frame["reverse_um_to_spot_lagged_flow_response_bp"] = 0.0
    frame["reverse_lagged_flow_response_diff_bp"] = 0.0
    monkeypatch.setattr(
        clasp,
        "strictly_prior_ticket_median",
        lambda ticket, clean, cfg: pd.Series(0.0, index=ticket.index),
    )
    monkeypatch.setattr(clasp, "prior_event_quantile", _zero_threshold)
    _, controls, sides, _ = clasp.classify_events(frame, clasp.Config(), quantile=0.75)
    assert controls["venue_swap"].all()
    assert sides["venue_swap"].eq(-1).all()
    assert not controls["primary"].any()


def test_nonoverlapping_schedule_is_next_open_fixed_24_bars() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=80, freq="5min"),
            "quarantined": False,
        }
    )
    active = pd.Series(False, index=frame.index)
    active.iloc[[1, 20, 26, 52]] = True
    signal = pd.DataFrame(
        {
            "side": np.where(active, 1, 0),
            "branch": np.where(active, "clasp24", "none"),
            "hold_bars": np.where(active, 24, 0),
        }
    )
    schedule = clasp.nonoverlapping_schedule(signal, frame)
    assert schedule["signal_position"].tolist() == [1, 26, 52]
    assert schedule["entry_position"].tolist() == [2, 27, 53]
    assert schedule["exit_position"].tolist() == [26, 51, 77]


def test_support_floor_is_fail_closed() -> None:
    schedule = pd.DataFrame(
        [
            {"entry_date": "2023-01-01 00:05:00", "side": 1},
            {"entry_date": "2023-07-01 00:05:00", "side": -1},
        ]
    )
    permissive = clasp.Config(
        minimum_nonoverlap_total=0,
        minimum_nonoverlap_per_year=0,
        minimum_nonoverlap_per_2023_half=0,
        minimum_nonoverlap_per_2023_quarter=0,
        minimum_side_share=0.0,
        minimum_side_events_per_year=0,
        minimum_active_months=0,
    )
    assert clasp._support(schedule, permissive)["passes_count_support"]
    strict = clasp.Config(
        minimum_nonoverlap_total=3,
        minimum_nonoverlap_per_year=0,
        minimum_nonoverlap_per_2023_half=0,
        minimum_nonoverlap_per_2023_quarter=0,
        minimum_side_share=0.0,
        minimum_side_events_per_year=0,
        minimum_active_months=0,
    )
    assert not clasp._support(schedule, strict)["passes_count_support"]


def test_load_causal_frame_rejects_execution_ohlc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    frame["open"] = 100.0
    monkeypatch.setattr(clasp.cross, "load_causal_frame", lambda _cfg: (frame, {}))
    with pytest.raises(ValueError, match="outcome columns"):
        clasp.load_causal_frame()


def test_control_contract_covers_every_non_primary_policy() -> None:
    assert set(clasp.SCORE_BEARING_CONTROLS) == set(clasp.CONTROL_SIDE_RULES) - {
        "primary",
        "direction_flip",
        "signal_delay_1bar",
    }
