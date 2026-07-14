from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import preregister_um_forced_flow_reversion as umfr


def _frame(rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "quarantined": False,
            "spot_quote_notional": 100.0,
            "um_quote_notional": 200.0,
            "spot_trade_count": 10.0,
            "um_trade_count": 10.0,
            "spot_flow_fraction": 0.1,
            "um_flow_fraction": 0.5,
            "um_flow_coherence": 0.8,
            "spot_log_return_5m": 0.0005,
            "um_log_return_5m": 0.0010,
            "spot_abs_path_return_bp": 10.0,
            "um_abs_path_return_bp": 30.0,
            "spot_activity_time_centroid": 0.2,
            "um_activity_time_centroid": 0.8,
            "spot_flow_time_centroid": 0.3,
            "um_flow_time_centroid": 0.9,
            "spot_return_time_centroid": 0.25,
            "um_return_time_centroid": 0.85,
            "basis_change_bp": 1.0,
        }
    )


def _zero_threshold(
    score: pd.Series, eligible: pd.Series, *, quantile: float, cfg: umfr.Config
) -> pd.Series:
    del eligible, quantile, cfg
    return pd.Series(0.0, index=score.index)


def test_primary_fades_late_um_flow_and_direction_flip_follows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    monkeypatch.setattr(
        umfr,
        "strictly_prior_ticket_median",
        lambda ticket, clean, cfg: pd.Series(0.0, index=ticket.index),
    )
    monkeypatch.setattr(umfr, "prior_event_quantile", _zero_threshold)
    signal, controls, sides, diagnostics = umfr.classify_events(
        frame, umfr.Config(), quantile=0.8
    )
    assert controls["primary"].all()
    assert signal["side"].eq(-1).all()
    assert sides["direction_flip"].eq(1).all()
    assert diagnostics["base_primary"].all()


def test_primary_requires_underresponse_not_spot_full_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    frame["spot_log_return_5m"] = frame["um_log_return_5m"]
    monkeypatch.setattr(
        umfr,
        "strictly_prior_ticket_median",
        lambda ticket, clean, cfg: pd.Series(0.0, index=ticket.index),
    )
    monkeypatch.setattr(umfr, "prior_event_quantile", _zero_threshold)
    _, controls, _, diagnostics = umfr.classify_events(
        frame, umfr.Config(), quantile=0.8
    )
    assert not controls["primary"].any()
    assert not diagnostics["base_primary"].any()


def test_missing_timing_component_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame()
    frame.loc[3, "um_flow_time_centroid"] = np.nan
    monkeypatch.setattr(
        umfr,
        "strictly_prior_ticket_median",
        lambda ticket, clean, cfg: pd.Series(0.0, index=ticket.index),
    )
    monkeypatch.setattr(umfr, "prior_event_quantile", _zero_threshold)
    _, controls, _, _ = umfr.classify_events(frame, umfr.Config(), quantile=0.8)
    assert not controls["primary"].iloc[3]


def test_nonoverlapping_schedule_is_next_open_fixed_36_bars() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=120, freq="5min"),
            "quarantined": False,
        }
    )
    active = pd.Series(False, index=frame.index)
    active.iloc[[1, 20, 38, 76]] = True
    signal = pd.DataFrame(
        {
            "side": np.where(active, -1, 0),
            "branch": np.where(active, "umfr36", "none"),
            "hold_bars": np.where(active, 36, 0),
        }
    )
    schedule = umfr.nonoverlapping_schedule(signal, frame)
    assert schedule["signal_position"].tolist() == [1, 38, 76]
    assert schedule["entry_position"].tolist() == [2, 39, 77]
    assert schedule["exit_position"].tolist() == [38, 75, 113]


def test_load_causal_frame_rejects_execution_ohlc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    frame["close"] = 100.0
    monkeypatch.setattr(umfr.cross, "load_causal_frame", lambda _cfg: (frame, {}))
    with pytest.raises(ValueError, match="outcome columns"):
        umfr.load_causal_frame()


def test_control_contract_covers_every_non_primary_policy() -> None:
    assert set(umfr.SCORE_BEARING_CONTROLS) == set(umfr.CONTROL_SIDE_RULES) - {
        "primary",
        "direction_flip",
        "signal_delay_1bar",
    }
