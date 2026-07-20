from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import preregister_btcdom_leverage_polarity_decomposition as dlpd


def _frame(rows: int = 680) -> pd.DataFrame:
    dates = pd.date_range("2021-12-01", periods=rows, freq="1h")
    btc = np.linspace(-1.0, 1.0, rows)
    dom = np.linspace(1.0, -1.0, rows)
    return pd.DataFrame(
        {
            "date": dates,
            "source_close_time": dates + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
            "feature_available_time": dates + pd.Timedelta(hours=1, seconds=1),
            "btcusdt_valid": True,
            "btcdomusdt_valid": True,
            "source_valid": True,
            "btcusdt_premium_close": btc,
            "btcdomusdt_premium_close": dom,
        }
    )


def test_prior_robust_z_never_uses_current_observation() -> None:
    cfg = replace(dlpd.FROZEN_CONFIG, lookback_hours=4, minimum_history_hours=4)
    values = pd.Series([0.0, 1.0, 2.0, 3.0, 100.0, -100.0])
    valid = pd.Series(True, index=values.index)
    first = dlpd.prior_robust_z(values.iloc[:5], valid.iloc[:5], cfg)
    full = dlpd.prior_robust_z(values, valid, cfg)
    assert full.iloc[:5].equals(first)
    assert np.isnan(full.iloc[3])
    assert np.isfinite(full.iloc[4])


def test_signal_mapping_and_stale_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame(4)
    btc = pd.Series([2.0, -2.0, 2.0, 0.0])
    dom = pd.Series([-2.0, 2.0, 2.0, -2.0])
    calls = iter([btc, dom])
    monkeypatch.setattr(dlpd, "prior_robust_z", lambda *_args, **_kwargs: next(calls))
    states = dlpd.signal_states(frame, dlpd.FROZEN_CONFIG)
    assert states["primary_active"].tolist() == [True, True, False, False]
    assert states["primary_side"].tolist() == [1, -1, 0, 0]
    assert states["same_sign_active"].tolist() == [False, False, True, False]
    assert states["btc_only_tail" + "_active"].tolist() == [True, True, True, False]
    assert states["dom_only_mirror_side"].tolist() == [1, -1, -1, 1]
    assert states["stale_btc_1h_active"].tolist() == [False, False, True, True]
    assert states["stale_dom_1h_active"].tolist() == [False, False, False, False]


def test_schedule_uses_onset_next_open_nonoverlap_and_containment() -> None:
    frame = _frame(7)
    frame["date"] = pd.to_datetime(
        [
            "2021-12-31 23:00",
            "2022-01-01 00:00",
            "2022-01-01 01:00",
            "2022-01-01 13:00",
            "2022-01-01 14:00",
            "2022-12-31 12:00",
            "2022-12-31 13:00",
        ]
    )
    frame["feature_available_time"] = frame["date"] + pd.Timedelta(hours=1, seconds=1)
    states = pd.DataFrame(index=frame.index)
    for control in dlpd.CONTROLS:
        states[f"{control}_active"] = [True, False, True, False, True, False, True]
        states[f"{control}_side"] = [1, 0, -1, 0, -1, 0, 1]
        states[f"{control}_btc_z"] = [2.0, 0.0, -2.0, 0.0, -2.0, 0.0, 2.0]
        states[f"{control}_dom_z"] = [-2.0, 0.0, 2.0, 0.0, 2.0, 0.0, -2.0]
    events = dlpd.schedule(frame, states, control="primary", year=2022)
    assert events["entry_time"].tolist() == [
        pd.Timestamp("2022-01-01 00:05"),
        pd.Timestamp("2022-01-01 15:05"),
    ]
    # The final event would cross the calendar boundary and is rejected.
    assert events["exit_time"].max() <= pd.Timestamp("2023-01-01")
    assert events.columns.tolist() == list(dlpd.EVENT_COLUMNS)


def test_preregistration_payload_keeps_all_outcomes_sealed() -> None:
    payload = dlpd.preregistration_payload()
    assert payload["candidate"] == "DLPD-12"
    assert payload["outcomes_opened"] is False
    assert payload["outcome_sources_opened"] is False
    assert payload["post_2023_source_rows_opened"] is False
    assert payload["real_event_incidence_opened"] is False
    assert payload["source_only_controls"] == list(dlpd.SOURCE_ONLY_CONTROLS)
    assert len(payload["support_comparators"]) == 5
    assert payload["manifest_hash"] == dlpd.canonical_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )


def test_frozen_config_rejects_repairs() -> None:
    with pytest.raises(ValueError, match="configuration is frozen"):
        dlpd.signal_states(_frame(), replace(dlpd.FROZEN_CONFIG, absolute_z_threshold=1.25))
