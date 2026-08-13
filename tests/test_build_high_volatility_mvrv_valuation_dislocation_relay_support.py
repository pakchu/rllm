from __future__ import annotations

import urllib.parse

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_mvrv_valuation_dislocation_relay_support as s


def raw_row(day: str, completion: str, mvrv: str = "2.5") -> dict[str, str]:
    return {
        "asset": "btc",
        "time": f"{day}T00:00:00.000000000Z",
        "CapMVRVCur": mvrv,
        "AssetEODCompletionTime": completion,
    }


def test_exact_source_and_btc_contract():
    query = urllib.parse.parse_qs(urllib.parse.urlparse(s.source_url()).query)
    assert query["metrics"] == ["CapMVRVCur,AssetEODCompletionTime"]
    normalized = " ".join(s.BTC_QUERY.split()).lower()
    assert normalized.startswith("select ts,open,close from bars_binance")
    assert not any(token in normalized for token in ("high,", "low,", "volume", "entry_price"))


def test_source_row_schema_value_and_completion_window():
    parsed = s.parse_source_row(raw_row("2022-01-01", "1641081601.5"))
    assert parsed["CapMVRVCur"] == "2.5"
    assert parsed["feature_available_time"] == pd.Timestamp("2022-01-02T00:00:01.500000Z")
    with pytest.raises(ValueError, match=r"after D\+1"):
        s.parse_source_row(raw_row("2022-01-01", "1641081600"))
    with pytest.raises(ValueError, match="positive and finite"):
        s.parse_source_row(raw_row("2022-01-01", "1641081601", "0"))


def test_panel_uses_strict_prior_log_mvrv_and_open_close_variation(monkeypatch):
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = start + pd.Timedelta(days=32)
    monkeypatch.setattr(s, "SOURCE_START", start)
    monkeypatch.setattr(s, "SOURCE_END", end)
    observations = pd.date_range(start, end, freq="1d", inclusive="left")
    values = np.exp(np.arange(32, dtype=float) / 100.0)
    source = pd.DataFrame({
        "observation_time": observations,
        "feature_available_time": observations + pd.Timedelta(days=1, seconds=1),
        "CapMVRVCur": values,
    })
    minutes = pd.date_range(
        start + pd.Timedelta(hours=12), end + pd.Timedelta(hours=12),
        freq="1min", inclusive="left",
    )
    bars = pd.DataFrame({"ts": minutes, "open": 100.0, "close": 100.0 * np.exp(0.001)})
    panel = s.build_panel(source, bars)
    assert np.isnan(panel.mvrv_local_z.iloc[29])
    prior = np.arange(30, dtype=float) / 100.0
    expected = (0.30 - prior.mean()) / prior.std(ddof=1)
    assert panel.mvrv_local_z.iloc[30] == pytest.approx(expected)
    assert panel.minute_count.iloc[30] == 1440
    assert panel.btc_variation.iloc[30] == pytest.approx(np.sqrt(1440 * 0.001**2))
    assert bool(panel.source_valid.iloc[30])


def clock_panel() -> pd.DataFrame:
    observations = pd.date_range("2024-01-01T00:00:00Z", periods=4, freq="1d")
    return pd.DataFrame({
        "observation_time": observations,
        "feature_available_time": observations + pd.Timedelta(days=1, hours=1),
        "decision_time": observations + pd.Timedelta(days=1, hours=12),
        "source_valid": True,
        "minute_count": 1440,
        "CapMVRVCur": 2.0,
        "mvrv_log_prior_mean_30d": 0.5,
        "mvrv_log_prior_std_30d": 0.1,
        "mvrv_local_z": [-0.5, 0.2, 0.5, -1.0],
        "btc_decision_close": 100.0,
        "market_proxy_log_prior_mean_30d": 4.0,
        "market_proxy_log_prior_std_30d": 0.2,
        "market_proxy_local_z": [0.1, -0.6, 0.2, 0.8],
        "btc_variation": 1.0,
        "btc_variation_rank": [0.65, 0.9, 0.9, 0.9],
    })


def test_clock_direction_timing_reservation_and_controls():
    panel = clock_panel()
    clock = s.build_clock(panel)
    assert list(clock.side) == [1, -1, 1]
    assert (clock.entry_time - clock.decision_time).eq(pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time - clock.entry_time).eq(pd.Timedelta(hours=24)).all()
    assert clock.entry_time.iloc[2] == clock.exit_time.iloc[1]
    assert not any("price" in column.lower() for column in clock.columns)
    assert s.CONTROLS == (
        "no_btc_variation_gate", "mvrv_direction_flip", "one_day_stale_mvrv",
        "market_cap_only_local_z", "same_clock_forced_long",
    )
    primary_side = s.active(panel)[1]
    assert s.active(panel, "mvrv_direction_flip")[1].equals(-primary_side)
    assert s.active(panel, "same_clock_forced_long")[1].eq(1).all()
    assert s.active(panel, "market_cap_only_local_z")[0].tolist() == [False, True, False, True]


def test_midrank_is_strict_prior_and_support_gates_are_frozen():
    values = pd.Series(np.arange(122, dtype=float))
    ranked = s.strict_prior_midrank(values)
    assert ranked.iloc[:120].isna().all()
    assert ranked.iloc[120] == 1.0
    assert s.GATES == {
        "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
        "minority_side_share_min": 0.2,
        "max_month_share": 0.45,
    }
