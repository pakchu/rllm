from __future__ import annotations

import gzip
import io
import urllib.parse

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_active_address_network_value_relay_support as support


def raw_row(day: str, *, completion: str, active: str = "300", cap: str = "1500") -> dict[str, str]:
    return {
        "asset": "btc",
        "time": f"{day}T00:00:00.000000000Z",
        "AdrActCnt": active,
        "CapMrktCurUSD": cap,
        "AssetEODCompletionTime": completion,
    }


def test_exact_source_url_and_read_only_query_contract() -> None:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(support.source_url()).query)
    assert query == {
        "assets": ["btc"],
        "metrics": ["AdrActCnt,CapMrktCurUSD,AssetEODCompletionTime"],
        "frequency": ["1d"],
        "start_time": ["2022-01-01"],
        "end_time": ["2026-07-29"],
        "page_size": ["10000"],
    }
    normalized = " ".join(support.BTC_QUERY.split()).lower()
    assert normalized.startswith("select ts,open,close from bars_binance")
    assert not any(column in normalized for column in ("high,", "low,", "volume", "entry_price", "exit_price"))


def test_source_row_strict_schema_values_and_completion_window() -> None:
    parsed = support.parse_source_row(raw_row("2022-01-01", completion="1641081601.5"))
    assert parsed["AdrActCnt"] == 300
    assert parsed["CapMrktCurUSD"] == "1500"
    assert parsed["feature_available_time"] == pd.Timestamp("2022-01-02T00:00:01.500000Z")
    for completion in ("1641081600", "1641124801"):
        with pytest.raises(ValueError, match=r"after D\+1"):
            support.parse_source_row(raw_row("2022-01-01", completion=completion))
    with pytest.raises(ValueError, match="schema drift"):
        support.parse_source_row({**raw_row("2022-01-01", completion="1641081601"), "extra": "x"})
    with pytest.raises(ValueError, match="positive integer"):
        support.parse_source_row(raw_row("2022-01-01", completion="1641081601", active="3.5"))


def test_pagination_rejects_query_drift_and_hashes_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(support, "SOURCE_START", pd.Timestamp("2022-01-01T00:00:00Z"))
    monkeypatch.setattr(support, "SOURCE_END", pd.Timestamp("2022-01-03T00:00:00Z"))
    first = support.source_url()
    next_url = first + "&next_page_token=two"
    pages = {
        first: {
            "data": [raw_row("2022-01-01", completion="1641081601")],
            "next_page_token": "two",
            "next_page_url": next_url,
        },
        next_url: {"data": [raw_row("2022-01-02", completion="1641168001")]},
    }
    frame, audit = support.download_coinmetrics(fetch=pages.__getitem__, sleep=lambda _: None)
    assert list(frame.AdrActCnt) == [300, 300]
    assert audit["response_pages"] == 2
    assert audit["response_chain_sha256"] == support.canonical_hash(audit["response_page_sha256"])
    assert audit["current_vintage_not_historical_revision_archive"] is True
    drifted = next_url.replace("assets=btc", "assets=eth")
    with pytest.raises(ValueError, match="changed the frozen query"):
        support._validate_next_page_url(first, drifted)


def test_strict_prior_midrank_excludes_current_caps_history_and_midranks_ties() -> None:
    values = pd.Series(np.arange(272, dtype=float))
    ranked = support.strict_prior_midrank(values)
    assert ranked.iloc[:180].isna().all()
    assert ranked.iloc[180] == 1.0
    tied = support.strict_prior_midrank(pd.Series([1.0, 1.0, 2.0]), maximum=2, minimum=2)
    assert tied.iloc[2] == 1.0
    reverse = support.strict_prior_midrank(pd.Series([2.0, 2.0, 2.0]), maximum=2, minimum=2)
    assert reverse.iloc[2] == 0.5


def test_panel_computes_aanv30_and_open_close_variation(monkeypatch: pytest.MonkeyPatch) -> None:
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = start + pd.Timedelta(days=31)
    monkeypatch.setattr(support, "SOURCE_START", start)
    monkeypatch.setattr(support, "SOURCE_END", end)
    observations = pd.date_range(start, end, freq="1d", inclusive="left")
    source = pd.DataFrame({
        "observation_time": observations,
        "feature_available_time": observations + pd.Timedelta(days=1, seconds=1),
        "AdrActCnt": np.arange(1, 32),
        "CapMrktCurUSD": 10.0,
    })
    minute_index = pd.date_range(start + pd.Timedelta(hours=12), end + pd.Timedelta(hours=12), freq="1min", inclusive="left")
    bars = pd.DataFrame({"ts": minute_index, "open": 100.0, "close": 100.0 * np.exp(0.01)})
    panel = support.build_panel(source, bars)
    assert np.isnan(panel.aanv30.iloc[28])
    assert panel.active_addresses_30d.iloc[29] == 15.5
    assert panel.aanv30.iloc[29] == 1.55
    assert panel.minute_count.iloc[29] == 1440
    assert panel.btc_variation.iloc[29] == pytest.approx(np.sqrt(1440 * 0.01**2))
    assert bool(panel.source_valid.iloc[29])


def clock_panel() -> pd.DataFrame:
    observations = pd.date_range("2024-01-01T00:00:00Z", periods=4, freq="1d")
    return pd.DataFrame({
        "observation_time": observations,
        "feature_available_time": observations + pd.Timedelta(days=1, hours=1),
        "decision_time": observations + pd.Timedelta(days=1, hours=12),
        "source_valid": True,
        "minute_count": 1440,
        "AdrActCnt": 100.0,
        "CapMrktCurUSD": 1000.0,
        "active_addresses_30d": 100.0,
        "aanv30": [0.1, 0.2, 0.3, 0.4],
        "aanv30_rank": [0.8, 0.5, 0.2, 0.9],
        "single_day_aanv": [0.1, 0.2, 0.3, 0.4],
        "single_day_aanv_rank": [0.1, 0.9, 0.5, 0.1],
        "btc_variation": 1.0,
        "btc_variation_rank": [0.65, 0.9, 0.9, 0.9],
    })


def test_clock_uses_tail_side_fixed_time_split_and_half_open_reservation() -> None:
    clock = support.build_clock(clock_panel())
    assert list(clock.side) == [1, -1, 1]
    assert (clock.entry_time - clock.decision_time).eq(pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time - clock.entry_time).eq(pd.Timedelta(hours=24)).all()
    assert clock.entry_time.iloc[2] == clock.exit_time.iloc[1]
    assert set(clock.split) == {"test"}
    assert not any("price" in column.lower() for column in clock.columns)


def test_exact_five_controls_and_support_gates() -> None:
    assert support.CONTROLS == (
        "no_btc_volatility_gate", "aanv_direction_flip", "one_day_stale_aanv",
        "single_day_active_address_to_value", "same_clock_forced_long",
    )
    panel = clock_panel()
    panel.loc[0, "btc_variation_rank"] = 0.1
    assert not support.active(panel)[0].iloc[0]
    assert support.active(panel, "no_btc_volatility_gate")[0].iloc[0]
    primary_side = support.active(panel)[1]
    assert support.active(panel, "aanv_direction_flip")[1].equals(-primary_side)
    assert support.active(panel, "same_clock_forced_long")[1].eq(1).all()
    single_active = support.active(panel, "single_day_active_address_to_value")[0]
    assert single_active.tolist() == [False, True, False, True]
    assert support.GATES == {
        "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
        "minority_side_share_min": 0.2,
        "max_month_share": 0.45,
    }


def test_artifact_encodings_are_deterministic_and_immutable(tmp_path) -> None:
    frame = pd.DataFrame({"x": [1.25], "y": ["a"]})
    first = support.deterministic_csv_gzip(frame)
    second = support.deterministic_csv_gzip(frame)
    assert first == second
    assert gzip.GzipFile(fileobj=io.BytesIO(first)).read() == b"x,y\n1.25,a\n"
    path = tmp_path / "artifact.csv.gz"
    support.write_immutable(path, first)
    support.write_immutable(path, first)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        support.write_immutable(path, b"different")
