from __future__ import annotations

import gzip
import io
import json
import urllib.parse

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_exchange_reserve_pressure_relay_support as support


def raw_row(day: str, completion: str, reserve: str = "2500000.5") -> dict[str, str]:
    return {
        "asset": "btc",
        "time": f"{day}T00:00:00.000000000Z",
        "SplyExNtv": reserve,
        "AssetEODCompletionTime": completion,
        "SplyExNtv-status": "reviewed",
        "SplyExNtv-status-time": f"{day}T00:00:00.000000000Z",
    }


def test_exact_source_url_and_read_only_btc_contract() -> None:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(support.source_url()).query)
    assert query == {
        "assets": ["btc"],
        "metrics": ["SplyExNtv,AssetEODCompletionTime"],
        "frequency": ["1d"],
        "start_time": ["2022-01-01"],
        "end_time": ["2026-07-29"],
        "page_size": ["10000"],
    }
    normalized = " ".join(support.BTC_QUERY.split()).lower()
    assert normalized.startswith("select ts,open,close from bars_binance")
    assert "symbol='btcusdt' and interval='1m'" in normalized
    assert not any(
        forbidden in normalized
        for forbidden in (
            "high,", "low,", "volume", "funding", "return", "pnl", "gross9",
            "entry_price", "exit_price",
        )
    )


def test_source_row_strict_schema_exact_value_and_completion_timestamp() -> None:
    parsed = support.parse_source_row(raw_row("2022-01-01", "1641081601.5"))
    assert parsed["SplyExNtv"] == "2500000.5"
    assert parsed["feature_available_time"] == pd.Timestamp("2022-01-02T00:00:01.500000Z")
    with pytest.raises(ValueError, match="schema drift"):
        support.parse_source_row({**raw_row("2022-01-01", "1641081601"), "extra": "x"})
    with pytest.raises(ValueError, match="non-empty string"):
        support.parse_source_row({**raw_row("2022-01-01", "1641081601"), "SplyExNtv-status": ""})
    with pytest.raises(ValueError, match="exact ISO-8601"):
        support.parse_source_row({
            **raw_row("2022-01-01", "1641081601"),
            "SplyExNtv-status-time": "not-a-time",
        })
    with pytest.raises(ValueError, match="positive and finite"):
        support.parse_source_row(raw_row("2022-01-01", "1641081601", "0"))
    with pytest.raises(ValueError, match="sub-microsecond"):
        support.parse_source_row(raw_row("2022-01-01", "1641081601.0000001"))


def test_pagination_contract_and_response_chain_are_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(support, "SOURCE_START", pd.Timestamp("2022-01-01T00:00:00Z"))
    monkeypatch.setattr(support, "SOURCE_END", pd.Timestamp("2022-01-03T00:00:00Z"))
    first = support.source_url()
    next_url = first + "&next_page_token=two"
    pages = {
        first: {
            "data": [raw_row("2022-01-01", "1641081601")],
            "next_page_token": "two",
            "next_page_url": next_url,
        },
        next_url: {"data": [raw_row("2022-01-02", "1641168001")]},
    }
    frame, audit = support.download_coinmetrics(fetch=pages.__getitem__, sleep=lambda _: None)
    assert frame.SplyExNtv.tolist() == ["2500000.5", "2500000.5"]
    assert audit["response_pages"] == 2
    assert audit["response_chain_sha256"] == support.canonical_hash(audit["response_page_sha256"])
    with pytest.raises(ValueError, match="changed the frozen query"):
        support._validate_next_page_url(first, next_url.replace("assets=btc", "assets=eth"))


def test_panel_uses_variable_completion_clock_reserve_change_and_decision_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = start + pd.Timedelta(days=3)
    monkeypatch.setattr(support, "SOURCE_START", start)
    monkeypatch.setattr(support, "SOURCE_END", end)
    observations = pd.date_range(start, end, freq="1d", inclusive="left")
    available = pd.to_datetime([
        "2022-01-02T00:00:01Z",
        "2022-01-03T01:02:03Z",
        "2022-01-04T12:00:00Z",
    ])
    source = pd.DataFrame({
        "observation_time": observations,
        "feature_available_time": available,
        "SplyExNtv": [100.0, 110.0, 99.0],
    })
    minutes = pd.date_range(start, end + pd.Timedelta(hours=12), freq="1min", inclusive="left")
    bars = pd.DataFrame({"ts": minutes, "open": 100.0, "close": 100.0 * np.exp(0.001)})

    panel = support.build_panel(source, bars)

    assert panel.decision_time.tolist() == list(pd.to_datetime([
        "2022-01-02T00:05:00Z",
        "2022-01-03T01:05:00Z",
        "2022-01-04T12:00:00Z",
    ]))
    assert not bool(panel.source_valid.iloc[0])
    assert bool(panel.source_valid.iloc[1])
    assert bool(panel.source_valid.iloc[2])
    assert panel.reserve_log_change.iloc[1] == pytest.approx(np.log(1.1))
    assert panel.reserve_log_change.iloc[2] == pytest.approx(np.log(0.9))
    assert panel.minute_count.iloc[1] == 1440
    assert panel.btc_variation.iloc[1] == pytest.approx(np.sqrt(1440 * 0.001**2))


def test_panel_rejects_boundary_late_and_invalid_prior_rows_without_imputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = start + pd.Timedelta(days=4)
    monkeypatch.setattr(support, "SOURCE_START", start)
    monkeypatch.setattr(support, "SOURCE_END", end)
    observations = pd.date_range(start, end, freq="1d", inclusive="left")
    source = pd.DataFrame({
        "observation_time": observations,
        "feature_available_time": pd.to_datetime([
            "2022-01-02T00:00:00Z",
            "2022-01-03T00:00:01Z",
            "2022-01-04T12:00:01Z",
            "2022-01-05T00:00:01Z",
        ]),
        "SplyExNtv": [100.0, 101.0, 102.0, 103.0],
    })
    minutes = pd.date_range(start, end + pd.Timedelta(hours=12, minutes=5), freq="1min", inclusive="left")
    bars = pd.DataFrame({"ts": minutes, "open": 100.0, "close": 100.1})

    panel = support.build_panel(source, bars)

    assert panel.source_valid.tolist() == [False, False, False, False]
    assert panel.reserve_log_change.isna().all()


def clock_panel() -> pd.DataFrame:
    observations = pd.date_range("2023-12-31T00:00:00Z", periods=4, freq="1d")
    available = pd.to_datetime([
        "2024-01-01T00:00:01Z",
        "2024-01-02T00:04:00Z",
        "2024-01-03T12:00:00Z",
        "2024-01-04T00:00:01Z",
    ])
    return pd.DataFrame({
        "observation_time": observations,
        "feature_available_time": available,
        "decision_time": available.ceil("5min"),
        "source_valid": True,
        "minute_count": 1440,
        "SplyExNtv": [100.0, 90.0, 120.0, 80.0],
        "reserve_log_change": [0.1, -0.1, 0.2, -0.2],
        "reserve_level_rank": [0.8, 0.2, 0.9, 0.1],
        "btc_variation": 1.0,
        "btc_variation_rank": 0.9,
    })


def test_clock_side_variable_entry_half_open_reservation_and_controls() -> None:
    panel = clock_panel()
    clock = support.build_clock(panel)
    assert clock.side.tolist() == [-1, 1, -1]
    assert clock.decision_time.tolist() == panel.decision_time.iloc[:3].tolist()
    assert (clock.entry_time - clock.decision_time).eq(pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time - clock.entry_time).eq(pd.Timedelta(hours=24)).all()
    assert clock.entry_time.iloc[1] == clock.exit_time.iloc[0]
    assert not any("price" in column.lower() for column in clock.columns)
    assert support.CONTROLS == (
        "no_btc_variation_gate", "exchange_reserve_direction_flip",
        "one_day_stale_reserve_change", "exchange_reserve_level_rank",
        "same_clock_forced_long",
    )
    primary_side = support.active(panel)[1]
    assert support.active(panel, "exchange_reserve_direction_flip")[1].equals(-primary_side)
    assert support.active(panel, "same_clock_forced_long")[1].eq(1).all()
    assert support.active(panel, "exchange_reserve_level_rank")[1].tolist() == [-1, 1, -1, 1]


def test_midrank_gates_and_artifact_encodings_are_deterministic() -> None:
    ranked = support.strict_prior_midrank(pd.Series(np.arange(122, dtype=float)))
    assert ranked.iloc[:120].isna().all()
    assert ranked.iloc[120] == 1.0
    assert support.GATES == {
        "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
        "minority_side_share_min": 0.2,
        "max_month_share": 0.45,
    }
    frame = pd.DataFrame({"x": [1.25], "label": ["한글"]})
    first = support.deterministic_csv_gzip(frame)
    assert first == support.deterministic_csv_gzip(frame)
    assert gzip.GzipFile(fileobj=io.BytesIO(first)).read().decode() == "x,label\n1.25,한글\n"
    encoded = support.deterministic_json({"label": "한글"})
    assert encoded == support.deterministic_json({"label": "한글"})
    assert "한글" in encoded.decode()
    assert json.loads(encoded) == {"label": "한글"}


def test_preregistration_hash_is_frozen() -> None:
    assert support.sha256_file(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA256
    registration = json.loads(support.prereg.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert registration == support.prereg.build()
