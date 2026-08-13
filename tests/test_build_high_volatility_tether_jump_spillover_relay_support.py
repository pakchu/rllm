from __future__ import annotations

import gzip
import io
import json
import urllib.parse

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_tether_jump_spillover_relay_support as support


def raw_row(day: str, completion: str, price: str = "1.00025") -> dict[str, str]:
    return {
        "asset": "usdt",
        "time": f"{day}T00:00:00.000000000Z",
        "PriceUSD": price,
        "AssetEODCompletionTime": completion,
        "PriceUSD-status": "reviewed",
        "PriceUSD-status-time": f"{day}T00:00:00.000000000Z",
    }


def test_exact_source_url_and_read_only_btc_contract() -> None:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(support.source_url()).query)
    assert query == {
        "assets": ["usdt"],
        "metrics": ["PriceUSD,AssetEODCompletionTime"],
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


def test_source_row_strict_schema_exact_price_and_completion_timestamp() -> None:
    parsed = support.parse_source_row(raw_row("2022-01-01", "1641081601.5"))
    assert parsed["PriceUSD"] == "1.00025"
    assert parsed["feature_available_time"] == pd.Timestamp("2022-01-02T00:00:01.500000Z")
    with pytest.raises(ValueError, match="schema drift"):
        support.parse_source_row({**raw_row("2022-01-01", "1641081601"), "extra": "x"})
    with pytest.raises(ValueError, match="non-empty string"):
        support.parse_source_row({**raw_row("2022-01-01", "1641081601"), "PriceUSD-status": ""})
    with pytest.raises(ValueError, match="exact ISO-8601"):
        support.parse_source_row({
            **raw_row("2022-01-01", "1641081601"),
            "PriceUSD-status-time": "not-a-time",
        })
    with pytest.raises(ValueError, match="positive and finite"):
        support.parse_source_row(raw_row("2022-01-01", "1641081601", "0"))
    with pytest.raises(ValueError, match="sub-microsecond"):
        support.parse_source_row(raw_row("2022-01-01", "1641081601.0000001"))


def test_pagination_contract_and_response_chain_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert frame.PriceUSD.tolist() == ["1.00025", "1.00025"]
    assert audit["response_pages"] == 2
    assert audit["response_chain_sha256"] == support.canonical_hash(
        audit["response_page_sha256"]
    )
    with pytest.raises(ValueError, match="changed the frozen query"):
        support._validate_next_page_url(first, next_url.replace("assets=usdt", "assets=btc"))


def _flat_bars(start: pd.Timestamp, end: pd.Timestamp, return_: float = 0.001) -> pd.DataFrame:
    minutes = pd.date_range(start, end, freq="1min", inclusive="left")
    return pd.DataFrame({"ts": minutes, "open": 100.0, "close": 100.0 * np.exp(return_)})


def test_panel_uses_completion_clock_returns_absolute_jump_rank_and_exact_btc_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = start + pd.Timedelta(days=4)
    monkeypatch.setattr(support, "SOURCE_START", start)
    monkeypatch.setattr(support, "SOURCE_END", end)
    monkeypatch.setattr(
        support,
        "strict_prior_midrank",
        lambda values: pd.Series([np.nan, 0.8, 0.7, 0.9], index=values.index),
    )
    observations = pd.date_range(start, end, freq="1d", inclusive="left")
    available = pd.to_datetime([
        "2022-01-02T00:00:01Z",
        "2022-01-03T01:02:03Z",
        "2022-01-04T23:59:59Z",
        "2022-01-05T12:00:00Z",
    ])
    source = pd.DataFrame({
        "observation_time": observations,
        "feature_available_time": available,
        "PriceUSD": [1.0, 1.1, 0.99, 1.089],
    })
    bars = _flat_bars(start, end + pd.Timedelta(days=2))

    panel = support.build_panel(source, bars)

    assert panel.decision_time.tolist() == list(pd.to_datetime([
        "2022-01-02T00:05:00Z",
        "2022-01-03T01:05:00Z",
        "2022-01-05T00:00:00Z",
        "2022-01-05T12:00:00Z",
    ]))
    assert panel.source_valid.tolist() == [False, True, True, True]
    assert panel.usdt_log_return.iloc[1] == pytest.approx(np.log(1.1))
    assert panel.usdt_log_return.iloc[2] == pytest.approx(np.log(0.9))
    assert panel.minute_count.iloc[1] == 1440
    assert panel.btc_variation.iloc[1] == pytest.approx(np.sqrt(1440 * 0.001**2))
    assert panel.jump_rank.tolist()[1:] == [0.8, 0.7, 0.9]

    missing = bars.drop(bars.index[bars.ts.eq(pd.Timestamp("2022-01-03T01:04:00Z"))])
    missing_panel = support.build_panel(source, missing)
    assert not bool(missing_panel.source_valid.iloc[1])
    assert missing_panel.minute_count.iloc[1] == 1439


def test_panel_rejects_completion_boundaries_invalid_prior_zero_return_and_grid_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = start + pd.Timedelta(days=5)
    monkeypatch.setattr(support, "SOURCE_START", start)
    monkeypatch.setattr(support, "SOURCE_END", end)
    observations = pd.date_range(start, end, freq="1d", inclusive="left")
    source = pd.DataFrame({
        "observation_time": observations,
        "feature_available_time": pd.to_datetime([
            "2022-01-02T00:00:00Z",  # D+1 exactly rejects
            "2022-01-04T00:00:00Z",  # D+2 exactly admits, but prior invalid
            "2022-01-05T00:00:01Z",  # after D+2 rejects
            "2022-01-05T00:00:01Z",
            "2022-01-06T00:00:01Z",
        ]),
        "PriceUSD": [1.0, 1.01, 1.02, 1.03, 1.03],
    })
    bars = _flat_bars(start, end + pd.Timedelta(days=2))

    panel = support.build_panel(source, bars)

    assert panel.source_valid.tolist() == [False, False, False, False, False]
    assert panel.usdt_log_return.iloc[:4].isna().all()
    assert pd.isna(panel.usdt_log_return.iloc[4])  # exact zero rejects
    drifted = bars.copy()
    drifted.loc[0, "ts"] += pd.Timedelta(seconds=1)
    with pytest.raises(RuntimeError, match="exact one-minute grid"):
        support.prepare_btc_bars(drifted)


def clock_panel() -> pd.DataFrame:
    observations = pd.date_range("2023-12-31T00:00:00Z", periods=5, freq="1d")
    available = pd.to_datetime([
        "2024-01-01T00:00:01Z",
        "2024-01-02T00:04:00Z",
        "2024-01-03T12:00:00Z",
        "2024-01-04T00:00:01Z",
        "2024-01-05T00:00:01Z",
    ])
    return pd.DataFrame({
        "observation_time": observations,
        "feature_available_time": available,
        "decision_time": available.ceil("5min"),
        "source_valid": True,
        "minute_count": 1440,
        "PriceUSD": [1.0, 0.99, 1.01, 0.98, 1.02],
        "usdt_log_return": [0.1, -0.1, 0.2, -0.2, 0.3],
        "jump_rank": [0.8, 0.9, 0.74, 0.95, 0.99],
        "btc_variation": 1.0,
        "btc_variation_rank": [0.9, 0.8, 0.9, 0.64, 0.95],
    })


def test_clock_contrarian_side_gates_reservation_and_controls() -> None:
    panel = clock_panel()
    clock = support.build_clock(panel)
    assert clock.side.tolist() == [-1, 1, -1]
    assert clock.decision_time.tolist() == panel.decision_time.iloc[[0, 1, 4]].tolist()
    assert (clock.entry_time - clock.decision_time).eq(pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time - clock.entry_time).eq(pd.Timedelta(hours=24)).all()
    assert clock.entry_time.iloc[1] == clock.exit_time.iloc[0]
    assert not any("price" in column.lower() and column != "PriceUSD" for column in clock.columns)
    assert support.CONTROLS == (
        "no_jump_tail", "no_btc_variation_gate", "one_day_stale_usdt_return",
        "direction_flip", "same_clock_forced_long",
    )
    primary_side = support.active(panel)[1]
    assert support.active(panel, "direction_flip")[1].equals(-primary_side)
    assert support.active(panel, "same_clock_forced_long")[1].eq(1).all()
    assert support.active(panel, "no_jump_tail")[0].tolist() == [True, True, True, False, True]
    assert support.active(panel, "no_btc_variation_gate")[0].tolist() == [True, True, False, True, True]


def test_strict_prior_ranks_gates_and_artifact_encodings_are_deterministic() -> None:
    values = pd.Series([*np.arange(180, dtype=float), 179.0, np.nan, 180.0])
    ranked = support.strict_prior_midrank(values)
    assert ranked.iloc[:180].isna().all()
    assert ranked.iloc[180] == pytest.approx((179 + 0.5) / 180)
    assert pd.isna(ranked.iloc[181])
    assert ranked.iloc[182] == 1.0
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
    assert json.loads(encoded) == {"label": "한글"}


def test_preregistration_hash_is_frozen() -> None:
    assert support.sha256_file(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA256
    registration = json.loads(support.prereg.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert registration == support.prereg.build()
    assert registration["manifest_hash"] == support.PREREG_MANIFEST_HASH
