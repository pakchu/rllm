from __future__ import annotations

import gzip
import io
import json
import urllib.parse

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_aggregate_spot_volume_confirmation_relay_support as support


def raw_row(
    day: str, completion: str, volume: str = "2500000.5"
) -> dict[str, str]:
    return {
        "asset": "btc",
        "time": f"{day}T00:00:00.000000000Z",
        "volume_reported_spot_usd_1d": volume,
        "AssetEODCompletionTime": completion,
    }


def test_exact_source_url_minimal_schema_and_read_only_btc_contract() -> None:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(support.source_url()).query)
    assert query == {
        "assets": ["btc"],
        "metrics": ["volume_reported_spot_usd_1d,AssetEODCompletionTime"],
        "frequency": ["1d"],
        "start_time": ["2022-01-01"],
        "end_time": ["2026-07-29"],
        "page_size": ["10000"],
    }
    assert support.RAW_ROW_KEYS == {
        "asset", "time", "volume_reported_spot_usd_1d", "AssetEODCompletionTime",
    }
    normalized = " ".join(support.BTC_QUERY.split()).lower()
    assert normalized.startswith("select ts,open,close from bars_binance")
    assert "symbol='btcusdt' and interval='1m'" in normalized
    assert not any(
        forbidden in normalized
        for forbidden in (
            "high,", "low,", "volume", "funding", "pnl", "gross9",
            "entry_price", "exit_price",
        )
    )
    assert support.BTC_QUERY_END == support.SOURCE_END + pd.Timedelta(hours=12, minutes=1)


def test_source_row_freezes_minimal_schema_exact_value_and_completion_timestamp() -> None:
    parsed = support.parse_source_row(raw_row("2022-01-01", "1641081601.5"))
    assert parsed["volume_reported_spot_usd_1d"] == "2500000.5"
    assert parsed["feature_available_time"] == pd.Timestamp("2022-01-02T00:00:01.500000Z")
    with pytest.raises(ValueError, match="schema drift"):
        support.parse_source_row({**raw_row("2022-01-01", "1641081601"), "extra": "x"})
    with pytest.raises(ValueError, match="schema drift"):
        support.parse_source_row({
            **raw_row("2022-01-01", "1641081601"),
            "volume_reported_spot_usd_1d-status": "reviewed",
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
    frame, audit = support.download_coinmetrics(
        fetch=pages.__getitem__, sleep=lambda _: None
    )
    assert frame.volume_reported_spot_usd_1d.tolist() == ["2500000.5", "2500000.5"]
    assert audit["response_pages"] == 2
    assert audit["response_chain_sha256"] == support.canonical_hash(
        audit["response_page_sha256"]
    )
    with pytest.raises(ValueError, match="changed the frozen query"):
        support._validate_next_page_url(first, next_url.replace("assets=btc", "assets=eth"))


def test_strict_prior_volume_and_variation_ranks_are_causal_capped_and_midranks() -> None:
    ranked = support.strict_prior_midrank(pd.Series(np.arange(272, dtype=float)))
    assert ranked.iloc[:180].isna().all()
    assert ranked.iloc[180] == 1.0
    assert ranked.iloc[271] == 1.0
    tied = support.strict_prior_midrank(
        pd.Series([1.0, 1.0, 2.0]), maximum=2, minimum=2
    )
    assert tied.iloc[2] == 1.0
    midrank = support.strict_prior_midrank(
        pd.Series([2.0, 2.0, 2.0]), maximum=2, minimum=2
    )
    assert midrank.iloc[2] == 0.5


def _single_day_panel(
    monkeypatch: pytest.MonkeyPatch,
    completion: str,
    *, remove_decision_open: bool = False,
    flat_open: bool = False,
) -> pd.DataFrame:
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = start + pd.Timedelta(days=1)
    monkeypatch.setattr(support, "SOURCE_START", start)
    monkeypatch.setattr(support, "SOURCE_END", end)
    available = pd.Timestamp(completion)
    decision = available.ceil("5min")
    source = pd.DataFrame({
        "observation_time": [start],
        "feature_available_time": [available],
        "volume_reported_spot_usd_1d": [100.0],
    })
    minutes = pd.date_range(
        decision - pd.Timedelta(hours=24), decision, freq="1min", inclusive="both"
    )
    opens = np.full(len(minutes), 100.0) if flat_open else 100.0 * np.exp(np.arange(len(minutes)) * 0.001)
    bars = pd.DataFrame({"ts": minutes, "open": opens, "close": opens * np.exp(0.002)})
    if remove_decision_open:
        bars = bars.iloc[:-1]
    return support.build_panel(source, bars)


def test_panel_uses_exact_decision_endpoint_open_return_and_variation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _single_day_panel(monkeypatch, "2022-01-02T00:01:00Z")
    assert panel.decision_time.iloc[0] == pd.Timestamp("2022-01-02T00:05:00Z")
    assert panel.minute_count.iloc[0] == 1440
    assert panel.btc_return.iloc[0] == pytest.approx(1.44)
    assert panel.btc_variation.iloc[0] == pytest.approx(np.sqrt(1440 * 0.002**2))
    assert bool(panel.source_valid.iloc[0])


def test_panel_enforces_completion_window_endpoint_open_and_nonzero_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at_lower = _single_day_panel(monkeypatch, "2022-01-02T00:00:00Z")
    assert not bool(at_lower.source_valid.iloc[0])
    at_upper = _single_day_panel(monkeypatch, "2022-01-02T12:00:00Z")
    assert bool(at_upper.source_valid.iloc[0])
    late = _single_day_panel(monkeypatch, "2022-01-02T12:00:01Z")
    assert not bool(late.source_valid.iloc[0])
    missing_endpoint = _single_day_panel(
        monkeypatch, "2022-01-02T00:01:00Z", remove_decision_open=True
    )
    assert missing_endpoint.minute_count.iloc[0] == 1440
    assert not bool(missing_endpoint.source_valid.iloc[0])
    zero_return = _single_day_panel(
        monkeypatch, "2022-01-02T00:01:00Z", flat_open=True
    )
    assert zero_return.btc_return.iloc[0] == 0.0
    assert not bool(zero_return.source_valid.iloc[0])


def clock_panel() -> pd.DataFrame:
    observations = pd.date_range("2023-12-31T00:00:00Z", periods=4, freq="1d")
    decisions = observations + pd.Timedelta(days=1, minutes=5)
    return pd.DataFrame({
        "observation_time": observations,
        "feature_available_time": decisions,
        "decision_time": decisions,
        "source_valid": True,
        "minute_count": 1440,
        "volume_reported_spot_usd_1d": [100.0, 200.0, 300.0, 400.0],
        "btc_return": [0.1, -0.1, 0.2, -0.2],
        "volume_rank": [0.8, 0.7, 0.9, 0.95],
        "btc_variation": 1.0,
        "btc_variation_rank": [0.7, 0.8, 0.6, 0.9],
    })


def test_controls_side_clock_reservation_and_source_gates() -> None:
    panel = clock_panel()
    primary, side, _ = support.active(panel)
    assert primary.tolist() == [True, False, False, True]
    assert side.tolist() == [1, -1, 1, -1]
    assert support.active(panel, "no_volume_tail")[0].tolist() == [True, True, False, True]
    assert support.active(panel, "no_btc_variation_gate")[0].tolist() == [True, False, True, True]
    assert support.active(panel, "direction_flip")[1].equals(-side)
    assert support.active(panel, "same_clock_forced_long")[1].eq(1).all()
    stale_eligible, _, stale = support.active(panel, "one_day_stale_volume")
    assert stale_eligible.tolist() == [False, True, False, True]
    assert stale.volume_reported_spot_usd_1d.iloc[1] == 100.0
    assert stale.volume_rank.iloc[1] == 0.8

    reservation_panel = panel.copy()
    reservation_panel.loc[1, "volume_rank"] = 0.8
    clock = support.build_clock(reservation_panel)
    assert clock.side.tolist() == [1, -1, -1]
    assert (clock.entry_time - clock.decision_time).eq(pd.Timedelta(minutes=5)).all()
    assert (clock.exit_time - clock.entry_time).eq(pd.Timedelta(hours=24)).all()
    assert clock.entry_time.iloc[1] == clock.exit_time.iloc[0]
    assert not any("price" in column.lower() for column in clock.columns)
    assert support.CONTROLS == (
        "no_volume_tail", "no_btc_variation_gate", "one_day_stale_volume",
        "direction_flip", "same_clock_forced_long",
    )
    assert support.GATES == {
        "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
        "minority_side_share_min": 0.2,
        "max_month_share": 0.45,
    }


def test_artifacts_are_deterministic_immutable_and_preregistration_is_bound(tmp_path) -> None:
    frame = pd.DataFrame({"x": [1.25], "label": ["한글"]})
    encoded = support.deterministic_csv_gzip(frame)
    assert encoded == support.deterministic_csv_gzip(frame)
    assert gzip.GzipFile(fileobj=io.BytesIO(encoded)).read().decode() == "x,label\n1.25,한글\n"
    path = tmp_path / "artifact.csv.gz"
    support.write_immutable(path, encoded)
    support.write_immutable(path, encoded)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        support.write_immutable(path, b"different")
    payload = support.deterministic_json({"label": "한글"})
    assert json.loads(payload) == {"label": "한글"}

    assert support.sha256_file(support.prereg.DEFAULT_OUTPUT) == support.PREREG_SHA256
    registration = json.loads(support.prereg.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert registration == support.prereg.build()
    assert registration["manifest_hash"] == support.PREREG_MANIFEST_HASH
