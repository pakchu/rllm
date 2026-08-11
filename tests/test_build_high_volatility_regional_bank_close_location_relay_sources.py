import json
import pandas as pd
from training import build_high_volatility_regional_bank_close_location_relay_sources as builder


def payload() -> bytes:
    return json.dumps({"chart": {"error": None, "result": [{"meta": {"symbol": "KRE", "exchangeTimezoneName": "America/New_York"}, "timestamp": [1662129000, 1662474600], "indicators": {"quote": [{"open": [10.0, 11.0], "high": [11.0, 11.5], "low": [9.5, 10.5], "close": [10.5, 10.75]}]}}]}}).encode()


def test_parse_uses_raw_ohlc_only():
    frame, metadata = builder.parse_payload(payload())
    assert list(frame.columns) == ["session_date", "open", "high", "low", "close"]
    assert frame.session_date.tolist() == [pd.Timestamp("2022-09-02"), pd.Timestamp("2022-09-06")]
    assert metadata["adjusted_close_read"] is False


def test_deterministic_gzip(tmp_path):
    frame, _ = builder.parse_payload(payload())
    first, second = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    builder.deterministic_gzip_csv(frame, first)
    builder.deterministic_gzip_csv(frame, second)
    assert first.read_bytes() == second.read_bytes()
