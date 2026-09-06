import json

import pandas as pd

from training import build_high_volatility_risk_parity_consensus_relay_sources as builder


def _payload(symbol: str) -> bytes:
    return json.dumps({
        "chart": {"error": None, "result": [{
            "meta": {"symbol": symbol, "exchangeTimezoneName": "America/New_York"},
            "timestamp": [1577975400, 1578061800],
            "indicators": {"quote": [{"open": [10.0, 11.0], "close": [10.5, 10.75]}]},
        }]}
    }).encode()


def test_parse_uses_only_raw_open_close():
    frame, metadata = builder.parse_payload(_payload("SPY"), "QQQ")
    assert list(frame.columns) == ["session_date", "open", "close"]
    assert frame["session_date"].tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    assert metadata["adjusted_close_read"] is False


def test_deterministic_gzip(tmp_path):
    frame, _ = builder.parse_payload(_payload("TLT"), "GLD")
    first, second = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    builder.deterministic_gzip_csv(frame, first)
    builder.deterministic_gzip_csv(frame, second)
    assert first.read_bytes() == second.read_bytes()
