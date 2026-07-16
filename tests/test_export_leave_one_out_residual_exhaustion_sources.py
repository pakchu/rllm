from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from training.export_leave_one_out_residual_exhaustion_sources import (
    deterministic_csv_gz,
    validate_funding,
    validate_market,
)


def market_frame(symbol: str = "ETHUSDT") -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=12, freq="5min")
    close = np.linspace(100.0, 101.1, len(dates))
    return pd.DataFrame({
        "date": dates,
        "open": close - 0.05,
        "high": close + 0.1,
        "low": close - 0.1,
        "close": close,
        "volume": 10.0,
        "quote_asset_volume": 1000.0,
        "number_of_trades": 100,
        "taker_buy_base": 5.0,
        "taker_buy_quote": 500.0,
        "tic": symbol,
        "day": 6,
    })


def test_market_validation_enforces_identity_and_exact_grid() -> None:
    frame = market_frame()
    got = validate_market(
        frame,
        "ETHUSDT",
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-01-01 01:00"),
    )
    assert len(got) == 12
    bad = frame.drop(index=3).reset_index(drop=True)
    with pytest.raises(ValueError, match="grid mismatch"):
        validate_market(
            bad,
            "ETHUSDT",
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-01 01:00"),
        )


def test_market_validation_rejects_bad_ohlc_and_taker_volume() -> None:
    bad_high = market_frame()
    bad_high.loc[0, "high"] = bad_high.loc[0, "close"] - 1
    with pytest.raises(ValueError, match="high below"):
        validate_market(bad_high, "ETHUSDT", exact_grid=False)
    bad_buy = market_frame()
    bad_buy.loc[0, "taker_buy_quote"] = 2000
    with pytest.raises(ValueError, match="taker buy"):
        validate_market(bad_buy, "ETHUSDT", exact_grid=False)


def test_funding_validation_uses_exact_event_time_and_cutoff() -> None:
    times = pd.to_datetime(
        ["2023-01-01", "2023-01-01 08:00:00.008", "2025-01-01"],
        format="mixed",
    )
    frame = pd.DataFrame({
        "date": times,
        "symbol": "ETHUSDT",
        "funding_rate": [0.0001, -0.0002, 0.0003],
        "funding_time": (times.view("int64") // 1_000_000).astype("int64"),
        "mark_price": [np.nan, np.nan, np.nan],
    })
    got = validate_funding(frame, "ETHUSDT")
    assert len(got) == 2
    assert got["funding_time"].max() < int(pd.Timestamp("2025-01-01").timestamp() * 1000)


def test_deterministic_gzip_has_stable_hash(tmp_path) -> None:
    frame = market_frame()
    one, two = tmp_path / "one.csv.gz", tmp_path / "two.csv.gz"
    deterministic_csv_gz(frame, one)
    deterministic_csv_gz(frame, two)
    assert hashlib.sha256(one.read_bytes()).hexdigest() == hashlib.sha256(two.read_bytes()).hexdigest()
