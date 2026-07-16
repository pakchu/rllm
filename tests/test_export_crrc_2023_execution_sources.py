from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import export_crrc_2023_execution_sources as export


def test_market_validator_requires_exact_2023_grid_and_valid_ohlc() -> None:
    dates = pd.date_range(export.START, export.END - pd.Timedelta(minutes=5), freq="5min")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
        }
    )
    result = export.validate_market_2023(frame)
    assert len(result) == export.MARKET_ROWS
    broken = frame.copy()
    broken.loc[0, "high"] = 99.0
    with pytest.raises(ValueError, match="high"):
        export.validate_market_2023(broken)


def test_funding_validator_preserves_exact_millisecond_jitter() -> None:
    expected = pd.date_range(
        export.START, export.END - pd.Timedelta(hours=8), freq="8h"
    )
    millis = expected.astype("int64") // 1_000_000
    millis = millis.to_numpy(copy=True)
    millis[1] += 8
    frame = pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "funding_rate": np.zeros(export.FUNDING_ROWS),
            "funding_time": millis,
        }
    )
    result = export.validate_funding_2023(frame)
    assert result.loc[1, "event_time"].microsecond == 8_000
    assert result.loc[1, "event_time"] != expected[1]


def test_funding_validator_rejects_shift_over_one_second() -> None:
    expected = pd.date_range(
        export.START, export.END - pd.Timedelta(hours=8), freq="8h"
    )
    millis = (expected.astype("int64") // 1_000_000).to_numpy(copy=True)
    millis[10] += 1_001
    frame = pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "funding_rate": 0.0,
            "funding_time": millis,
        }
    )
    with pytest.raises(ValueError, match="one second"):
        export.validate_funding_2023(frame)


def test_source_contract_is_physically_pre2024_and_outcome_blind() -> None:
    assert export.MARKET_SOURCE.name.endswith("2023-12-31.csv.gz")
    assert export.FUNDING_SOURCE.name.endswith("2023-12-31.csv.gz")
    assert export.MARKET_ROWS == 105_120
    assert export.FUNDING_ROWS == 1_095
    source = open(export.__file__).read()
    for forbidden in ("equity_stats(", "strict_mdd", "cagr_pct", "gross_return"):
        assert forbidden not in source
