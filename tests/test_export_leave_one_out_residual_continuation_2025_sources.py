from __future__ import annotations

import pandas as pd
import pytest

from training.export_leave_one_out_residual_continuation_2025_sources import (
    END,
    EXPECTED_PROTOCOL_HASH,
    HOLDOUT_START,
    START,
)
from training.export_leave_one_out_residual_exhaustion_sources import validate_market
from training.preregister_leave_one_out_residual_continuation import canonical_hash, protocol


def market_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({
        "date": dates,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1.0,
        "quote_asset_volume": 100.0,
        "number_of_trades": 1,
        "taker_buy_base": 0.5,
        "taker_buy_quote": 50.0,
        "tic": "ETHUSDT",
        "day": 1,
    })


def test_protocol_hash_and_physical_boundaries_are_frozen() -> None:
    assert canonical_hash(protocol()) == EXPECTED_PROTOCOL_HASH
    assert START == pd.Timestamp("2024-01-01")
    assert HOLDOUT_START == pd.Timestamp("2025-01-01")
    assert END == pd.Timestamp("2026-01-01")


def test_validation_excludes_2026_rows_when_grid_check_is_disabled() -> None:
    dates = pd.DatetimeIndex(["2025-12-31 23:55", "2026-01-01 00:00"])
    out = validate_market(market_frame(dates), "ETHUSDT", START, END, exact_grid=False)
    assert out["date"].max() == pd.Timestamp("2025-12-31 23:55")
    assert (out["date"] < END).all()


def test_validation_rejects_missing_exact_grid() -> None:
    dates = pd.DatetimeIndex([START, START + pd.Timedelta(minutes=10)])
    with pytest.raises(ValueError, match="market grid mismatch"):
        validate_market(market_frame(dates), "ETHUSDT", START, START + pd.Timedelta(minutes=15))
