from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from training import build_binance_usdt_collateral_breadth_source as builder


PANEL = Path(
    "data/binance_usdt_collateral_breadth_2023/"
    "stablecoin_usdt_breadth_1h_2023-08-01T00_2023-12-31T23.csv.gz"
)
MANIFEST = Path("data/binance_usdt_collateral_breadth_2023/build_manifest.json")
EXPECTED_PANEL_SHA256 = (
    "e96fae39c869f6db0dc30bccc5b2fa72f5e7f717c2528038afede18dd5b9892d"
)
EXPECTED_MANIFEST_SHA256 = (
    "26e142b818306275d48690711b7adca00b43750041d104e5c27a65b355c424f2"
)
EXPECTED_BUILDER_SHA256 = (
    "a962ae5c774a837da481403cba2a6061f93bbdcc25d08451fb487e3c42f09ef7"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ucbr_source_artifacts_are_hash_bound_and_outcome_blind() -> None:
    assert _sha256(PANEL) == EXPECTED_PANEL_SHA256
    assert _sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256
    assert _sha256(builder.BUILDER_PATH) == EXPECTED_BUILDER_SHA256
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["combined_sha256"] == EXPECTED_PANEL_SHA256
    assert manifest["builder_sha256"] == EXPECTED_BUILDER_SHA256
    assert len(manifest["archives"]) == 20
    assert {
        (item["symbol"], item["month"]): item["archive_sha256"]
        for item in manifest["archives"]
    } == builder.EXPECTED_ARCHIVE_SHA256
    protocol = manifest["protocol"]
    assert protocol["outcomes_opened"] is False
    assert protocol["btc_prices_opened"] is False
    assert protocol["perpetual_ohlc_or_funding_opened"] is False
    assert protocol["future_returns_labels_or_pnl_opened"] is False
    assert protocol["post_2023_rows_requested"] is False
    assert protocol["raw_ohlc_retained"] is False
    assert protocol["volume_trade_count_or_taker_flow_retained"] is False


def test_ucbr_source_grid_and_validity_are_exact() -> None:
    frame = pd.read_csv(PANEL, parse_dates=["date", "source_available_at"])
    expected = pd.date_range(
        "2023-08-01", "2024-01-01", freq="1h", inclusive="left"
    )
    assert tuple(frame.columns) == builder.OUTPUT_COLUMNS
    assert frame["date"].equals(pd.Series(expected, name="date"))
    assert frame["source_available_at"].equals(
        pd.Series(expected + pd.Timedelta("1h"), name="source_available_at")
    )
    assert len(frame) == 3_672
    assert frame["source_complete"].all()
    assert frame["valid_breadth"].value_counts().sort_index().to_dict() == {
        3: 46,
        4: 3_626,
    }
    assert {
        column: int(frame[column].astype(bool).sum())
        for column in builder.VALID_COLUMNS
    } == {
        "usdcusdt_valid": 3_672,
        "tusdusdt_valid": 3_672,
        "usdpusdt_valid": 3_626,
        "fdusdusdt_valid": 3_672,
    }


def test_ucbr_source_schema_contains_no_raw_market_or_outcome_fields() -> None:
    frame = pd.read_csv(PANEL, nrows=1)
    forbidden_exact = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_notional",
        "trade_count",
        "taker_buy",
        "btc_price",
        "funding",
        "return",
        "label",
        "pnl",
        "cagr",
        "drawdown",
    }
    assert forbidden_exact.isdisjoint(column.lower() for column in frame.columns)
