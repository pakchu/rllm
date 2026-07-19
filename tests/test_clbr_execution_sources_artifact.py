from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


MANIFEST = Path("results/clbr_execution_sources_2023_2024_manifest.json")
EXPECTED_MANIFEST_SHA256 = (
    "50b86d6ab896a1c913ee83311f416f67392f29f6fb5a143f59f3abc08448d0c6"
)
EXPECTED_FILES = {
    "train": {
        "market": "fa78e344e576ed3d1e911325613bce1465bfc76c259c0a3733cb350e1cdac2e4",
        "funding": "b94daae411b41d447e52dd0490a269ffd28eaf9316ddbea0da8a6a293d7d44ce",
        "market_rows": 32_256,
        "funding_rows": 336,
        "recorded_marks": 0,
        "proxy_marks": 336,
    },
    "test": {
        "market": "3cbc1198ee32b5d77cdfa468bdaf9ed34af346a962b7026c70c70f3ff0ba7af7",
        "funding": "4b16e60417d30592679d41eeac2d08231c0bd37d337a73dbe9b8c0e43d285414",
        "market_rows": 52_704,
        "funding_rows": 549,
        "recorded_marks": 500,
        "proxy_marks": 49,
    },
    "eval": {
        "market": "212a441e2e8213eda528e2cd586853515785f51ee4291ef8bf8f05ae0d6e52f4",
        "funding": "07dc50bbdff43f6704d819bea0ef0e32c5ff93d7072cdb8252753c122bec8fbd",
        "market_rows": 52_704,
        "funding_rows": 549,
        "recorded_marks": 549,
        "proxy_marks": 0,
    },
}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_execution_source_manifest_and_split_files_are_frozen() -> None:
    assert _sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256
    manifest = json.loads(MANIFEST.read_text())
    protocol = manifest["protocol"]
    assert protocol["outcomes_opened"] is False
    assert protocol["strategy_returns_computed"] is False
    assert protocol["clbr_clocks_loaded"] is False
    assert protocol["market_symbol"] == "BTCUSDT"
    assert protocol["market_interval"] == "5m"
    assert len(manifest["market_archives"]) == 478
    assert all(
        item["archive_sha256"] == item["expected_archive_sha256"]
        and item["rows"] == 288
        for item in manifest["market_archives"]
    )
    assert manifest["funding_quality"] == {
        "events": 1434,
        "recorded_mark_overlap_events": 1049,
        "mark_proxy_events": 385,
        "maximum_funding_time_offset_ms": 15,
        "maximum_recorded_mark_error_bp": 13.484319911147846,
        "maximum_funding_cash_error_bp_notional": 0.0013484319911147846,
    }

    total_market = 0
    total_funding = 0
    for split, expected in EXPECTED_FILES.items():
        files = manifest["files"][split]
        market_path = Path(files["market"]["path"])
        funding_path = Path(files["funding"]["path"])
        assert _sha256(market_path) == expected["market"] == files["market"]["sha256"]
        assert _sha256(funding_path) == expected["funding"] == files["funding"]["sha256"]
        market = pd.read_csv(market_path, compression="gzip", parse_dates=["date"])
        funding = pd.read_csv(
            funding_path, compression="gzip", parse_dates=["funding_time"]
        )
        assert len(market) == expected["market_rows"] == files["market"]["rows"]
        assert len(funding) == expected["funding_rows"] == files["funding"]["rows"]
        start = pd.Timestamp(files["start_inclusive"])
        end = pd.Timestamp(files["end_exclusive"])
        expected_grid = pd.Series(
            pd.date_range(start, end, freq="5min", inclusive="left"), name="date"
        )
        assert market["date"].equals(expected_grid)
        assert market["date"].is_unique
        assert bool(
            cast(pd.Series, market["high"])
            .ge(market[["open", "close"]].max(axis=1))
            .all()
        )
        assert bool(
            cast(pd.Series, market["low"])
            .le(market[["open", "close"]].min(axis=1))
            .all()
        )
        assert funding["funding_time"].is_unique
        assert funding["funding_time"].is_monotonic_increasing
        assert cast(pd.Series, funding["funding_time"]).ge(start).all()
        assert cast(pd.Series, funding["funding_time"]).lt(end).all()
        assert np.isfinite(
            funding[["funding_rate", "settlement_mark_price"]].to_numpy(float)
        ).all()
        assert bool(cast(pd.Series, funding["settlement_mark_price"]).gt(0.0).all())
        mark_source = cast(pd.Series, funding["mark_source"])
        assert int(mark_source.eq("funding_history_recorded_mark").sum()) == expected[
            "recorded_marks"
        ]
        assert int(mark_source.eq("binance_8h_mark_price_kline_open").sum()) == expected[
            "proxy_marks"
        ]
        total_market += len(market)
        total_funding += len(funding)
    assert total_market == 137_664
    assert total_funding == 1_434
