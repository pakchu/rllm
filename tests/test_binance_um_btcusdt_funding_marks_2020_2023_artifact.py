from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training import freeze_binance_um_btcusdt_funding_marks_2020_2023 as freeze


DATA = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
EXPECTED_DATA_SHA256 = "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
EXPECTED_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)
EXPECTED_MANIFEST_HASH = (
    "3b447e94d9dbb6ba4994713df565b7d6ec5b38c26c4b568ad7f4e102fefc299c"
)


def test_funding_mark_artifacts_are_frozen_before_strategy_outcomes() -> None:
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == EXPECTED_DATA_SHA256
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA256
    payload = json.loads(MANIFEST.read_text())
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    assert freeze._canonical_hash(core) == payload["manifest_hash"]
    assert payload["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert payload["outcomes_opened"] is False
    assert payload["strategy_outcomes_calculated"] == []
    assert payload["sealed"] == [
        "all_strategy_returns",
        "2024",
        "2025",
        "2026_ytd",
    ]


def test_funding_mark_source_is_complete_and_low_error() -> None:
    payload = json.loads(MANIFEST.read_text())
    assert payload["data"]["rows"] == 4_383
    assert payload["quality"] == {
        "events": 4_383,
        "recorded_mark_overlap_events": 185,
        "backfilled_events": 4_198,
        "maximum_funding_time_offset_ms": 47,
        "maximum_recorded_vs_8h_open_mark_error_bp": 13.484319911147846,
        "maximum_proxy_funding_cash_error_bp_notional": 0.0013484319911147846,
    }
    assert (
        payload["quality"]["maximum_proxy_funding_cash_error_bp_notional"]
        <= freeze.MAXIMUM_PROXY_FUNDING_CASH_ERROR_BP_NOTIONAL
    )


def test_every_funding_event_has_a_positive_frozen_settlement_mark() -> None:
    frame = pd.read_csv(DATA)
    assert len(frame) == 4_383
    assert frame["funding_time_ms"].is_monotonic_increasing
    assert not frame["funding_time_ms"].duplicated().any()
    assert frame["mark_source"].eq("binance_8h_mark_price_kline_open").all()
    marks = frame["settlement_mark_price"].to_numpy(float)
    rates = frame["funding_rate"].to_numpy(float)
    assert np.isfinite(marks).all() and (marks > 0.0).all()
    assert np.isfinite(rates).all()
    assert frame["funding_time_offset_ms"].between(0, 47).all()
    assert pd.to_datetime(frame["funding_time_utc"], utc=True).max() < pd.Timestamp(
        "2024-01-01", tz="UTC"
    )
