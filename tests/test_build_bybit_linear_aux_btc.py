from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from training import build_bybit_linear_aux_btc_2021_2023 as builder


def test_chunk_intervals_are_gapless_and_end_exclusive() -> None:
    start = pd.Timestamp("2021-01-01")
    end = pd.Timestamp("2021-01-11")
    chunks = builder.chunk_intervals(start, end, days=4)
    assert chunks == [
        (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-05")),
        (pd.Timestamp("2021-01-05"), pd.Timestamp("2021-01-09")),
        (pd.Timestamp("2021-01-09"), pd.Timestamp("2021-01-11")),
    ]


def test_parse_funding_rows_rejects_another_symbol() -> None:
    payload = {
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "fundingRate": "0.0001",
                    "fundingRateTimestamp": "1609459200000",
                }
            ]
        }
    }
    rows = builder.parse_funding_rows(payload, symbol="BTCUSDT")
    assert rows[0]["date"] == pd.Timestamp("2021-01-01")
    assert rows[0]["funding_rate"] == 0.0001
    payload["result"]["list"][0]["symbol"] = "ETHUSDT"
    with pytest.raises(ValueError, match="another symbol"):
        builder.parse_funding_rows(payload, symbol="BTCUSDT")


def test_parse_premium_rows_handles_reverse_api_order() -> None:
    payload = {
        "result": {
            "symbol": "BTCUSDT",
            "list": [
                ["1609462800000", "0.1", "0.2", "0.0", "0.15"],
                ["1609459200000", "0.0", "0.1", "-0.1", "0.05"],
            ],
        }
    }
    rows = builder.parse_premium_rows(payload, symbol="BTCUSDT")
    frame = builder._sorted_unique(pd.DataFrame(rows))
    assert frame["date"].tolist() == [
        pd.Timestamp("2021-01-01 00:00"),
        pd.Timestamp("2021-01-01 01:00"),
    ]


def test_validate_premium_requires_complete_hourly_grid() -> None:
    cfg = replace(
        builder.Config(),
        start="2021-01-01",
        end="2021-01-01 03:00",
    )
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2021-01-01 00:00", "2021-01-01 02:00"]
            ),
            "open": [0.0, 0.0],
            "high": [0.1, 0.1],
            "low": [-0.1, -0.1],
            "close": [0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="complete hourly grid"):
        builder._validate_premium(frame, cfg)


def test_deterministic_gzip_has_stable_hash(tmp_path) -> None:
    frame = pd.DataFrame({"date": ["2021-01-01"], "value": [1.0]})
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    first_hash = builder._write_deterministic_gzip(frame, first)
    second_hash = builder._write_deterministic_gzip(frame, second)
    assert first.read_bytes() == second.read_bytes()
    assert first_hash == second_hash
    assert first_hash == hashlib.sha256(first.read_bytes()).hexdigest()


def test_frozen_manifest_matches_local_pre2024_files() -> None:
    manifest = json.loads(
        Path("results/bybit_linear_aux_btc_2021_2023_manifest.json").read_text()
    )
    assert manifest["protocol"]["outcomes_opened"] is False
    assert manifest["protocol"]["post_2023_rows_requested"] is False
    assert manifest["files"]["funding"]["rows"] == 3_285
    assert manifest["files"]["premium"]["rows"] == 26_280
    for item in manifest["files"].values():
        path = Path(item["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
