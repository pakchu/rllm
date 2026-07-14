from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from training import build_binance_um_aux_btc_2021_2023 as builder


def test_normalize_funding_time_accepts_millisecond_jitter() -> None:
    values = pd.Series([1609459200002, 1609488000047])
    normalized, jitter = builder.normalize_funding_time(
        values,
        maximum_jitter_ms=1_000,
    )
    assert normalized.tolist() == [
        pd.Timestamp("2021-01-01 00:00"),
        pd.Timestamp("2021-01-01 08:00"),
    ]
    assert jitter.tolist() == [2.0, 47.0]


def test_normalize_funding_time_rejects_large_jitter() -> None:
    values = pd.Series([1609459202000])
    with pytest.raises(ValueError, match="jitter exceeds"):
        builder.normalize_funding_time(values, maximum_jitter_ms=1_000)


def test_frozen_manifest_matches_physical_pre2024_files() -> None:
    manifest = json.loads(
        Path("results/binance_um_aux_btc_2021_2023_manifest.json").read_text()
    )
    assert manifest["protocol"]["outcomes_opened"] is False
    assert manifest["protocol"]["post_2023_rows_written"] is False
    assert manifest["files"]["funding"]["rows"] == 3_285
    assert manifest["files"]["premium"]["rows"] == 26_280
    for item in manifest["files"].values():
        path = Path(item["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
