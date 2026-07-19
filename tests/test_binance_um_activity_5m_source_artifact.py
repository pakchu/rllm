from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


SOURCE = Path(
    "data/binance_um_activity_5m_2023_2024/"
    "BTCUSDT_5m_activity_2023-06-25_2024-10-15_exclusive.csv.gz"
)
MANIFEST = Path("results/binance_um_activity_5m_2023_2024_manifest.json")
SOURCE_SHA256 = "dde78b3b14ca1689abaacd00e9085a81f63429ee077de7e22d7e108ad4eb697e"
MANIFEST_SHA256 = "ee22fa7facc901b4bac383f753391c527eafadd71e521dcfd9187de6f2d4b493"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_activity_source_artifact() -> None:
    assert _sha256(SOURCE) == SOURCE_SHA256
    assert _sha256(MANIFEST) == MANIFEST_SHA256

    manifest = json.loads(MANIFEST.read_text())
    assert manifest["protocol"]["outcomes_opened"] is False
    assert manifest["protocol"]["activity_only"] is True
    assert manifest["protocol"]["prices_retained"] is False
    assert manifest["protocol"]["returns_pnl_or_signals_included"] is False
    assert manifest["protocol"]["archive_checksums_verified"] is True
    assert manifest["validation"]["complete_5m_grid"] is True
    assert manifest["validation"]["actual_rows"] == 137_664
    assert len(manifest["archives"]) == 478
    assert manifest["file"]["sha256"] == SOURCE_SHA256

    frame = pd.read_csv(SOURCE, compression="gzip")
    assert len(frame) == 137_664
    assert list(frame.columns) == manifest["protocol"]["retained_columns"]
    assert not {"open", "high", "low", "close"}.intersection(frame.columns)
    assert frame["date"].iloc[0] == "2023-06-25 00:00:00"
    assert frame["date"].iloc[-1] == "2024-10-14 23:55:00"
