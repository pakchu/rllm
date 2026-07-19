from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


MANIFEST = Path("results/binance_um_premium_path_btc_2020_2026_manifest.json")
DATA = Path(
    "data/binance_um_premium_path_btc_2020_2026/"
    "BTCUSDT_premium_path_1m_2020-01-01_2026-06-30.csv.gz"
)
EXPECTED_MANIFEST_SHA256 = (
    "821e84f2f03bf893a03d7904bf665b6fd7f6d38edd845d1a9c4eef384d1c1dd8"
)
EXPECTED_DATA_SHA256 = (
    "7fbaae1f85482b9fc9e148af357c7315e4e7fc4b4e3ae36c31f27545109f8aa9"
)
EXPECTED_GAPS = {
    "2020-01": 29,
    "2020-12": 106,
    "2021-07": 7_200,
    "2022-07": 50,
    "2022-10": 1_440,
    "2023-02": 1_440,
    "2023-11": 14,
    "2024-08": 2,
    "2026-06": 1_440,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_frozen_manifest_is_outcome_free_and_checksum_complete() -> None:
    assert _sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256
    manifest = json.loads(MANIFEST.read_text())
    protocol = manifest["protocol"]
    assert protocol["source_only"] is True
    assert protocol["outcomes_opened"] is False
    assert protocol["archive_checksums_verified"] is True
    assert protocol["raw_archives_retained"] is False
    assert protocol["btc_execution_prices_retained"] is False
    assert protocol["returns_or_pnl_retained"] is False
    assert len(manifest["archives"]) == 78
    assert manifest["missing_archive_months"] == []
    assert all(item["available"] for item in manifest["archives"])
    gaps = {
        item["month"]: item["missing_rows"]
        for item in manifest["archives"]
        if item["missing_rows"]
    }
    assert gaps == EXPECTED_GAPS
    assert sum(gaps.values()) == 11_721
    assert manifest["file"]["sha256"] == EXPECTED_DATA_SHA256


def test_frozen_panel_is_causal_complete_and_has_no_execution_outcomes() -> None:
    assert _sha256(DATA) == EXPECTED_DATA_SHA256
    frame = pd.read_csv(
        DATA,
        compression="gzip",
        parse_dates=["date", "source_close_time", "feature_available_time"],
    )
    assert len(frame) == 3_417_120
    assert frame["date"].iloc[0] == pd.Timestamp("2020-01-01 00:00")
    assert frame["date"].iloc[-1] == pd.Timestamp("2026-06-30 23:59")
    assert not frame["date"].duplicated().any()
    assert bool(cast(pd.Series, frame["date"].diff().iloc[1:]).eq(pd.Timedelta(minutes=1)).all())
    assert bool(
        cast(pd.Series, frame["source_close_time"] - frame["date"])
        .eq(pd.Timedelta(seconds=59, milliseconds=999))
        .all()
    )
    assert bool(
        cast(pd.Series, frame["feature_available_time"] - frame["date"])
        .eq(pd.Timedelta(minutes=1, seconds=1))
        .all()
    )
    assert int(frame["source_valid"].sum()) == 3_405_399

    premium = ["premium_open", "premium_high", "premium_low", "premium_close"]
    valid = cast(pd.Series, frame["source_valid"]).astype(bool)
    assert bool(frame.loc[valid, premium].notna().all().all())
    assert bool(frame.loc[~valid, premium].isna().all().all())
    assert np.isfinite(frame.loc[valid, premium].to_numpy(float)).all()
    assert bool(
        frame.loc[valid, "premium_high"]
        .ge(frame.loc[valid, ["premium_open", "premium_close"]].max(axis=1))
        .all()
    )
    assert bool(
        frame.loc[valid, "premium_low"]
        .le(frame.loc[valid, ["premium_open", "premium_close"]].min(axis=1))
        .all()
    )
    assert not any(
        forbidden in column
        for column in frame.columns
        for forbidden in ("btc_open", "btc_close", "return", "pnl", "funding")
    )

    transition = frame.loc[frame["date"].eq(pd.Timestamp("2025-01-01 00:00"))]
    assert len(transition) == 1
    assert transition["source_close_time"].iloc[0] == pd.Timestamp(
        "2025-01-01 00:00:59.999"
    )
