from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


MANIFEST = Path(
    "results/binance_coinm_liquidation_snapshot_btc_2023_2024_manifest.json"
)
EXPECTED_MANIFEST_SHA256 = (
    "5d78686e7c40d69261f09bc77e27ff734f682abba4abb95c2291e8282380053e"
)
EXPECTED_DATA_SHA256 = (
    "a23b93d8567a589e9f045ae4a56393e493a8da2748c5a051804c9bdf9388ccc3"
)
MISSING_DATES = {
    "2023-09-09",
    "2023-09-23",
    "2023-09-25",
    "2024-06-01",
    "2024-06-11",
    "2024-06-12",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_manifest_and_panel_are_self_consistent() -> None:
    assert _sha256(MANIFEST) == EXPECTED_MANIFEST_SHA256
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["protocol"]["outcomes_opened"] is False
    assert manifest["protocol"]["source_only"] is True
    assert manifest["contract_size_usd"] == 100.0
    assert set(manifest["missing_archive_dates"]) == MISSING_DATES
    assert len(manifest["archives"]) == 478
    available = [item for item in manifest["archives"] if item["available"]]
    assert len(available) == 472
    assert sum(item["snapshot_rows"] for item in available) == 53_398
    assert sum(item["raw_rows"] for item in available) == 106_822
    assert sum(item["duplicate_rows_removed"] for item in available) == 53_424

    ordinary = [item for item in available if item["date"] != "2023-09-21"]
    assert all(item["raw_rows"] == 2 * item["snapshot_rows"] for item in ordinary)
    anomaly = next(item for item in available if item["date"] == "2023-09-21")
    assert (anomaly["raw_rows"], anomaly["snapshot_rows"], anomaly["duplicate_rows_removed"]) == (
        196,
        85,
        111,
    )

    data_path = Path(manifest["file"]["path"])
    assert _sha256(data_path) == EXPECTED_DATA_SHA256 == manifest["file"]["sha256"]
    frame = pd.read_csv(
        data_path,
        compression="gzip",
        parse_dates=["date", "feature_available_time"],
    )
    assert len(frame) == manifest["file"]["rows"] == 137_664
    expected = pd.Series(
        pd.date_range("2023-06-25", "2024-10-15", freq="5min", inclusive="left"),
        name="date",
    )
    assert frame["date"].equals(expected)
    assert frame["date"].duplicated().sum() == 0
    availability_delay = cast(
        pd.Series, frame["feature_available_time"].sub(frame["date"])
    )
    assert bool(availability_delay.eq(pd.Timedelta(minutes=5, seconds=1)).all())
    assert int(frame["source_valid"].sum()) == 135_936
    assert int(frame["event_count"].fillna(0).sum()) == 53_398
    assert int(frame["event_count"].fillna(0).gt(0).sum()) == 18_897

    missing_mask = cast(
        pd.Series, frame["date"].dt.strftime("%Y-%m-%d").isin(MISSING_DATES)
    )
    missing = frame.loc[missing_mask]
    assert len(missing) == len(MISSING_DATES) * 288
    assert not bool(cast(pd.Series, missing["source_valid"]).any())
    assert bool(cast(pd.Series, missing["event_count"]).isna().all())
    valid_mask = cast(pd.Series, frame["source_valid"]).astype(bool)
    valid = frame.loc[valid_mask]
    assert bool(cast(pd.Series, valid["event_count"]).notna().all())
    assert np.allclose(
        valid["short_liquidation_usd"] + valid["long_liquidation_usd"],
        valid["total_liquidation_usd"],
    )
    assert np.allclose(
        valid["short_liquidation_usd"] - valid["long_liquidation_usd"],
        valid["signed_liquidation_usd"],
    )
    assert np.allclose(
        np.mod(
            cast(pd.Series, valid["total_liquidation_usd"]).to_numpy(float), 100.0
        ),
        0.0,
    )
